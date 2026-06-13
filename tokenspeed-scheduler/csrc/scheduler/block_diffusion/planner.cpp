// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Block-diffusion planner policy: scheduling decisions and plan-row
// generation for Denoising/Committing requests. Extracted from
// scheduler/operations/forward.cpp so the generic forward planner stays
// mode-agnostic; the FSM transitions live in block_diffusion/lifecycle.cpp.

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "fsm/forward_events.h"
#include "fsm/forward_states.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "scheduler/block_diffusion/planner.h"
#include "scheduler/operations/forward.h"
#include "scheduler/request.h"
#include "scheduler/scheduler.h"

namespace tokenspeed {

namespace block_diffusion {

bool RetractableUnderKvPressure(const Request& request) {
    if (request.Is<fsm::Denoising>()) return true;  // canvas progress is regenerable
    if (request.Is<fsm::Committing>()) {
        // Don't discard a commit whose result is already in flight.
        return request.GetDiffusionProgress().phase != fsm::DiffusionProgress::Phase::kCommitInFlight;
    }
    return false;
}

}  // namespace block_diffusion

// Block-diffusion: schedule one denoise pass. Entering a canvas (from
// PrefillDone or from Committing after the commit result) reserves
// page-coverage for canvas_len tokens up front so the eventual commit can
// never fail; a continuing pass allocates nothing. Every diffusion row costs
// canvas_len forward tokens against max_scheduled_tokens.
std::optional<fsm::ScheduleDenoiseEvent> Scheduler::scheduleDenoise(Request* request, std::int32_t remaining,
                                                                    std::map<std::string, std::int32_t>& simulated_free) {
    const auto& params = request->GetBlockDiffusionParams();

    std::int32_t canvas_len = 0;
    bool entering_canvas = false;
    if (request->Is<fsm::Denoising>()) {
        canvas_len = request->GetDiffusionProgress().canvas_len;
    } else {
        // PrefillDone (first canvas) or Committing with commit_done (next
        // canvas); truncate the final canvas to the remaining budget.
        canvas_len = std::min(params.canvas_length, params.max_new_tokens - request->GeneratedSize());
        entering_canvas = true;
    }
    if (canvas_len <= 0 || canvas_len > remaining) {
        return {};
    }

    if (entering_canvas) {
        const std::int32_t tail_available = request->TailPageAvailableTokens();
        const std::int32_t extra_tokens = std::max(0, canvas_len - tail_available);
        const std::int32_t pages_needed = (extra_tokens + config_.page_size - 1) / config_.page_size;
        if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(pages_needed)) {
            return {};
        }
        // Paged-cache groups: admit growth to cover the canvas span before
        // reserving, so a converged canvas can never fail its commit on
        // group-pool pressure either.
        const std::int32_t first_pos = request->TokenSize();
        if (hybrid_prefix_cache_ &&
            !hybrid_prefix_cache_->AdmitChunk(request->Id(), first_pos, first_pos + canvas_len, simulated_free)) {
            return {};
        }
    }

    return fsm::ScheduleDenoiseEvent{canvas_len, request->IssueDiffusionPassEpoch(),
                                     hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr};
}

// Block-diffusion: Retracted → Denoising recovery. Mirrors
// scheduleDecodeFromRetracted: re-match committed history, LoadBack the
// host-only pages, and reserve a fresh canvas. The canvas restarts at step 0.
std::optional<fsm::ScheduleDenoiseFromRetractedEvent> Scheduler::scheduleDenoiseFromRetracted(Request* request,
                                                                                              std::int32_t remaining) {
    if (hybrid_prefix_cache_) {
        // Unreachable in practice: with paged-cache groups configured, the
        // KV-pressure path aborts diffusion victims instead of retracting
        // them (Phase-1 snapshot-only restore cannot rebuild group rows over
        // the full committed history; see contract doc 7b). Defensive guard
        // mirroring the lost-Mamba-state precedent above.
        spdlog::warn(
            "[Scheduler] Retracted block-diffusion request {} cannot be resumed with paged-cache groups configured, "
            "aborting request",
            request->Id());
        request->Apply(fsm::AbortEvent{});
        return {};
    }
    if (req_pool_allocator_.AvailableSlots() == 0) return {};

    const auto& params = request->GetBlockDiffusionParams();
    const std::int32_t canvas_len = std::min(params.canvas_length, params.max_new_tokens - request->GeneratedSize());
    if (canvas_len <= 0 || canvas_len > remaining) {
        return {};
    }

    MatchResult match_result = kv_prefix_cache_.Match(request->GetFullPagedTokens(true), MatchIntent::StateRecovery);
    std::vector<TreeNode*> loadback_diff = match_result.NodesWithout<ResourceType::Device>();

    const std::int32_t device_matched = match_result.device.DepthInPage();
    const std::int32_t host_matched = match_result.host.DepthInPage();
    std::int32_t num_tokens = canvas_len;
    if (host_matched > device_matched) {
        num_tokens += config_.page_size * (host_matched - device_matched);
    }
    std::int32_t device_pages_needed = (num_tokens + config_.page_size - 1) / config_.page_size;

    std::unique_ptr<DeviceNodeRef> temp_lock = std::make_unique<DeviceNodeRef>(match_result.device.last_node);
    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Device>(device_pages_needed)) {
        return {};
    }

    return fsm::ScheduleDenoiseFromRetractedEvent{
        canvas_len,        request->IssueDiffusionPassEpoch(),
        &device_allocator_, &req_pool_allocator_,
        &kv_prefix_cache_,  std::move(match_result),
        loadback_diff,
    };
}

// Block-diffusion: fill the explicit per-row KV write span (finding: a
// consumer must not need DiffusionKind to know which occupied_pages are the
// dark canvas reservation). The span covers the pages holding canvas
// positions [committed_len, committed_len + canvas_len); only kCommit may
// write them.
static void FillDiffusionWriteSpan(DiffusionOperation& op, std::int32_t page_size) {
    const std::int32_t first_canvas_page = op.committed_len / page_size;
    const std::int32_t end_canvas_page = (op.committed_len + op.canvas_len + page_size - 1) / page_size;
    op.write_page_begin = first_canvas_page;
    op.write_page_count = op.kind == DiffusionKind::kCommit ? end_canvas_page - first_canvas_page : 0;
}

DiffusionOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::ScheduleDenoiseEvent event) {
    const bool entering_canvas = !request->Is<fsm::Denoising>();
    std::int32_t begin = static_cast<std::int32_t>(request->GetOccupiedPages().size());
    request->Apply(std::move(event));
    std::vector<std::int32_t> all_pages = request->GetOccupiedPages();
    std::int32_t sz = static_cast<std::int32_t>(all_pages.size()) - begin;

    const auto progress = request->GetDiffusionProgress();
    DiffusionOperation op{{
        .request_id = request->Id(),
        .request_pool_index = request->GetReqPoolIndex(),
        .input_length = progress.canvas_len,
        .occupied_pages = std::move(all_pages),
        .begin = begin,
        .size = sz,
        .prefill_length = request->PrefillSize(),
    }};
    op.kind = DiffusionKind::kDenoise;
    op.canvas_len = progress.canvas_len;
    op.committed_len = request->TokenSize();
    op.steps_taken = progress.steps_taken;
    op.pass_epoch = progress.pass_epoch;
    FillDiffusionWriteSpan(op, config_.page_size);
    // Paged-cache groups. Order: attach, acquire, populate (matches the AR
    // decode path). On canvas entry the event inserted the committed full
    // pages into the radix tree; CommitChunk publishes snapshots at
    // history-alignment boundaries over them ("snapshot/insert on commit
    // boundaries"). AcquireForRequest grows the per-group tables to cover the
    // canvas span and drives the sliding-window release from the committed
    // length (first_raw_position_of_op = committed_len).
    if (hybrid_prefix_cache_) {
        if (entering_canvas) {
            hybrid_prefix_cache_->CommitChunk(op.request_id, const_cast<TreeNode*>(request->GetDeviceNode()));
        }
        hybrid_prefix_cache_->AcquireForRequest(op.request_id, op.committed_len, op.committed_len + op.canvas_len);
        hybrid_prefix_cache_->PopulateOp(op);
    }
    return op;
}

DiffusionOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::ScheduleDenoiseFromRetractedEvent event) {
    request->Apply(std::move(event));
    if (!request->Is<fsm::Denoising>()) {
        throw std::logic_error(
            "Scheduler::applyEventAndGenerateOp: expected state=Denoising after loadback recovery; got state=" +
            request->StateName());
    }
    std::vector<std::int32_t> all_pages = request->GetOccupiedPages();
    std::int32_t sz = static_cast<std::int32_t>(all_pages.size());

    const auto progress = request->GetDiffusionProgress();
    DiffusionOperation op{{
        .request_id = request->Id(),
        .request_pool_index = request->GetReqPoolIndex(),
        .input_length = progress.canvas_len,
        .occupied_pages = std::move(all_pages),
        .begin = 0,
        .size = sz,
        .prefill_length = request->PrefillSize(),
    }};
    op.kind = DiffusionKind::kDenoise;
    op.canvas_len = progress.canvas_len;
    op.committed_len = request->TokenSize();
    op.steps_taken = progress.steps_taken;  // 0: executor re-inits the canvas
    op.pass_epoch = progress.pass_epoch;
    FillDiffusionWriteSpan(op, config_.page_size);
    return op;
}

DiffusionOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::ScheduleCommitEvent event) {
    std::int32_t begin = static_cast<std::int32_t>(request->GetOccupiedPages().size());
    request->Apply(std::move(event));

    const auto progress = request->GetDiffusionProgress();
    DiffusionOperation op{{
        .request_id = request->Id(),
        .request_pool_index = request->GetReqPoolIndex(),
        .input_length = progress.canvas_len,
        .occupied_pages = request->GetOccupiedPages(),
        .begin = begin,
        .size = 0,  // the reservation was already surfaced when the canvas was entered
        .prefill_length = request->PrefillSize(),
    }};
    op.kind = DiffusionKind::kCommit;
    op.canvas_len = progress.canvas_len;
    op.committed_len = request->TokenSize();
    op.steps_taken = progress.steps_taken;
    op.pass_epoch = progress.pass_epoch;
    FillDiffusionWriteSpan(op, config_.page_size);
    // Paged-cache groups: the canvas span was acquired/admitted at canvas
    // entry; the commit pass only needs the block tables republished.
    if (hybrid_prefix_cache_) {
        hybrid_prefix_cache_->PopulateOp(op);
    }
    return op;
}

// One planning attempt for an active diffusion request (PrefillDone /
// Denoising / Committing). Produces at most one diffusion row in `op_out`;
// the outcome tells the generic planner loop whether the request was
// scheduled, is waiting on the executor, or starved on resources (the latter
// feeds the KV-pressure retraction trigger).
block_diffusion::PlanOutcome Scheduler::scheduleDiffusionPass(Request* request, std::int32_t token_budget,
                                                              std::map<std::string, std::int32_t>& simulated_free,
                                                              std::optional<DiffusionOperation>& op_out) {
    using block_diffusion::PlanOutcome;

    if (request->Is<fsm::PrefillDone>()) {
        // First canvas entry.
        if (auto ev = scheduleDenoise(request, token_budget, simulated_free)) {
            op_out = applyEventAndGenerateOp(request, std::move(*ev));
            return PlanOutcome::kScheduled;
        }
        return PlanOutcome::kStarved;
    }

    using Phase = fsm::DiffusionProgress::Phase;
    const auto progress = request->GetDiffusionProgress();
    switch (progress.phase) {
        case Phase::kDenoisePassInFlight:
        case Phase::kCommitInFlight:
            // One pass at a time: wait for the DenoiseResult / commit
            // ExtendResult.
            return PlanOutcome::kWaitingOnExecutor;
        case Phase::kCommitReady:
            if (progress.canvas_len <= token_budget) {
                op_out = applyEventAndGenerateOp(request, fsm::ScheduleCommitEvent{request->IssueDiffusionPassEpoch()});
                return PlanOutcome::kScheduled;
            }
            return PlanOutcome::kStarved;
        case Phase::kCommitDone:
            if (request->GeneratedSize() >= request->GetBlockDiffusionParams().max_new_tokens) {
                // Scheduler-enforced termination: the executor reported the
                // commit without Finish but the generation budget is spent.
                // Apply the FSM transition directly — the planner never
                // re-enters the outside-event path.
                finishForward(request);
                return PlanOutcome::kNoop;
            }
            if (auto ev = scheduleDenoise(request, token_budget, simulated_free)) {
                // Next canvas (fresh reservation, steps_taken = 0).
                op_out = applyEventAndGenerateOp(request, std::move(*ev));
                return PlanOutcome::kScheduled;
            }
            return PlanOutcome::kStarved;
        case Phase::kDenoisePassReady:
            // One more pass over the current canvas.
            if (auto ev = scheduleDenoise(request, token_budget, simulated_free)) {
                op_out = applyEventAndGenerateOp(request, std::move(*ev));
                return PlanOutcome::kScheduled;
            }
            return PlanOutcome::kStarved;
    }
    return PlanOutcome::kNoop;
}

}  // namespace tokenspeed
