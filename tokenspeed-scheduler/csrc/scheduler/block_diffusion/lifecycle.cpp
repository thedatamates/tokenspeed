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

// Block-diffusion FSM lifecycle: the Denoising/Committing state transitions
// (canvas entry, pass scheduling, denoise results, retraction-resume).
// Extracted from fsm/forward_events.cpp so the generic forward FSM stays
// mode-agnostic; the planner-side policy lives in block_diffusion/planner.cpp.
// The event types remain declared in fsm/forward_events.h (the FSM surface is
// unchanged); only the definitions live here.

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <variant>

#include <spdlog/spdlog.h>

#include "core/token_container.h"
#include "fsm/forward_events.h"
#include "fsm/forward_states.h"
#include "resource/allocator/kv_allocator.h"
#include "resource/allocator/req_pool_allocator.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/radix_tree/tree_node.h"
#include "resource/types.h"

namespace tokenspeed::fsm {

// PrefillDone -> Denoising: enter the first canvas. On the vanilla path the
// prefill pages stay in the local allocator (radix insert happens at
// finish/retract, mirroring ScheduleDecodeEvent); with the hybrid prefix
// cache the committed full pages are inserted here (except-last convention:
// the page holding the final committed token stays local) so paged-cache
// snapshots can attach at history-alignment boundaries. The canvas
// reservation is acquired here so the eventual commit cannot fail.
Denoising ScheduleDenoiseEvent::operator()(PrefillDone&& state) {
    auto local_kv_allocator = std::move(state).TakeLocalKVAllocator();
    auto device_node_ref = std::move(state).TakeDeviceNodeRef();
    auto host_node_ref = std::move(state).TakeHostNodeRef();

    InsertHybridCache(hybrid_prefix_cache_, state.GetFullPagedTokens(true), device_node_ref, local_kv_allocator.get(),
                      /*local_mamba_allocator=*/nullptr, /*chunk_begin=*/0, /*chunk_size=*/0, state.GetPageSize());
    local_kv_allocator->Acquire(canvas_len_);

    return Denoising{state.GetTokenContainer(),
                     state.GetPageSize(),
                     std::move(host_node_ref),
                     std::move(device_node_ref),
                     std::move(local_kv_allocator),
                     std::move(state).TakeReqPoolIndex(),
                     canvas_len_,
                     /*steps_taken=*/0,
                     pass_epoch_,
                     Denoising::SubState::kPassInFlight};
}

// Committing -> Denoising: the committed canvas became request history
// (ExtendResult already appended the tokens); reserve the next canvas and
// restart the step counter. steps_taken == 0 tells the executor to
// reinitialize its canvas scratch. With the hybrid prefix cache, the newly
// committed full pages are inserted into the radix tree (except-last) — this
// is the "snapshot/insert on commit boundaries" hook: the scheduler-side
// CommitChunk that follows publishes paged-cache snapshots over them.
Denoising ScheduleDenoiseEvent::operator()(Committing&& state) {
    auto local_kv_allocator = std::move(state).TakeLocalKVAllocator();
    auto device_node_ref = std::move(state).TakeDeviceNodeRef();
    auto host_node_ref = std::move(state).TakeHostNodeRef();

    InsertHybridCache(hybrid_prefix_cache_, state.GetFullPagedTokens(true), device_node_ref, local_kv_allocator.get(),
                      /*local_mamba_allocator=*/nullptr, /*chunk_begin=*/0, /*chunk_size=*/0, state.GetPageSize());
    local_kv_allocator->Acquire(canvas_len_);

    return Denoising{state.GetTokenContainer(),
                     state.GetPageSize(),
                     std::move(host_node_ref),
                     std::move(device_node_ref),
                     std::move(local_kv_allocator),
                     std::move(state).TakeReqPoolIndex(),
                     canvas_len_,
                     /*steps_taken=*/0,
                     pass_epoch_,
                     Denoising::SubState::kPassInFlight};
}

// Denoising -> Denoising: one more pass over the same canvas; no allocation.
Denoising ScheduleDenoiseEvent::operator()(Denoising&& state) {
    const std::int32_t canvas_len = state.GetCanvasLen();
    const std::int32_t steps_taken = state.GetStepsTaken();
    auto local_kv_allocator = std::move(state).TakeLocalKVAllocator();
    auto device_node_ref = std::move(state).TakeDeviceNodeRef();
    auto host_node_ref = std::move(state).TakeHostNodeRef();

    return Denoising{state.GetTokenContainer(),
                     state.GetPageSize(),
                     std::move(host_node_ref),
                     std::move(device_node_ref),
                     std::move(local_kv_allocator),
                     std::move(state).TakeReqPoolIndex(),
                     canvas_len,
                     steps_taken,
                     pass_epoch_,
                     Denoising::SubState::kPassInFlight};
}

// Retracted -> Denoising: recover committed KV via LoadBack (host → device),
// then restart the interrupted canvas with a fresh reservation at step 0.
Denoising ScheduleDenoiseFromRetractedEvent::operator()(Retracted&& state) {
    std::unique_ptr<HostNodeRef> host_node_ref{nullptr};
    std::unique_ptr<DeviceNodeRef> device_node_ref{nullptr};
    if (match_result_.host.DepthInPage() > match_result_.device.DepthInPage()) {
        host_node_ref = std::make_unique<HostNodeRef>(match_result_.host.last_node);
        if (!kv_prefix_cache_->AllocateResourceOfType<ResourceType::Device>(
                match_result_.NodesWithout<ResourceType::Device>())) {
            throw std::logic_error(
                "ScheduleDenoiseFromRetractedEvent: failed to allocate device pages for host cache recovery");
        }
        device_node_ref = std::make_unique<DeviceNodeRef>(match_result_.host.last_node);
    } else {
        device_node_ref = std::make_unique<DeviceNodeRef>(match_result_.device.last_node);
    }
    TokenContainer* token_container = state.GetTokenContainer();
    std::int32_t page_size = state.GetPageSize();
    auto local_kv_allocator = std::move(state).TakeKVAllocator();
    auto req_pool_index = std::make_unique<ReqPoolIndex>(req_pool_allocator_->Allocate());
    local_kv_allocator->Acquire(canvas_len_);
    return Denoising{token_container,
                     page_size,
                     std::move(host_node_ref),
                     std::move(device_node_ref),
                     std::move(local_kv_allocator),
                     std::move(req_pool_index),
                     canvas_len_,
                     /*steps_taken=*/0,
                     pass_epoch_,
                     Denoising::SubState::kPassInFlight};
}

// Denoising -> Denoising / Committing: executor reported one finished denoise
// pass. The scheduler owns the step counter; the backstop fires here even for
// an executor that never reports convergence.
std::variant<Denoising, Committing> DenoiseResultEvent::operator()(Denoising&& state) {
    // Pass identity: a result must echo the epoch its pass was scheduled
    // with. A mismatch is a stale in-flight result from a discarded canvas
    // (retract → resume restarts at a fresh epoch); accepting it would
    // advance the step counter / convergence of the wrong canvas.
    if (pass_epoch_ != state.GetPassEpoch()) {
        spdlog::warn("[fsm] Dropping stale DenoiseResult: echoed pass_epoch={} but current pass_epoch={}", pass_epoch_,
                     state.GetPassEpoch());
        return std::move(state);
    }

    switch (state.GetSubState()) {
        case Denoising::SubState::kPassReady:
            // Strict one-pass-in-flight: no pass is outstanding, so this
            // result is a stale duplicate. Drop it without touching the
            // step counter.
            spdlog::warn("[fsm] Dropping DenoiseResult with no denoise pass in flight (stale duplicate)");
            return std::move(state);
        case Denoising::SubState::kPassInFlight:
            break;
    }

    const std::int32_t canvas_len = state.GetCanvasLen();
    const std::int32_t steps_taken = state.GetStepsTaken() + 1;
    const std::int64_t pass_epoch = state.GetPassEpoch();
    auto local_kv_allocator = std::move(state).TakeLocalKVAllocator();
    auto device_node_ref = std::move(state).TakeDeviceNodeRef();
    auto host_node_ref = std::move(state).TakeHostNodeRef();

    if (converged_ || steps_taken >= max_denoising_steps_) {
        return Committing{state.GetTokenContainer(),
                          state.GetPageSize(),
                          std::move(host_node_ref),
                          std::move(device_node_ref),
                          std::move(local_kv_allocator),
                          std::move(state).TakeReqPoolIndex(),
                          canvas_len,
                          steps_taken,
                          pass_epoch};
    }
    return Denoising{state.GetTokenContainer(),
                     state.GetPageSize(),
                     std::move(host_node_ref),
                     std::move(device_node_ref),
                     std::move(local_kv_allocator),
                     std::move(state).TakeReqPoolIndex(),
                     canvas_len,
                     steps_taken,
                     pass_epoch,
                     Denoising::SubState::kPassReady};
}

// A DenoiseResult can never answer a commit pass (commits answer with
// ExtendResult), so any receipt in Committing is a stale result from a
// discarded canvas — e.g. the pre-retract pass arriving after the resumed
// request already converged again.
Committing DenoiseResultEvent::operator()(Committing&& state) {
    spdlog::warn("[fsm] Dropping stale DenoiseResult (pass_epoch={}) received while Committing", pass_epoch_);
    return std::move(state);
}

// Block-diffusion: canvas progress is regenerable — drop the unconsumed
// reservation before the standard retract writeback. The page holding the
// last committed tokens (not radix-insertable due to except_last) is kept in
// the local allocator across Retracting/Retracted, so its device KV survives
// and no recovery forward is needed on resume. applyRetract is the shared
// retract machinery (defined and explicitly instantiated in
// fsm/forward_events.cpp).
Retracting ScheduleRetractEvent::operator()(Denoising&& state) {
    state.ReleaseCanvasReservation();
    return applyRetract(std::move(state));
}

Retracting ScheduleRetractEvent::operator()(Committing&& state) {
    state.ReleaseCanvasReservation();
    return applyRetract(std::move(state));
}

}  // namespace tokenspeed::fsm
