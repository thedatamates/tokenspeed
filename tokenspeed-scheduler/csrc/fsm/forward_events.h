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

#pragma once

// Put transitions and resource ownership transfer into each particular event
// Put resource allocation into function call operators of events

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "fsm/base_event.h"
#include "fsm/forward_states.h"
#include "resource/types.h"
#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"
#include "resource/allocator/mamba_chunk_allocator.h"
#include "resource/allocator/local_mamba_allocator.h"
#include "utils.h"

namespace tokenspeed {
class PageAllocator;
class KVPrefixCache;
class ReqPoolAllocator;
class TreeNode;
}  // namespace tokenspeed

namespace tokenspeed::fsm {

struct PrefetchDone;
struct Prefetching;

void InsertHybridCache(HybridPrefixCache* hybrid_prefix_cache,
                       const std::vector<std::span<const std::int32_t>>& full_paged_tokens,
                       std::unique_ptr<DeviceNodeRef>& device_node_ref, LocalKVAllocator* local_kv_allocator,
                       LocalMambaAllocator* local_mamba_allocator, std::int32_t chunk_begin, std::int32_t chunk_size,
                       std::int32_t page_size);

struct SchedulePrefillFirstChunkEvent : InvalidTransitionHandler<SchedulePrefillFirstChunkEvent> {
    using InvalidTransitionHandler<SchedulePrefillFirstChunkEvent>::operator();
    SchedulePrefillFirstChunkEvent(std::int32_t tokens_this_round, std::int32_t decode_input_tokens,
                                   PageAllocator* device_allocator, ReqPoolAllocator* req_pool_allocator,
                                   MatchResult match_result, Role role, KVPrefixCache* kv_prefix_cache,
                                   bool disable_l2_cache, std::vector<TreeNode*> loadback_diff,
                                   HybridPrefixCache* hybrid_prefix_cache = nullptr,
                                   MambaChunkAllocator* mamba_allocator = nullptr,
                                   std::vector<TreeNode*> mamba_loadback_nodes = {})
        : tokens_this_round_(tokens_this_round),
          decode_input_tokens_(decode_input_tokens),
          device_allocator_(device_allocator),
          req_pool_allocator_(req_pool_allocator),
          match_result_(match_result),
          role_{role},
          disable_l2_cache_{disable_l2_cache},
          loadback_diff_(std::move(loadback_diff)),
          mamba_loadback_nodes_(std::move(mamba_loadback_nodes)),
          kv_prefix_cache_(kv_prefix_cache),
          hybrid_prefix_cache_(hybrid_prefix_cache),
          mamba_allocator_(mamba_allocator) {}

    // Returns PrefillDone (single-chunk or last chunk) or Prefilling (more chunks remain).
    std::variant<PrefillDone, Prefilling> operator()(Submitted&& state);

    const MatchResult GetMatchResult() const { return match_result_; }

    const std::vector<TreeNode*>& GetLoadbackDiff() const { return loadback_diff_; }
    const std::vector<TreeNode*>& GetMambaLoadbackNodes() const { return mamba_loadback_nodes_; }

private:
    std::int32_t tokens_this_round_{};
    std::int32_t decode_input_tokens_{};
    PageAllocator* device_allocator_{};
    ReqPoolAllocator* req_pool_allocator_{};
    const MatchResult match_result_{};
    const Role role_;
    bool disable_l2_cache_{};
    std::vector<TreeNode*> loadback_diff_;
    std::vector<TreeNode*> mamba_loadback_nodes_;
    KVPrefixCache* kv_prefix_cache_;
    HybridPrefixCache* hybrid_prefix_cache_{};
    MambaChunkAllocator* mamba_allocator_{};
};

struct SchedulePrefillEvent : InvalidTransitionHandler<SchedulePrefillEvent> {
    using InvalidTransitionHandler<SchedulePrefillEvent>::operator();
    SchedulePrefillEvent(std::int32_t tokens_this_round, std::int32_t reserve_num_tokens_in_next_schedule_event,
                         HybridPrefixCache* hybrid_prefix_cache = nullptr)
        : tokens_this_round_(tokens_this_round),
          reserve_num_tokens_in_next_schedule_event_(reserve_num_tokens_in_next_schedule_event),
          hybrid_prefix_cache_(hybrid_prefix_cache) {}

    // Returns PrefillDone (last chunk) or Prefilling (more chunks remain).
    std::variant<PrefillDone, Prefilling> operator()(Prefilling&& state);

private:
    std::int32_t tokens_this_round_{};
    std::int32_t reserve_num_tokens_in_next_schedule_event_{};
    HybridPrefixCache* hybrid_prefix_cache_{};
};

struct ScheduleDecodeEvent : InvalidTransitionHandler<ScheduleDecodeEvent> {
    using InvalidTransitionHandler<ScheduleDecodeEvent>::operator();

    ScheduleDecodeEvent(std::int32_t decode_input_tokens, HybridPrefixCache* hybrid_prefix_cache = nullptr)
        : decode_input_tokens_(decode_input_tokens), hybrid_prefix_cache_(hybrid_prefix_cache) {}

    Decoding operator()(PrefillDone&& state);
    Decoding operator()(Decoding&& state);

private:
    std::int32_t decode_input_tokens_;
    HybridPrefixCache* hybrid_prefix_cache_{};
};

struct ScheduleDecodeFromRetractedEvent : InvalidTransitionHandler<ScheduleDecodeFromRetractedEvent> {
    using InvalidTransitionHandler<ScheduleDecodeFromRetractedEvent>::operator();

    // Constructor for Retracted → Decoding recovery (LoadBack from host).
    ScheduleDecodeFromRetractedEvent(std::int32_t decode_input_tokens, PageAllocator* device_allocator,
                                     ReqPoolAllocator* req_pool_allocator, KVPrefixCache* kv_prefix_cache,
                                     MatchResult match_result, std::vector<TreeNode*> loadback_diff,
                                     MambaChunkAllocator* mamba_allocator = nullptr,
                                     std::vector<TreeNode*> mamba_loadback_nodes = {})
        : decode_input_tokens_(decode_input_tokens),
          device_allocator_(device_allocator),
          req_pool_allocator_(req_pool_allocator),
          kv_prefix_cache_(kv_prefix_cache),
          match_result_(std::move(match_result)),
          loadback_diff_(std::move(loadback_diff)),
          mamba_loadback_nodes_(std::move(mamba_loadback_nodes)),
          mamba_allocator_(mamba_allocator) {}

    Decoding operator()(Retracted&& state);

    const MatchResult& GetMatchResult() const { return match_result_; }

    const std::vector<TreeNode*>& GetLoadbackDiff() const { return loadback_diff_; }
    const std::vector<TreeNode*>& GetMambaLoadbackNodes() const { return mamba_loadback_nodes_; }

private:
    std::int32_t decode_input_tokens_{};
    PageAllocator* device_allocator_{};
    ReqPoolAllocator* req_pool_allocator_{};
    KVPrefixCache* kv_prefix_cache_{};
    MatchResult match_result_{};
    std::vector<TreeNode*> loadback_diff_;
    std::vector<TreeNode*> mamba_loadback_nodes_;
    MambaChunkAllocator* mamba_allocator_{};
};

// Block-diffusion: schedule one denoise pass. Entering a canvas
// (PrefillDone → Denoising, Committing → Denoising) acquires the canvas
// reservation; Denoising → Denoising only marks the pass in flight.
struct ScheduleDenoiseEvent : InvalidTransitionHandler<ScheduleDenoiseEvent> {
    using InvalidTransitionHandler<ScheduleDenoiseEvent>::operator();

    explicit ScheduleDenoiseEvent(std::int32_t canvas_len) : canvas_len_(canvas_len) {}

    // Canvas entry: reserve ceil-page coverage for canvas_len tokens so the
    // eventual commit cannot fail, then start at step 0 (executor inits canvas).
    Denoising operator()(PrefillDone&& state);
    Denoising operator()(Committing&& state);
    // Continuing canvas: one more pass over the same canvas, no allocation.
    Denoising operator()(Denoising&& state);

private:
    std::int32_t canvas_len_{};
};

// Block-diffusion: Retracted → Denoising recovery (mirrors
// ScheduleDecodeFromRetractedEvent). Committed KV is recovered via LoadBack;
// canvas progress was discarded at retraction, so the canvas restarts with a
// fresh reservation at step 0.
struct ScheduleDenoiseFromRetractedEvent : InvalidTransitionHandler<ScheduleDenoiseFromRetractedEvent> {
    using InvalidTransitionHandler<ScheduleDenoiseFromRetractedEvent>::operator();

    ScheduleDenoiseFromRetractedEvent(std::int32_t canvas_len, PageAllocator* device_allocator,
                                      ReqPoolAllocator* req_pool_allocator, KVPrefixCache* kv_prefix_cache,
                                      MatchResult match_result, std::vector<TreeNode*> loadback_diff)
        : canvas_len_(canvas_len),
          device_allocator_(device_allocator),
          req_pool_allocator_(req_pool_allocator),
          kv_prefix_cache_(kv_prefix_cache),
          match_result_(std::move(match_result)),
          loadback_diff_(std::move(loadback_diff)) {}

    Denoising operator()(Retracted&& state);

    const std::vector<TreeNode*>& GetLoadbackDiff() const { return loadback_diff_; }

private:
    std::int32_t canvas_len_{};
    PageAllocator* device_allocator_{};
    ReqPoolAllocator* req_pool_allocator_{};
    KVPrefixCache* kv_prefix_cache_{};
    MatchResult match_result_{};
    std::vector<TreeNode*> loadback_diff_;
};

// Block-diffusion: schedule the single commit pass for a converged canvas.
struct ScheduleCommitEvent : InvalidTransitionHandler<ScheduleCommitEvent> {
    using InvalidTransitionHandler<ScheduleCommitEvent>::operator();

    Committing operator()(Committing&& state) {
        state.MarkCommitScheduled();
        return std::move(state);
    }
};

// Block-diffusion outside event: the executor finished one denoise pass.
// converged=false increments steps_taken and stays Denoising unless the
// scheduler-enforced max_denoising_steps backstop is reached; converged=true
// (or the backstop) transitions to Committing.
struct DenoiseResultEvent : InvalidTransitionHandler<DenoiseResultEvent> {
    using InvalidTransitionHandler<DenoiseResultEvent>::operator();

    DenoiseResultEvent(bool converged, std::int32_t max_denoising_steps)
        : converged_(converged), max_denoising_steps_(max_denoising_steps) {}

    std::variant<Denoising, Committing> operator()(Denoising&& state);

    // Overlap scheduling / retraction races: a denoise result can arrive after
    // the canvas was discarded (retract) or the request terminalized. Canvas
    // progress is regenerable, so the stale result is dropped.
    Retracting operator()(Retracting&& state) { return std::move(state); }
    Retracted operator()(Retracted&& state) { return std::move(state); }
    Finished operator()(Finished&& state) { return std::move(state); }

private:
    bool converged_{};
    std::int32_t max_denoising_steps_{};
};

struct FinishEvent : InvalidTransitionHandler<FinishEvent> {
    using InvalidTransitionHandler<FinishEvent>::operator();
    explicit FinishEvent(KVPrefixCache* kv_prefix_cache, PageAllocator* host_allocator,
                         std::vector<std::string> page_hashes = {}, bool disable_l2_cache = false,
                         HybridPrefixCache* hybrid_prefix_cache = nullptr)
        : kv_prefix_cache_(kv_prefix_cache),
          host_allocator_(host_allocator),
          page_hashes_(std::move(page_hashes)),
          disable_l2_cache_(disable_l2_cache),
          hybrid_prefix_cache_(hybrid_prefix_cache) {}

    // Returns Draining (needs device→host writeback) or Finished.
    std::variant<Draining, Finished> operator()(Decoding&& state);
    std::variant<Draining, Finished> operator()(PrefillDone&& state);
    // Block-diffusion: EOS reported with the commit, or max_new_tokens reached.
    // Unused canvas-reservation pages are released here (local allocator pages
    // beyond the committed history are dropped back to the pool).
    std::variant<Draining, Finished> operator()(Committing&& state);

    // Retracting: writeback already in-flight.
    WritingBack operator()(Retracting&& state);
    Finished operator()(Retracted&& state) { return Finished{}; };
    // Defensive: late forward finish after terminalization, stay Finished.
    Finished operator()(Finished&& state) { return std::move(state); }

private:
    KVPrefixCache* kv_prefix_cache_{};
    std::vector<std::string> page_hashes_;
    PageAllocator* host_allocator_;
    bool disable_l2_cache_;
    HybridPrefixCache* hybrid_prefix_cache_{};

    template <typename ForwardStateT>
    std::variant<Draining, Finished> apply(ForwardStateT&& state);
};

struct AbortEvent : InvalidTransitionHandler<AbortEvent> {
    using InvalidTransitionHandler<AbortEvent>::operator();

    Finished operator()(Submitted&& state);
    Aborting operator()(Prefetching&& state);
    Finished operator()(PrefetchDone&&);
    Finished operator()(Prefilling&&);
    Finished operator()(PrefillDone&&);
    Finished operator()(Decoding&&);
    Finished operator()(Denoising&&);
    Finished operator()(Committing&&);
    Finished operator()(Retracting&&);
    Finished operator()(Retracted&&);
    Finished operator()(Draining&&);
    // Defensive: late or duplicate abort after terminalization, stay Finished.
    Finished operator()(Finished&& state) { return std::move(state); }
    Aborting operator()(Aborting&& state);  // Defensive: duplicate abort, stay Aborting
};

struct ScheduleRetractEvent : InvalidTransitionHandler<ScheduleRetractEvent> {
    using InvalidTransitionHandler<ScheduleRetractEvent>::operator();
    ScheduleRetractEvent(KVPrefixCache* kv_prefix_cache, PageAllocator* host_allocator, MatchResult match_result,
                         HybridPrefixCache* hybrid_prefix_cache = nullptr)
        : kv_prefix_cache_(kv_prefix_cache),
          host_allocator_(host_allocator),
          match_result_(match_result),
          hybrid_prefix_cache_(hybrid_prefix_cache) {}

    Retracting operator()(Decoding&& state);
    Retracting operator()(PrefillDone&& state);
    // Block-diffusion: canvas progress is discarded (regenerable) and the
    // canvas reservation freed; committed KV follows the normal writeback path.
    Retracting operator()(Denoising&& state);
    Retracting operator()(Committing&& state);

    MatchResult GetMatchResult() { return match_result_; }

private:
    template <typename ForwardStateT>
    Retracting applyRetract(ForwardStateT&& state);

    KVPrefixCache* kv_prefix_cache_{};
    PageAllocator* host_allocator_{};
    const MatchResult match_result_{};
    HybridPrefixCache* hybrid_prefix_cache_{};
};

// Draining → WritingBack: WriteBack op has been generated this round; transfer
// RAII locks from Draining into WritingBack so pages stay pinned during transfer.
struct CommitDrainingEvent : InvalidTransitionHandler<CommitDrainingEvent> {
    using InvalidTransitionHandler<CommitDrainingEvent>::operator();
    WritingBack operator()(Draining&& state);
};

// WritingBack → Finished:  async Device→Host transfer complete; node-ref locks released.
// Retracting  → Retracted: same transfer path for preempted requests;
//                          device_node_ref drops (frees GPU pages), host_node_ref moves into Retracted.
struct WriteBackDoneEvent : InvalidTransitionHandler<WriteBackDoneEvent> {
    explicit WriteBackDoneEvent(KVPrefixCache* kv_prefix_cache = nullptr,
                                HybridPrefixCache* hybrid_prefix_cache = nullptr)
        : kv_prefix_cache_(kv_prefix_cache), hybrid_prefix_cache_(hybrid_prefix_cache) {}

    using InvalidTransitionHandler<WriteBackDoneEvent>::operator();
    Finished operator()(WritingBack&& state);
    Retracted operator()(Retracting&& state);

private:
    KVPrefixCache* kv_prefix_cache_{};
    HybridPrefixCache* hybrid_prefix_cache_{};
};

struct UpdateReserveNumTokensEvent : InvalidTransitionHandler<UpdateReserveNumTokensEvent> {
    using InvalidTransitionHandler<UpdateReserveNumTokensEvent>::operator();

    explicit UpdateReserveNumTokensEvent(std::int32_t new_value) : new_value_(new_value) {}

    Decoding operator()(Decoding&& state) {
        state.SetReserveNumTokensInNextScheduleEvent(new_value_);
        return std::move(state);
    }

    Retracting operator()(Retracting&& state) { return std::move(state); }

    Retracted operator()(Retracted&& state) { return std::move(state); }

    // Overlap scheduling can commit an already-dispatched decode result after
    // this request was terminalized (for example retract failure -> AbortEvent).
    // The reserve hint only affects a future schedule round, so it is stale
    // once Finished. Other invalid states still fall through to the strict FSM
    // handler.
    Finished operator()(Finished&& state) { return std::move(state); }

private:
    std::int32_t new_value_;
};

struct ExtendResultEvent : InvalidTransitionHandler<ExtendResultEvent> {
    using InvalidTransitionHandler<ExtendResultEvent>::operator();
    ExtendResultEvent() = delete;

    ExtendResultEvent(std::vector<std::int32_t> result_tokens) : result_tokens_(result_tokens) {}

public:
    template <typename S>
        requires CanExtendTokenContainer<S>
    std::remove_cvref_t<S> operator()(S&& state) {
        state.ExtendResultTokens(result_tokens_);
        return std::move(state);
    }

    // Overlap scheduling can commit an already-dispatched forward result after
    // this request was terminalized (for example retract failure -> AbortEvent).
    // The result tokens are stale and must not mutate TokenContainer or revive
    // the request. Other invalid states still fall through to the strict FSM
    // handler.
    Finished operator()(Finished&& state) { return std::move(state); }

private:
    std::vector<std::int32_t> result_tokens_;
};

}  // namespace tokenspeed::fsm
