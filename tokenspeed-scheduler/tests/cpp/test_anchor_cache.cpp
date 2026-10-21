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

#include "integration_test_helper.h"

namespace tokenspeed::test {
const ForwardBatch& anchor_forward(const ExecutionPlan& plan) {
    for (const auto& operation : plan.Operations())
        if (const auto* batch = std::get_if<ForwardBatch>(&operation)) return *batch;
    throw std::runtime_error("expected forward");
}
void anchor_done(Scheduler& scheduler, const std::string& id, std::vector<std::int32_t> tokens) {
    ExecutionEvent event;
    event.With(forward::ExtendResult{.request_id = id, .tokens = std::move(tokens)});
    scheduler.Advance(std::move(event));
}
TEST(AnchorCache, PackedHostEvictionReplayAndNamespaceIsolation) {
    {
        BlockPool device(1, std::vector<std::int32_t>{1});
        BlockPool host(1, std::vector<std::int32_t>{2});
        CacheGroupSpec spec{};
        spec.kind = AttnKind::kSlidingWindow;
        spec.sliding_window = 4;
        spec.block_granularity = 2;
        spec.cache_blocks_per_lcm_block = 1;
        auto coordinator = MakeCoordinator(std::span<const CacheGroupSpec>(&spec, 1), 2, device, false, &host, true);
        const std::vector<std::uint32_t> requested{0, 0};
        auto allocation = coordinator.AcquireHostBlocks(requested);
        ASSERT_TRUE(allocation.blocks.size() == 2 && allocation.blocks[0] && allocation.blocks[1]);
        ASSERT_TRUE(coordinator.ResolveHostBlockId(0, allocation.blocks[0]->Location()) == 1);
        ASSERT_TRUE(coordinator.ResolveHostBlockId(0, allocation.blocks[1]->Location()) == 2);
        ASSERT_TRUE(host.NumEmptyLcmBlocks() == 0);
        ASSERT_TRUE(!coordinator.AcquireHostBlock(0));
        allocation.blocks.clear();
        ASSERT_TRUE(host.NumEmptyLcmBlocks() == 1);
    }

    {
        BlockPool device(2, std::vector<std::int32_t>{1, 1});
        BlockPool host(1, std::vector<std::int32_t>{2, 1});
        std::vector<CacheGroupSpec> specs(2);
        for (auto& spec : specs) {
            spec.block_granularity = 2;
            spec.cache_blocks_per_lcm_block = 1;
        }
        auto coordinator = MakeCoordinator(specs, 2, device, false, &host, true);
        auto a = coordinator.AcquireHostBlock(0);
        auto b = coordinator.AcquireHostBlock(0);
        coordinator.CacheHostBlock(a, CacheKey{.group_id = 0, .content_hash = "a"});
        coordinator.CacheHostBlock(b, CacheKey{.group_id = 0, .content_hash = "b"});
        a.reset();
        b.reset();
        ASSERT_TRUE(coordinator.NumHostCachedBlocks() == 2);
        auto other = coordinator.AcquireHostBlock(1);
        ASSERT_TRUE(other && coordinator.NumHostCachedBlocks() == 0);
        ASSERT_TRUE(coordinator.ResolveHostBlockId(1, other->Location()) == 1);
    }
    SchedulerConfig config{};
    config.prefix_granularity = 2;
    config.device_allocator.total_pages = 128;
    config.host_allocator.total_pages = 0;
    config.max_scheduled_tokens = 64;
    config.max_batch_size = 1;
    config.disable_l2_cache = true;
    config.enable_l3_storage = false;
    config.cache_groups.push_back(CacheGroupConfig{.group_id = "history",
                                                   .block_granularity = 2,
                                                   .total_pages = 128,
                                                   .retention = CacheGroupConfig::Retention::FullHistory,
                                                   .family = CacheGroupFamily::History});
    config.cache_groups.push_back(CacheGroupConfig{.group_id = "window",
                                                   .block_granularity = 2,
                                                   .total_pages = 128,
                                                   .retention = CacheGroupConfig::Retention::SlidingWindow,
                                                   .sliding_window_tokens = 4,
                                                   .family = CacheGroupFamily::History});
    Scheduler scheduler(config);
    RequestSpec spec{};
    spec.request_id = "seed";
    spec.tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    spec.max_new_tokens = 4;
    spec.cache_anchor_tokens = 6;
    scheduler.SubmitRequests({spec});
    auto first = scheduler.NextExecutionPlan();
    ASSERT_TRUE(anchor_forward(first).extend_prefix_lens[0] == 0 && anchor_forward(first).input_lengths[0] == 6);
    anchor_done(scheduler, "seed", {});
    auto second = scheduler.NextExecutionPlan();
    ASSERT_TRUE(anchor_forward(second).extend_prefix_lens[0] == 6 && anchor_forward(second).input_lengths[0] == 6);
    anchor_done(scheduler, "seed", {42});
    ExecutionEvent finish;
    finish.With(forward::Finish{.request_id = "seed"});
    scheduler.Advance(std::move(finish));
    scheduler.NextExecutionPlan();
    spec.request_id = "replay";
    scheduler.SubmitRequests({spec});
    auto replay = scheduler.NextExecutionPlan();
    ASSERT_TRUE(anchor_forward(replay).extend_prefix_lens[0] == 6 && anchor_forward(replay).input_lengths[0] == 6);
    anchor_done(scheduler, "replay", {42});
    ExecutionEvent retract;
    retract.With(forward::Retract{.request_id = "replay"});
    scheduler.Advance(std::move(retract));
    auto recovery = scheduler.NextExecutionPlan();
    ASSERT_TRUE(anchor_forward(recovery).extend_prefix_lens[0] <= 6);
    anchor_done(scheduler, "replay", {43});
    ExecutionEvent end_replay;
    end_replay.With(forward::Finish{.request_id = "replay"});
    scheduler.Advance(std::move(end_replay));
    scheduler.NextExecutionPlan();
    spec.request_id = "other-format";
    spec.cache_namespace = "producer-half-distinct-format";
    scheduler.SubmitRequests({spec});
    auto other_format = scheduler.NextExecutionPlan();
    ASSERT_TRUE(anchor_forward(other_format).extend_prefix_lens[0] == 0);
    ASSERT_TRUE(anchor_forward(other_format).input_lengths[0] == 6);
}
TEST(AnchorCache, TieredAnchorPreservesResidentWindowHit) {
    SchedulerConfig config{};
    config.prefix_granularity = 2;
    config.device_allocator.total_pages = 128;
    config.host_allocator.total_pages = 0;
    config.max_scheduled_tokens = 64;
    config.max_batch_size = 1;
    config.disable_l2_cache = true;
    config.enable_l3_storage = false;
    config.cache_groups.push_back(CacheGroupConfig{.group_id = "history",
                                                   .block_granularity = 2,
                                                   .total_pages = 128,
                                                   .retention = CacheGroupConfig::Retention::FullHistory,
                                                   .family = CacheGroupFamily::History});
    config.cache_groups.push_back(CacheGroupConfig{.group_id = "window",
                                                   .block_granularity = 2,
                                                   .total_pages = 128,
                                                   .retention = CacheGroupConfig::Retention::SlidingWindow,
                                                   .sliding_window_tokens = 4,
                                                   .family = CacheGroupFamily::History});
    Scheduler scheduler(config);
    RequestSpec spec{};
    spec.request_id = "seed";
    spec.tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    spec.max_new_tokens = 4;
    spec.cache_anchor_tokens = 6;
    spec.cache_anchor_host_only = true;
    scheduler.SubmitRequests({spec});
    auto first = scheduler.NextExecutionPlan();
    ASSERT_TRUE(anchor_forward(first).extend_prefix_lens[0] == 0 && anchor_forward(first).input_lengths[0] == 6);
    anchor_done(scheduler, "seed", {});
    auto second = scheduler.NextExecutionPlan();
    ASSERT_TRUE(anchor_forward(second).extend_prefix_lens[0] == 6 && anchor_forward(second).input_lengths[0] == 7);
    anchor_done(scheduler, "seed", {42});
    ExecutionEvent finish;
    finish.With(forward::Finish{.request_id = "seed"});
    scheduler.Advance(std::move(finish));
    scheduler.NextExecutionPlan();
    spec.request_id = "replay";
    scheduler.SubmitRequests({spec});
    auto replay = scheduler.NextExecutionPlan();
    ASSERT_TRUE(anchor_forward(replay).extend_prefix_lens[0] > 6);
    anchor_done(scheduler, "replay", {42});
}
}  // namespace tokenspeed::test
