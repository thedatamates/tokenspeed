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

#include <algorithm>

namespace tokenspeed::test {
namespace {
SchedulerConfig WidthConfig(std::int32_t budget) {
    SchedulerConfig config{};
    config.prefix_granularity = 2;
    config.decode_input_tokens = 6;
    config.max_scheduled_tokens = budget;
    config.max_batch_size = 8;
    config.device_allocator.total_pages = 128;
    config.host_allocator.total_pages = 0;
    config.disable_l2_cache = true;
    config.disable_prefix_cache = true;
    config.enable_l3_storage = false;
    config.cache_groups.push_back(CacheGroupConfig{.group_id = "history",
                                                   .block_granularity = 2,
                                                   .total_pages = 128,
                                                   .retention = CacheGroupConfig::Retention::FullHistory,
                                                   .family = CacheGroupFamily::History});
    return config;
}
const ForwardBatch& FirstBatch(const ExecutionPlan& plan) {
    for (const auto& operation : plan.Operations())
        if (const auto* batch = std::get_if<ForwardBatch>(&operation)) return *batch;
    throw std::runtime_error("missing forward batch");
}
void SubmitWidthRequest(Scheduler& scheduler, const std::string& id, std::vector<std::int32_t> tokens) {
    RequestSpec request{};
    request.request_id = id;
    request.tokens = std::move(tokens);
    request.max_new_tokens = 0;
    scheduler.SubmitRequests({request});
}
void CommitWidthTokens(Scheduler& scheduler, const std::string& id, std::vector<std::int32_t> tokens) {
    ExecutionEvent event;
    event.With(forward::ExtendResult{.request_id = id, .tokens = std::move(tokens)});
    scheduler.Advance(event);
}
void SelectWidth(Scheduler& scheduler, const std::string& id, std::int32_t width, std::int32_t fallback) {
    ExecutionEvent event;
    event.With(forward::UpdateDecodeWidth{.request_id = id, .input_tokens = width, .fallback_input_tokens = fallback});
    scheduler.Advance(event);
}
void FinishWidthRequest(Scheduler& scheduler, const std::string& id) {
    ExecutionEvent event;
    event.With(forward::Finish{.request_id = id});
    scheduler.Advance(event);
}
}  // namespace

TEST(SelectedDecodeWidth, PartialCommitRetractionAndRestore) {
    Scheduler scheduler(WidthConfig(64));
    SubmitWidthRequest(scheduler, "a", {1, 2, 3, 4});
    SelectWidth(scheduler, "a", 1, 1);
    auto plan = scheduler.NextExecutionPlan();
    EXPECT_EQ(FirstBatch(plan).input_lengths, std::vector<std::int32_t>({4}));
    EXPECT_EQ(FirstBatch(plan).block_tables.at("history")[0].size(), 3);
    CommitWidthTokens(scheduler, "a", {10});
    SelectWidth(scheduler, "a", 6, 1);
    plan = scheduler.NextExecutionPlan();
    EXPECT_EQ(FirstBatch(plan).input_lengths, std::vector<std::int32_t>({6}));
    EXPECT_GE(FirstBatch(plan).block_tables.at("history")[0].size(), 5);
    EXPECT_THROW(SelectWidth(scheduler, "a", 1, 1), std::invalid_argument);
    CommitWidthTokens(scheduler, "a", {11, 12});
    SelectWidth(scheduler, "a", 1, 1);
    plan = scheduler.NextExecutionPlan();
    EXPECT_EQ(FirstBatch(plan).input_lengths, std::vector<std::int32_t>({1}));
    CommitWidthTokens(scheduler, "a", {13});
    SelectWidth(scheduler, "a", 6, 1);
    plan = scheduler.NextExecutionPlan();
    EXPECT_EQ(FirstBatch(plan).input_lengths, std::vector<std::int32_t>({6}));
    CommitWidthTokens(scheduler, "a", {14, 15, 16, 17, 18, 19});
    ExecutionEvent retract;
    retract.With(forward::Retract{.request_id = "a"});
    scheduler.Advance(retract);
    plan = scheduler.NextExecutionPlan();
    const std::vector<std::int32_t> history{1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    const auto& batch = FirstBatch(plan);
    ASSERT_EQ(batch.extend_prefix_lens.size(), 1);
    const auto start = batch.extend_prefix_lens[0];
    ASSERT_GE(start, 0);
    ASSERT_LE(start, history.size());
    EXPECT_EQ(batch.input_ids, std::vector<std::int32_t>(history.begin() + start, history.end()));
    CommitWidthTokens(scheduler, "a", {20});
    plan = scheduler.NextExecutionPlan();
    EXPECT_EQ(FirstBatch(plan).input_lengths, std::vector<std::int32_t>({1}));
    CommitWidthTokens(scheduler, "a", {21});
    SelectWidth(scheduler, "a", 6, 1);
    plan = scheduler.NextExecutionPlan();
    EXPECT_EQ(FirstBatch(plan).input_lengths, std::vector<std::int32_t>({6}));
    CommitWidthTokens(scheduler, "a", {22});
    FinishWidthRequest(scheduler, "a");
    scheduler.NextExecutionPlan();
}

TEST(SelectedDecodeWidth, MixedBudgetSkipsNonfittingWideRequest) {
    Scheduler scheduler(WidthConfig(7));
    for (const auto* id : {"a", "b", "c"}) {
        SubmitWidthRequest(scheduler, id, {1, 2});
        SelectWidth(scheduler, id, 1, 1);
    }
    scheduler.NextExecutionPlan();
    for (const auto* id : {"a", "b", "c"}) CommitWidthTokens(scheduler, id, {10});
    SelectWidth(scheduler, "a", 6, 1);
    SelectWidth(scheduler, "b", 6, 1);
    SelectWidth(scheduler, "c", 1, 1);
    const auto plan = scheduler.NextExecutionPlan();
    EXPECT_EQ(FirstBatch(plan).request_ids, std::vector<std::string>({"a", "c"}));
    EXPECT_EQ(FirstBatch(plan).input_lengths, std::vector<std::int32_t>({6, 1}));
    CommitWidthTokens(scheduler, "a", {11});
    CommitWidthTokens(scheduler, "c", {11});
    for (const auto* id : {"a", "b", "c"}) FinishWidthRequest(scheduler, id);
    scheduler.NextExecutionPlan();
}

TEST(SelectedDecodeWidth, InvalidEventsDoNotChangeSelection) {
    Scheduler scheduler(WidthConfig(64));
    SubmitWidthRequest(scheduler, "a", {1, 2});
    SelectWidth(scheduler, "a", 1, 1);
    scheduler.NextExecutionPlan();
    CommitWidthTokens(scheduler, "a", {10});
    EXPECT_THROW(SelectWidth(scheduler, "a", 0, 1), std::invalid_argument);
    EXPECT_THROW(SelectWidth(scheduler, "a", 7, 1), std::invalid_argument);
    EXPECT_THROW(SelectWidth(scheduler, "a", 6, 0), std::invalid_argument);
    EXPECT_THROW(SelectWidth(scheduler, "a", 1, 6), std::invalid_argument);
    EXPECT_THROW(SelectWidth(scheduler, "missing", 1, 1), std::invalid_argument);
    const auto plan = scheduler.NextExecutionPlan();
    EXPECT_EQ(FirstBatch(plan).input_lengths, std::vector<std::int32_t>({1}));
    CommitWidthTokens(scheduler, "a", {11});
    FinishWidthRequest(scheduler, "a");
    EXPECT_THROW(SelectWidth(scheduler, "a", 1, 1), std::invalid_argument);
    scheduler.NextExecutionPlan();
}

TEST(SelectedDecodeWidth, NoOverridePreservesConfiguredWidthAndNewLifetime) {
    Scheduler scheduler(WidthConfig(64));
    SubmitWidthRequest(scheduler, "a", {1, 2});
    scheduler.NextExecutionPlan();
    CommitWidthTokens(scheduler, "a", {10});
    auto plan = scheduler.NextExecutionPlan();
    EXPECT_EQ(FirstBatch(plan).input_lengths, std::vector<std::int32_t>({6}));
    CommitWidthTokens(scheduler, "a", {11});
    SelectWidth(scheduler, "a", 1, 1);
    FinishWidthRequest(scheduler, "a");
    scheduler.NextExecutionPlan();
    SubmitWidthRequest(scheduler, "a", {1, 2});
    scheduler.NextExecutionPlan();
    CommitWidthTokens(scheduler, "a", {10});
    plan = scheduler.NextExecutionPlan();
    EXPECT_EQ(FirstBatch(plan).input_lengths, std::vector<std::int32_t>({6}));
    CommitWidthTokens(scheduler, "a", {11});
    FinishWidthRequest(scheduler, "a");
    scheduler.NextExecutionPlan();
}

namespace {
struct RetainedCapacityResult {
    std::int32_t generated{1};
    std::size_t initial_pages{0};
    std::size_t high_water_pages{0};
    bool empty_plan{false};
};

RetainedCapacityResult RunRetainedCapacityCase(std::optional<std::int32_t> selected_width,
                                               std::int32_t retained_per_step, std::int32_t max_decode_steps) {
    auto config = WidthConfig(64);
    config.max_batch_size = 1;
    config.device_allocator.total_pages = 24;
    config.cache_groups.front().total_pages = 24;
    Scheduler scheduler(config);
    RequestSpec request{};
    request.request_id = "reserved-generation";
    request.tokens = {1, 2, 3, 4};
    request.max_new_tokens = 32;
    scheduler.SubmitRequests({request});
    if (selected_width) SelectWidth(scheduler, request.request_id, 1, 1);
    auto plan = scheduler.NextExecutionPlan();
    RetainedCapacityResult result;
    result.initial_pages = FirstBatch(plan).block_tables.at("history").front().size();
    result.high_water_pages = result.initial_pages;
    EXPECT_EQ(result.initial_pages, 18);
    CommitWidthTokens(scheduler, request.request_id, {10});
    if (selected_width) SelectWidth(scheduler, request.request_id, *selected_width, 1);
    for (std::int32_t step = 0; step < max_decode_steps && result.generated < 32; ++step) {
        plan = scheduler.NextExecutionPlan();
        const auto& batch = FirstBatch(plan);
        if (batch.request_ids.empty()) {
            result.empty_plan = true;
            break;
        }
        EXPECT_EQ(batch.request_ids, std::vector<std::string>({request.request_id}));
        EXPECT_EQ(batch.input_lengths, std::vector<std::int32_t>({selected_width.value_or(6)}));
        const auto pages = batch.block_tables.at("history").front().size();
        result.high_water_pages = std::max(result.high_water_pages, pages);
        if (selected_width) {
            // There are 46 usable physical positions: full generation plus enough q6
            // lookahead. Rejecting candidates must not spend them permanently.
            const auto input_end = 4 + result.generated - 1 + *selected_width;
            const auto needed_pages = std::max(result.initial_pages, static_cast<std::size_t>((input_end + 1) / 2));
            EXPECT_LE(pages, needed_pages)
                << "generated=" << result.generated << " retained_per_step=" << retained_per_step;
        }
        const auto retained = std::min(retained_per_step, 32 - result.generated);
        CommitWidthTokens(scheduler, request.request_id, std::vector<std::int32_t>(retained, 11));
        result.generated += retained;
    }
    FinishWidthRequest(scheduler, request.request_id);
    scheduler.NextExecutionPlan();
    EXPECT_EQ(scheduler.ActiveLcmBlocks(), 0);
    return result;
}
}  // namespace

TEST(SelectedDecodeWidth, PartialCommitDoesNotConsumeRejectedReservationCapacity) {
    for (const auto retained : {1, 3}) {
        SCOPED_TRACE(retained);
        const auto result = RunRetainedCapacityCase(6, retained, 64);
        EXPECT_FALSE(result.empty_plan) << "stranded after " << result.generated << " outputs";
        EXPECT_EQ(result.generated, 32);
    }
}

TEST(SelectedDecodeWidth, SingleRowCommitCompletesWithinGenerationReservation) {
    const auto result = RunRetainedCapacityCase(1, 1, 64);
    EXPECT_FALSE(result.empty_plan);
    EXPECT_EQ(result.generated, 32);
    EXPECT_EQ(result.high_water_pages, result.initial_pages);
}

TEST(SelectedDecodeWidth, FullSixRowCommitCompletesWithBoundedLookahead) {
    const auto result = RunRetainedCapacityCase(6, 6, 64);
    EXPECT_FALSE(result.empty_plan);
    EXPECT_EQ(result.generated, 32);
    EXPECT_LE(result.high_water_pages, 20);
}

TEST(SelectedDecodeWidth, LegacyReserveFeedbackDoesNotInferSelectedWidthRefund) {
    // No explicit width selection: legacy callers own their reserve feedback.
    // The selected-width correction must not reinterpret their result length.
    const auto result = RunRetainedCapacityCase(std::nullopt, 1, 6);
    EXPECT_FALSE(result.empty_plan);
    EXPECT_EQ(result.generated, 7);
    EXPECT_EQ(result.high_water_pages, 20);
}

TEST(SelectedDecodeWidth, ZeroRetainedResultReturnsItsWholeExplicitReservation) {
    auto config = WidthConfig(64);
    config.device_allocator.total_pages = 24;
    config.cache_groups.front().total_pages = 24;
    Scheduler scheduler(config);
    scheduler.SubmitRequests({RequestSpec{.request_id = "zero", .tokens = {1, 2, 3, 4}, .max_new_tokens = 32}});
    SelectWidth(scheduler, "zero", 1, 1);
    auto plan = scheduler.NextExecutionPlan();
    const auto pages = FirstBatch(plan).block_tables.at("history").front().size();
    CommitWidthTokens(scheduler, "zero", {10});
    SelectWidth(scheduler, "zero", 6, 1);
    for (std::int32_t step = 0; step < 12; ++step) {
        plan = scheduler.NextExecutionPlan();
        const auto& batch = FirstBatch(plan);
        ASSERT_EQ(batch.request_ids, std::vector<std::string>({"zero"}));
        EXPECT_EQ(batch.block_tables.at("history").front().size(), pages);
        CommitWidthTokens(scheduler, "zero", {});
    }
    FinishWidthRequest(scheduler, "zero");
    scheduler.NextExecutionPlan();
}

TEST(SelectedDecodeWidth, PipelinedEmptyPrefillResultsDoNotRefundDecodeReservations) {
    auto config = WidthConfig(8);
    config.device_allocator.total_pages = 32;
    config.cache_groups.front().total_pages = 32;
    Scheduler scheduler(config);
    scheduler.SubmitRequests({RequestSpec{.request_id = "chunks", .tokens = MakeTokens(20), .max_new_tokens = 32}});
    SelectWidth(scheduler, "chunks", 1, 1);
    for (const auto expected : {8, 8, 4}) {
        const auto plan = scheduler.NextExecutionPlan();
        EXPECT_EQ(FirstBatch(plan).input_lengths, std::vector<std::int32_t>({expected}));
    }
    CommitWidthTokens(scheduler, "chunks", {});
    CommitWidthTokens(scheduler, "chunks", {});
    CommitWidthTokens(scheduler, "chunks", {10});
    SelectWidth(scheduler, "chunks", 6, 1);
    for (std::int32_t step = 0; step < 12; ++step) {
        const auto plan = scheduler.NextExecutionPlan();
        const auto& batch = FirstBatch(plan);
        ASSERT_EQ(batch.request_ids, std::vector<std::string>({"chunks"}));
        EXPECT_EQ(batch.block_tables.at("history").front().size(), 26);
        CommitWidthTokens(scheduler, "chunks", {11});
    }
    FinishWidthRequest(scheduler, "chunks");
    scheduler.NextExecutionPlan();
}

TEST(SelectedDecodeWidth, PartialCommitsFitExactBoundedGenerationWithHorizonFallback) {
    for (const auto retained_per_step : {1, 3}) {
        SCOPED_TRACE(retained_per_step);
        auto config = WidthConfig(64);
        config.prefix_granularity = 4;
        config.device_allocator.total_pages = 130;
        config.cache_groups.front().block_granularity = 4;
        config.cache_groups.front().total_pages = 130;
        Scheduler scheduler(config);
        scheduler.SubmitRequests({RequestSpec{.request_id = "bounded", .tokens = {1, 2, 3, 4}, .max_new_tokens = 512}});
        SelectWidth(scheduler, "bounded", 1, 1);
        auto plan = scheduler.NextExecutionPlan();
        ASSERT_EQ(FirstBatch(plan).block_tables.at("history").front().size(), 129);
        CommitWidthTokens(scheduler, "bounded", {10});
        std::int32_t generated = 1;
        while (generated < 512) {
            const auto width = 512 - generated >= 6 ? 6 : 1;
            SelectWidth(scheduler, "bounded", width, 1);
            plan = scheduler.NextExecutionPlan();
            const auto& batch = FirstBatch(plan);
            ASSERT_EQ(batch.request_ids, std::vector<std::string>({"bounded"})) << "generated=" << generated;
            EXPECT_EQ(batch.input_lengths, std::vector<std::int32_t>({width}));
            EXPECT_EQ(batch.block_tables.at("history").front().size(), 129);
            const auto retained = width == 6 ? retained_per_step : 1;
            CommitWidthTokens(scheduler, "bounded", std::vector<std::int32_t>(retained, 11));
            generated += retained;
        }
        FinishWidthRequest(scheduler, "bounded");
        scheduler.NextExecutionPlan();
        EXPECT_EQ(scheduler.ActiveLcmBlocks(), 0);
    }
}

TEST(SelectedDecodeWidth, PartialCommitRefundWorksAcrossSparseReplayableGroups) {
    auto config = WidthConfig(64);
    config.prefix_granularity = 8;
    config.device_allocator.total_pages = 128;
    config.cache_groups.front().block_granularity = 8;
    config.cache_groups.front().total_pages = 128;
    config.disable_prefix_cache = false;
    CacheGroupConfig replay{};
    replay.group_id = "replay";
    replay.block_granularity = 4;
    replay.total_pages = 128;
    replay.retention = CacheGroupConfig::Retention::SlidingWindow;
    replay.sliding_window_tokens = 16;
    replay.replayable = true;
    config.cache_groups.push_back(replay);
    Scheduler scheduler(config);
    const auto prompt = MakeTokens(32);
    scheduler.SubmitRequests({RequestSpec{.request_id = "seed", .tokens = prompt, .max_new_tokens = 64}});
    SelectWidth(scheduler, "seed", 1, 1);
    scheduler.NextExecutionPlan();
    CommitWidthTokens(scheduler, "seed", {10});
    FinishWidthRequest(scheduler, "seed");
    scheduler.NextExecutionPlan();
    scheduler.SubmitRequests({RequestSpec{.request_id = "hit", .tokens = prompt, .max_new_tokens = 64}});
    SelectWidth(scheduler, "hit", 1, 1);
    auto plan = scheduler.NextExecutionPlan();
    const auto& prefill = FirstBatch(plan);
    ASSERT_EQ(prefill.request_ids, std::vector<std::string>({"hit"}));
    EXPECT_GT(prefill.extend_prefix_lens.front(), 0);
    const auto& replay_pages = prefill.block_tables.at("replay").front();
    EXPECT_TRUE(std::ranges::any_of(replay_pages, [](std::int32_t page) { return page == 0; }));
    CommitWidthTokens(scheduler, "hit", {10});
    SelectWidth(scheduler, "hit", 6, 1);
    for (std::int32_t step = 0; step < 20; ++step) {
        plan = scheduler.NextExecutionPlan();
        const auto& batch = FirstBatch(plan);
        ASSERT_EQ(batch.request_ids, std::vector<std::string>({"hit"}));
        EXPECT_EQ(batch.block_tables.at("history").front().size(), 12);
        CommitWidthTokens(scheduler, "hit", {11});
    }
    FinishWidthRequest(scheduler, "hit");
    scheduler.NextExecutionPlan();
    EXPECT_EQ(scheduler.ActiveLcmBlocks(), 0);
}
}  // namespace tokenspeed::test
