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
}  // namespace tokenspeed::test
