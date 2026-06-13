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

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "integration_test_helper.h"

namespace tokenspeed::test {

// ============================================================
//  Block-diffusion scheduling: Denoising ⇄ Committing lifecycle,
//  per-canvas page reservations, step backstop, retraction.
// ============================================================

// A diffusion row extracted from a FlatForwardOperation, joined with its
// per-request columns.
struct DiffusionRow {
    std::string id;
    std::int32_t pool_index;
    DiffusionKind kind;
    std::int32_t canvas_len;
    std::int32_t committed_len;
    std::int32_t steps_taken;
    std::int64_t pass_epoch;
    std::int32_t write_page_begin;
    std::int32_t write_page_count;
    std::int32_t input_length;
    std::vector<std::int32_t> occupied_pages;
    std::int32_t begin;
    std::int32_t size;
};

class BlockDiffusionTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.page_size = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 32;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.disable_l2_cache = true;
        return cfg;
    }

    RequestSpec MakeDiffusionSpec(const std::string& id, std::int32_t num_prompt_pages, std::int32_t canvas_length,
                                  std::int32_t max_denoising_steps, std::int32_t max_new_tokens, token_t start = 1) {
        auto spec = MakeRequestSpec(id, num_prompt_pages, start);
        spec.block_diffusion = BlockDiffusionParams{
            .canvas_length = canvas_length,
            .max_denoising_steps = max_denoising_steps,
            .max_new_tokens = max_new_tokens,
        };
        return spec;
    }

    // Echo an explicit pass_epoch (used by the staleness tests).
    void SendDenoiseResult(const std::string& request_id, bool converged, std::int64_t pass_epoch) {
        ExecutionEvent event;
        event.With(ForwardEvent{forward::DenoiseResult{
            .request_id = request_id,
            .converged = converged,
            .pass_epoch = pass_epoch,
        }});
        scheduler_->Advance(std::move(event));
    }

    // Behave like a correct executor: echo the epoch of the most recent
    // denoise row scheduled for this request (tracked by PlanOnce below).
    void SendDenoiseResult(const std::string& request_id, bool converged) {
        SendDenoiseResult(request_id, converged, last_denoise_epoch_.at(request_id));
    }

    // Shadows the base helper to record each scheduled denoise pass's epoch,
    // so tests can echo it like a real executor would.
    ExecutionPlan PlanOnce() {
        auto plan = SchedulerTestSuite::PlanOnce();
        if (const auto* fwd = GetForwardOp(plan)) {
            for (const auto& row : DiffusionRows(*fwd)) {
                if (row.kind == DiffusionKind::kDenoise) {
                    last_denoise_epoch_[row.id] = row.pass_epoch;
                }
            }
        }
        return plan;
    }

    void SendAbort(const std::string& request_id) {
        ExecutionEvent event;
        event.With(ForwardEvent{forward::Abort{
            .request_id = request_id,
        }});
        scheduler_->Advance(std::move(event));
    }

    static const FlatForwardOperation* GetForwardOp(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* f = std::get_if<FlatForwardOperation>(&op)) return f;
        }
        return nullptr;
    }

    static const FlatWriteBackOperation* GetWriteBack(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* cop = std::get_if<CacheOperation>(&op)) {
                if (auto* wb = std::get_if<FlatWriteBackOperation>(cop)) return wb;
            }
        }
        return nullptr;
    }

    static const FlatLoadBackOperation* GetLoadBack(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* cop = std::get_if<CacheOperation>(&op)) {
                if (auto* lb = std::get_if<FlatLoadBackOperation>(cop)) return lb;
            }
        }
        return nullptr;
    }

    // Joins the diffusion SoA columns with the per-request columns. Also
    // asserts the row partition invariant: diffusion rows are the batch tail.
    static std::vector<DiffusionRow> DiffusionRows(const FlatForwardOperation& fwd) {
        const std::size_t num_diffusion = fwd.num_diffusion();
        const std::size_t base = fwd.request_ids.size() - num_diffusion;
        EXPECT_GE(fwd.request_ids.size(), num_diffusion);
        std::vector<DiffusionRow> rows;
        for (std::size_t i = 0; i < num_diffusion; ++i) {
            rows.push_back(DiffusionRow{
                .id = fwd.request_ids[base + i],
                .pool_index = fwd.request_pool_indices[base + i],
                .kind = fwd.diffusion_kinds[i],
                .canvas_len = fwd.diffusion_canvas_lens[i],
                .committed_len = fwd.diffusion_committed_lens[i],
                .steps_taken = fwd.diffusion_steps_taken[i],
                .pass_epoch = fwd.diffusion_pass_epochs[i],
                .write_page_begin = fwd.diffusion_write_page_begins[i],
                .write_page_count = fwd.diffusion_write_page_counts[i],
                .input_length = fwd.input_lengths[base + i],
                .occupied_pages = fwd.occupied_pages[base + i],
                .begin = fwd.begins[base + i],
                .size = fwd.sizes[base + i],
            });
        }
        return rows;
    }

    static const DiffusionRow* FindRow(const std::vector<DiffusionRow>& rows, const std::string& id) {
        for (const auto& row : rows) {
            if (row.id == id) return &row;
        }
        return nullptr;
    }

    // Plan exactly one diffusion row for `id` and return it (asserts shape).
    DiffusionRow PlanSingleDiffusionRow(const std::string& id) {
        auto plan = PlanOnce();
        const auto* fwd = GetForwardOp(plan);
        EXPECT_NE(fwd, nullptr);
        auto rows = DiffusionRows(*fwd);
        EXPECT_EQ(rows.size(), 1u);
        EXPECT_EQ(rows[0].id, id);
        return rows[0];
    }

    // Epoch of the most recently scheduled denoise pass, per request.
    std::map<std::string, std::int64_t> last_denoise_epoch_;
};

// ------------------------------------------------------------
//  Obligation 1 — progress: full single-request lifecycle with exact
//  reservation/row bookkeeping, then multi-request interleaving with mixed
//  prompt/canvas lengths and per-request step counts.
// ------------------------------------------------------------

TEST_F(BlockDiffusionTestSuite, SingleRequest_FullLifecycle_TwoCanvases) {
    const auto baseline = scheduler_->AvailableKvPages();
    // prompt 4 tokens (2 pages), CL=8 (4 pages/canvas), 2 canvases.
    Submit(MakeDiffusionSpec("r1", /*num_prompt_pages=*/2, /*canvas_length=*/8, /*max_denoising_steps=*/4,
                             /*max_new_tokens=*/16));

    // Plan 1: prefill (extend row), no diffusion rows yet.
    {
        auto plan = PlanOnce();
        const auto* fwd = GetForwardOp(plan);
        ASSERT_NE(fwd, nullptr);
        ASSERT_EQ(fwd->request_ids.size(), 1u);
        EXPECT_EQ(fwd->num_extends(), 1u);
        EXPECT_EQ(fwd->num_diffusion(), 0u);
        EXPECT_EQ(fwd->input_lengths[0], 4);
    }

    // Plan 2: first denoise pass; entering Denoising acquires the canvas
    // reservation: ceil(8/2)=4 new pages on top of the 2 prompt pages.
    {
        auto row = PlanSingleDiffusionRow("r1");
        EXPECT_EQ(row.kind, DiffusionKind::kDenoise);
        EXPECT_EQ(row.canvas_len, 8);
        EXPECT_EQ(row.committed_len, 4);
        EXPECT_EQ(row.steps_taken, 0);  // executor must init the canvas
        EXPECT_EQ(row.pass_epoch, 1);   // scheduler-issued pass identity
        EXPECT_EQ(row.write_page_begin, 2);  // canvas span = the 4 reservation pages
        EXPECT_EQ(row.write_page_count, 0);  // denoise must not write KV
        EXPECT_EQ(row.input_length, 8);
        EXPECT_EQ(row.occupied_pages.size(), 6u);
        EXPECT_EQ(row.begin, 2);
        EXPECT_EQ(row.size, 4);
        EXPECT_EQ(scheduler_->AvailableKvPages(), baseline - 6);
    }

    // Pass 1 does not converge: one more denoise pass, no new pages.
    SendDenoiseResult("r1", false);
    {
        auto row = PlanSingleDiffusionRow("r1");
        EXPECT_EQ(row.kind, DiffusionKind::kDenoise);
        EXPECT_EQ(row.steps_taken, 1);
        EXPECT_EQ(row.pass_epoch, 2);  // strictly increasing per pass
        EXPECT_EQ(row.size, 0);
        EXPECT_EQ(row.occupied_pages.size(), 6u);
    }

    // Pass 2 converges: exactly one commit pass over the reserved pages.
    SendDenoiseResult("r1", true);
    {
        auto row = PlanSingleDiffusionRow("r1");
        EXPECT_EQ(row.kind, DiffusionKind::kCommit);
        EXPECT_EQ(row.canvas_len, 8);
        EXPECT_EQ(row.committed_len, 4);
        EXPECT_EQ(row.steps_taken, 2);
        EXPECT_EQ(row.pass_epoch, 3);  // commit passes consume epochs too
        EXPECT_EQ(row.write_page_begin, 2);  // commit writes exactly the canvas span
        EXPECT_EQ(row.write_page_count, 4);
        EXPECT_EQ(row.size, 0);  // commit writes into the existing reservation
        EXPECT_EQ(scheduler_->AvailableKvPages(), baseline - 6);
    }

    // Commit result: canvas tokens become committed history.
    SendForwardDone("r1", MakeTokens(8, 100));
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r1"), 12);

    // Next canvas: fresh reservation, steps restart at 0; the pass epoch
    // keeps increasing across canvases (it identifies passes, not steps).
    {
        auto row = PlanSingleDiffusionRow("r1");
        EXPECT_EQ(row.kind, DiffusionKind::kDenoise);
        EXPECT_EQ(row.committed_len, 12);
        EXPECT_EQ(row.steps_taken, 0);
        EXPECT_EQ(row.pass_epoch, 4);
        EXPECT_EQ(row.write_page_begin, 6);
        EXPECT_EQ(row.write_page_count, 0);
        EXPECT_EQ(row.begin, 6);
        EXPECT_EQ(row.size, 4);
        EXPECT_EQ(row.occupied_pages.size(), 10u);
    }

    SendDenoiseResult("r1", true);
    {
        auto row = PlanSingleDiffusionRow("r1");
        EXPECT_EQ(row.kind, DiffusionKind::kCommit);
        EXPECT_EQ(row.committed_len, 12);
        EXPECT_EQ(row.steps_taken, 1);
    }

    // Final commit: EOS at the generation budget.
    SendForwardDone("r1", MakeTokens(8, 200));
    SendFinish("r1");
    PlanOnce();  // erases the finished request
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r1"), -1);
    // Finished requests donate page-aligned committed KV to the prefix cache
    // (19 of 20 committed tokens → 9 pages, AR except-last convention); the
    // 10th held page was unused and returns to the pool.
    EXPECT_EQ(scheduler_->AvailableKvPages(), baseline - 9);
    EXPECT_EQ(scheduler_->ActiveKvPages(), 0u);
}

TEST_F(BlockDiffusionTestSuite, Progress_ConcurrentRequests_MixedLengthsAndStepCounts) {
    // r1: 1-page prompt, CL=4, converges at pass 2 of 5, single canvas.
    // r2: 3-page prompt, CL=8, never converges (backstop 2), two canvases.
    Submit(MakeDiffusionSpec("r1", 1, /*canvas_length=*/4, /*max_denoising_steps=*/5, /*max_new_tokens=*/4,
                             /*start=*/1));
    Submit(MakeDiffusionSpec("r2", 3, /*canvas_length=*/8, /*max_denoising_steps=*/2, /*max_new_tokens=*/16,
                             /*start=*/500));

    PlanOnce();  // prefills

    // Plans interleave denoise rows from both requests.
    {
        auto plan = PlanOnce();
        auto rows = DiffusionRows(*GetForwardOp(plan));
        ASSERT_EQ(rows.size(), 2u);
        const auto* r1 = FindRow(rows, "r1");
        const auto* r2 = FindRow(rows, "r2");
        ASSERT_NE(r1, nullptr);
        ASSERT_NE(r2, nullptr);
        EXPECT_EQ(r1->canvas_len, 4);
        EXPECT_EQ(r1->committed_len, 2);
        EXPECT_EQ(r1->steps_taken, 0);
        EXPECT_EQ(r2->canvas_len, 8);
        EXPECT_EQ(r2->committed_len, 6);
        EXPECT_EQ(r2->steps_taken, 0);
    }

    SendDenoiseResult("r1", false);
    SendDenoiseResult("r2", false);
    {
        auto plan = PlanOnce();
        auto rows = DiffusionRows(*GetForwardOp(plan));
        ASSERT_EQ(rows.size(), 2u);
        EXPECT_EQ(FindRow(rows, "r1")->steps_taken, 1);
        EXPECT_EQ(FindRow(rows, "r2")->steps_taken, 1);
    }

    // r1 converges; r2 hits its per-request backstop (steps == 2).
    SendDenoiseResult("r1", true);
    SendDenoiseResult("r2", false);
    {
        auto plan = PlanOnce();
        auto rows = DiffusionRows(*GetForwardOp(plan));
        ASSERT_EQ(rows.size(), 2u);
        const auto* r1 = FindRow(rows, "r1");
        const auto* r2 = FindRow(rows, "r2");
        EXPECT_EQ(r1->kind, DiffusionKind::kCommit);
        EXPECT_EQ(r1->steps_taken, 2);
        EXPECT_EQ(r2->kind, DiffusionKind::kCommit);
        EXPECT_EQ(r2->steps_taken, 2);
    }

    // r1 finishes at its budget; r2 continues to canvas 2.
    SendForwardDone("r1", MakeTokens(4, 100));
    SendFinish("r1");
    SendForwardDone("r2", MakeTokens(8, 600));
    {
        auto plan = PlanOnce();
        auto rows = DiffusionRows(*GetForwardOp(plan));
        ASSERT_EQ(rows.size(), 1u);
        EXPECT_EQ(rows[0].id, "r2");
        EXPECT_EQ(rows[0].kind, DiffusionKind::kDenoise);
        EXPECT_EQ(rows[0].committed_len, 14);
        EXPECT_EQ(rows[0].steps_taken, 0);
    }
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r1"), -1);

    SendDenoiseResult("r2", false);
    PlanOnce();
    SendDenoiseResult("r2", false);  // backstop again
    {
        auto row = PlanSingleDiffusionRow("r2");
        EXPECT_EQ(row.kind, DiffusionKind::kCommit);
        EXPECT_EQ(row.steps_taken, 2);
    }
    SendForwardDone("r2", MakeTokens(8, 700));
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r2"), -1);
    EXPECT_EQ(scheduler_->ActiveKvPages(), 0u);
}

// ------------------------------------------------------------
//  Obligation 2 — step backstop: an executor that never reports
//  converged=true still commits at max_denoising_steps, and the request
//  terminates at max_new_tokens without an executor Finish.
// ------------------------------------------------------------

TEST_F(BlockDiffusionTestSuite, Backstop_NeverConverges_CommitsAndTerminatesAtMaxNewTokens) {
    Submit(MakeDiffusionSpec("r1", 2, /*canvas_length=*/8, /*max_denoising_steps=*/3, /*max_new_tokens=*/8));
    PlanOnce();  // prefill

    // Exactly max_denoising_steps denoise passes are scheduled.
    for (std::int32_t step = 0; step < 3; ++step) {
        auto row = PlanSingleDiffusionRow("r1");
        EXPECT_EQ(row.kind, DiffusionKind::kDenoise);
        EXPECT_EQ(row.steps_taken, step);
        SendDenoiseResult("r1", false);
    }

    // The backstop forces the commit despite converged never being reported.
    {
        auto row = PlanSingleDiffusionRow("r1");
        EXPECT_EQ(row.kind, DiffusionKind::kCommit);
        EXPECT_EQ(row.steps_taken, 3);
    }

    // The fake executor reports the committed tokens but never Finish:
    // the scheduler terminates the request at max_new_tokens itself.
    SendForwardDone("r1", MakeTokens(8, 100));
    {
        auto plan = PlanOnce();
        const auto* fwd = GetForwardOp(plan);
        ASSERT_NE(fwd, nullptr);
        EXPECT_TRUE(fwd->request_ids.empty());
    }
    PlanOnce();  // erase
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r1"), -1);
    EXPECT_EQ(scheduler_->ActiveKvPages(), 0u);
}

// ------------------------------------------------------------
//  Obligation 3 — stop & free: EOS at the first / a middle / the (truncated)
//  last canvas releases the slot and the unused reservation pages. Exact
//  accounting: pages retained == full pages of (committed − 1) tokens (the
//  AR except-last prefix-cache convention); everything else returns to the
//  pool.
// ------------------------------------------------------------

TEST_F(BlockDiffusionTestSuite, StopAndFree_EosAtFirstCanvas_ReleasesUnusedReservation) {
    const auto baseline = scheduler_->AvailableKvPages();
    Submit(MakeDiffusionSpec("r1", 2, /*canvas_length=*/8, /*max_denoising_steps=*/4, /*max_new_tokens=*/24));
    PlanOnce();
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableKvPages(), baseline - 6);  // 2 prompt + 4 reservation
    SendDenoiseResult("r1", true);
    PlanOnce();  // commit

    // EOS after 3 of 8 canvas tokens: 5 reserved token slots go unused.
    SendForwardDone("r1", MakeTokens(3, 100));
    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r1"), -1);
    // committed 4+3=7 tokens → 3 full pages retained as prefix cache; the
    // other 3 held pages (incl. the unused reservation tail) are freed.
    EXPECT_EQ(scheduler_->AvailableKvPages(), baseline - 3);
    EXPECT_EQ(scheduler_->ActiveKvPages(), 0u);
}

TEST_F(BlockDiffusionTestSuite, StopAndFree_EosMidStream_SecondCanvas) {
    const auto baseline = scheduler_->AvailableKvPages();
    Submit(MakeDiffusionSpec("r1", 2, /*canvas_length=*/8, /*max_denoising_steps=*/4, /*max_new_tokens=*/24));
    PlanOnce();
    PlanOnce();
    SendDenoiseResult("r1", true);
    PlanOnce();
    SendForwardDone("r1", MakeTokens(8, 100));  // canvas 1 commits in full

    PlanOnce();  // canvas 2 entry: +4 pages
    EXPECT_EQ(scheduler_->AvailableKvPages(), baseline - 10);
    SendDenoiseResult("r1", true);
    PlanOnce();
    SendForwardDone("r1", MakeTokens(5, 200));  // EOS mid-canvas
    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r1"), -1);
    // committed 4+8+5=17 tokens → 8 full pages retained, 2 freed.
    EXPECT_EQ(scheduler_->AvailableKvPages(), baseline - 8);
}

TEST_F(BlockDiffusionTestSuite, StopAndFree_TruncatedLastCanvas) {
    const auto baseline = scheduler_->AvailableKvPages();
    // max_new_tokens=12 with CL=8: canvas 2 is truncated to 4 tokens.
    Submit(MakeDiffusionSpec("r1", 2, /*canvas_length=*/8, /*max_denoising_steps=*/4, /*max_new_tokens=*/12));
    PlanOnce();
    PlanOnce();
    SendDenoiseResult("r1", true);
    PlanOnce();
    SendForwardDone("r1", MakeTokens(8, 100));

    {
        auto row = PlanSingleDiffusionRow("r1");
        EXPECT_EQ(row.kind, DiffusionKind::kDenoise);
        EXPECT_EQ(row.canvas_len, 4);  // truncated to the remaining budget
        EXPECT_EQ(row.input_length, 4);
        EXPECT_EQ(row.size, 2);  // reservation sized for the truncated canvas
    }
    SendDenoiseResult("r1", true);
    {
        auto row = PlanSingleDiffusionRow("r1");
        EXPECT_EQ(row.kind, DiffusionKind::kCommit);
        EXPECT_EQ(row.canvas_len, 4);
        // Truncated canvas: the write span shrinks with it (2 pages of 4 tokens).
        EXPECT_EQ(row.write_page_begin, 6);
        EXPECT_EQ(row.write_page_count, 2);
    }
    SendForwardDone("r1", MakeTokens(4, 200));
    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r1"), -1);
    // committed 16 tokens → 7 full pages retained (except-last), 1 freed.
    EXPECT_EQ(scheduler_->AvailableKvPages(), baseline - 7);
}

// ------------------------------------------------------------
//  Obligation 4 — isolation: concurrent requests own disjoint page sets and
//  pool slots; aborting one mid-denoise frees its pages and leaves the
//  others' plans unchanged.
// ------------------------------------------------------------

TEST_F(BlockDiffusionTestSuite, Isolation_DisjointPagesAndSlots_AbortMidDenoise) {
    const auto baseline = scheduler_->AvailableKvPages();
    Submit(MakeDiffusionSpec("r1", 2, 8, 4, 16, /*start=*/1));
    Submit(MakeDiffusionSpec("r2", 2, 8, 4, 16, /*start=*/100));
    Submit(MakeDiffusionSpec("r3", 2, 8, 4, 16, /*start=*/200));
    PlanOnce();  // prefills

    std::vector<DiffusionRow> first_rows;
    {
        auto plan = PlanOnce();
        first_rows = DiffusionRows(*GetForwardOp(plan));
        ASSERT_EQ(first_rows.size(), 3u);

        std::set<std::int32_t> pool_indices;
        std::set<std::int32_t> all_pages;
        std::size_t total_pages = 0;
        for (const auto& row : first_rows) {
            pool_indices.insert(row.pool_index);
            all_pages.insert(row.occupied_pages.begin(), row.occupied_pages.end());
            total_pages += row.occupied_pages.size();
            EXPECT_EQ(row.occupied_pages.size(), 6u);
        }
        EXPECT_EQ(pool_indices.size(), 3u);          // distinct request slots
        EXPECT_EQ(all_pages.size(), total_pages);    // page sets are disjoint
    }
    EXPECT_EQ(scheduler_->AvailableKvPages(), baseline - 18);

    // Abort r2 mid-denoise (its pass is still in flight).
    SendDenoiseResult("r1", false);
    SendDenoiseResult("r3", false);
    SendAbort("r2");
    EXPECT_EQ(scheduler_->AvailableKvPages(), baseline - 12);  // r2's 6 pages freed

    {
        auto plan = PlanOnce();
        auto rows = DiffusionRows(*GetForwardOp(plan));
        ASSERT_EQ(rows.size(), 2u);
        EXPECT_EQ(FindRow(rows, "r2"), nullptr);
        for (const auto& id : {"r1", "r3"}) {
            const auto* row = FindRow(rows, id);
            const auto* before = FindRow(first_rows, id);
            ASSERT_NE(row, nullptr);
            // Survivors are unaffected: same slot, same pages, advanced step.
            EXPECT_EQ(row->pool_index, before->pool_index);
            EXPECT_EQ(row->occupied_pages, before->occupied_pages);
            EXPECT_EQ(row->steps_taken, 1);
        }
    }

    // Aborts release everything — the pool returns to baseline exactly.
    SendAbort("r1");
    SendAbort("r3");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableKvPages(), baseline);
    EXPECT_EQ(scheduler_->ActiveKvPages(), 0u);
}

// ------------------------------------------------------------
//  Obligation 5 — mixed AR + diffusion batches: the row partition
//  [ extends | decodes | diffusion ] holds, with the AR arrays unchanged.
//  (AR-only byte-identity is pinned by the untouched pre-existing suite.)
// ------------------------------------------------------------

class BlockDiffusionMixedTestSuite : public BlockDiffusionTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = BlockDiffusionTestSuite::MakeConfig();
        cfg.enable_mixed_prefill_decode = true;
        return cfg;
    }
};

TEST_F(BlockDiffusionMixedTestSuite, MixedBatch_PartitionInvariant) {
    Submit(MakeRequestSpec("a1", 2, /*start=*/1));                  // AR
    Submit(MakeDiffusionSpec("d1", 2, 8, 4, 16, /*start=*/100));    // diffusion

    // Plan 1: two prefills, no diffusion rows.
    {
        auto plan = PlanOnce();
        const auto* fwd = GetForwardOp(plan);
        ASSERT_NE(fwd, nullptr);
        EXPECT_EQ(fwd->num_extends(), 2u);
        EXPECT_EQ(fwd->num_diffusion(), 0u);
    }
    SendForwardDone("a1", {42});  // AR first token

    // Plan 2: AR decode + diffusion denoise. Submit a third AR request so the
    // mixed plan exercises all three partitions at once.
    Submit(MakeRequestSpec("a2", 2, /*start=*/300));
    {
        auto plan = PlanOnce();
        const auto* fwd = GetForwardOp(plan);
        ASSERT_NE(fwd, nullptr);
        ASSERT_EQ(fwd->request_ids.size(), 3u);
        // [ extends | decodes | diffusion ]
        EXPECT_EQ(fwd->num_extends(), 1u);
        EXPECT_EQ(fwd->request_ids[0], "a2");
        EXPECT_EQ(fwd->request_ids[1], "a1");
        EXPECT_EQ(fwd->request_ids[2], "d1");
        EXPECT_EQ(fwd->num_diffusion(), 1u);
        // AR decode arrays cover decodes only; diffusion arrays cover the tail.
        EXPECT_EQ(fwd->decode_input_ids.size(), 1u);
        EXPECT_EQ(fwd->hist_token_lens.size(), 1u);
        EXPECT_EQ(fwd->input_lengths[1], 1);
        EXPECT_EQ(fwd->input_lengths[2], 8);
        EXPECT_EQ(fwd->diffusion_kinds[0], DiffusionKind::kDenoise);
        EXPECT_EQ(fwd->diffusion_committed_lens[0], 4);
        // input_ids carries extend tokens only (a2's prefill chunk).
        EXPECT_EQ(fwd->input_ids.size(), 4u);
    }

    // Both modes advance independently to completion.
    SendForwardDone("a1", {43});
    SendDenoiseResult("d1", true);
    {
        auto plan = PlanOnce();
        const auto* fwd = GetForwardOp(plan);
        auto rows = DiffusionRows(*fwd);
        ASSERT_EQ(rows.size(), 1u);
        EXPECT_EQ(rows[0].kind, DiffusionKind::kCommit);
    }
    SendFinish("a1");
    SendFinish("a2");
    SendForwardDone("d1", MakeTokens(8, 500));
    SendFinish("d1");
    PlanOnce();
    PlanOnce();
    EXPECT_EQ(scheduler_->GetRequestTokenSize("d1"), -1);
    EXPECT_EQ(scheduler_->ActiveKvPages(), 0u);
}

// ------------------------------------------------------------
//  Obligation 6 — backpressure: KV-pressure retraction of a diffusion
//  request discards canvas progress, frees the canvas reservation, writes
//  committed KV back, and the request later resumes from step 0 (fresh
//  reservation, fresh canvas) and completes.
// ------------------------------------------------------------

class BlockDiffusionBackpressureTestSuite : public BlockDiffusionTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = BlockDiffusionTestSuite::MakeConfig();
        // Usable device pages: 15 (page 0 reserved). Canvas-1 peak is
        // r1(3+4) + r2(2+4) = 13 pages; canvas-2 entry needs 4 more → starves.
        cfg.device_allocator.total_pages = 16;
        cfg.disable_l2_cache = false;  // retraction needs the host writeback path
        return cfg;
    }
};

TEST_F(BlockDiffusionBackpressureTestSuite, Retraction_DiscardsCanvas_ResumesFromStepZero) {
    Submit(MakeDiffusionSpec("r1", 3, 8, 4, 16, /*start=*/1));
    Submit(MakeDiffusionSpec("r2", 2, 8, 4, 16, /*start=*/300));
    PlanOnce();  // prefills
    {
        auto plan = PlanOnce();  // both enter canvas 1
        EXPECT_EQ(DiffusionRows(*GetForwardOp(plan)).size(), 2u);
    }
    const std::int64_t pre_retract_epoch = last_denoise_epoch_.at("r1");
    SendDenoiseResult("r1", true);
    SendDenoiseResult("r2", true);
    PlanOnce();  // both commit
    SendForwardDone("r1", MakeTokens(8, 100));
    SendForwardDone("r2", MakeTokens(8, 400));

    // Canvas-2 entry: neither reservation fits → KV pressure. The scheduler
    // retracts the longest request (r1, 14 tokens) and emits its writeback.
    cache_op_id retract_op_id{};
    {
        auto plan = PlanOnce();
        const auto* fwd = GetForwardOp(plan);
        ASSERT_NE(fwd, nullptr);
        EXPECT_TRUE(fwd->request_ids.empty());
        const auto* wb = GetWriteBack(plan);
        ASSERT_NE(wb, nullptr);
        ASSERT_EQ(wb->op_ids.size(), 1u);
        retract_op_id = wb->op_ids[0];
        EXPECT_EQ(scheduler_->RetractedSize(), 1u);
    }
    SendWriteBackDone(retract_op_id);

    // The freed pages let r2 enter canvas 2; r1 stays retracted for now.
    {
        auto plan = PlanOnce();
        auto rows = DiffusionRows(*GetForwardOp(plan));
        ASSERT_EQ(rows.size(), 1u);
        EXPECT_EQ(rows[0].id, "r2");
        EXPECT_EQ(rows[0].steps_taken, 0);
        EXPECT_EQ(rows[0].committed_len, 12);
        EXPECT_EQ(scheduler_->RetractedSize(), 1u);
    }
    SendDenoiseResult("r2", true);
    PlanOnce();  // r2 commit
    SendForwardDone("r2", MakeTokens(8, 450));
    SendFinish("r2");

    // r2's finish drains to host; ack its writeback so its device pages can
    // be evicted for r1's recovery.
    {
        auto plan = PlanOnce();
        const auto* wb = GetWriteBack(plan);
        ASSERT_NE(wb, nullptr);
        ASSERT_EQ(wb->op_ids.size(), 1u);
        SendWriteBackDone(wb->op_ids[0]);
    }

    // r1 resumes: committed KV loads back, the canvas restarts at step 0 with
    // a fresh reservation; the discarded canvas progress is regenerated.
    std::int64_t resume_epoch = 0;
    {
        auto plan = PlanOnce();
        auto rows = DiffusionRows(*GetForwardOp(plan));
        ASSERT_EQ(rows.size(), 1u);
        EXPECT_EQ(rows[0].id, "r1");
        EXPECT_EQ(rows[0].kind, DiffusionKind::kDenoise);
        EXPECT_EQ(rows[0].steps_taken, 0);     // resume from step 0
        EXPECT_EQ(rows[0].committed_len, 14);  // committed history survived
        EXPECT_EQ(rows[0].canvas_len, 8);
        EXPECT_EQ(rows[0].begin, 0);  // fresh slot: full page table republished
        EXPECT_EQ(rows[0].size, static_cast<std::int32_t>(rows[0].occupied_pages.size()));
        EXPECT_GT(rows[0].pass_epoch, pre_retract_epoch);  // restart = fresh pass identity
        resume_epoch = rows[0].pass_epoch;
        ASSERT_NE(GetLoadBack(plan), nullptr);  // host → device recovery copies
        EXPECT_EQ(scheduler_->RetractedSize(), 0u);
    }

    // A late DenoiseResult for the discarded canvas must be ignored: it
    // echoes the pre-retract epoch, so the FSM drops it and the request keeps
    // waiting for the restarted pass (next plan schedules nothing for r1).
    SendDenoiseResult("r1", true, pre_retract_epoch);
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r1"), 14);
    {
        auto plan = PlanOnce();
        const auto* fwd = GetForwardOp(plan);
        ASSERT_NE(fwd, nullptr);
        EXPECT_TRUE(fwd->request_ids.empty());
    }

    // The fresh pass's result (echoing the restarted epoch) is accepted.
    SendDenoiseResult("r1", true, resume_epoch);
    {
        auto row = PlanSingleDiffusionRow("r1");
        EXPECT_EQ(row.kind, DiffusionKind::kCommit);
        EXPECT_EQ(row.steps_taken, 1);
    }
    SendForwardDone("r1", MakeTokens(8, 200));
    SendFinish("r1");
    {
        auto plan = PlanOnce();
        const auto* wb = GetWriteBack(plan);
        ASSERT_NE(wb, nullptr);
        ASSERT_EQ(wb->op_ids.size(), 1u);
        SendWriteBackDone(wb->op_ids[0]);
    }
    PlanOnce();
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r1"), -1);
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r2"), -1);
    EXPECT_EQ(scheduler_->ActiveKvPages(), 0u);
}

// A denoise result that lands while the request is retracting is stale
// (canvas progress was discarded) and must not corrupt the FSM.
TEST_F(BlockDiffusionBackpressureTestSuite, Retraction_StaleDenoiseResultIsDropped) {
    Submit(MakeDiffusionSpec("r1", 3, 8, 4, 16, /*start=*/1));
    Submit(MakeDiffusionSpec("r2", 2, 8, 4, 16, /*start=*/300));
    PlanOnce();
    PlanOnce();
    SendDenoiseResult("r1", true);
    SendDenoiseResult("r2", true);
    PlanOnce();
    SendForwardDone("r1", MakeTokens(8, 100));
    SendForwardDone("r2", MakeTokens(8, 400));

    auto plan = PlanOnce();  // retracts r1
    const auto* wb = GetWriteBack(plan);
    ASSERT_NE(wb, nullptr);
    EXPECT_EQ(scheduler_->RetractedSize(), 1u);

    // Stale results during Retracting and Retracted are dropped silently.
    SendDenoiseResult("r1", true);
    EXPECT_EQ(scheduler_->RetractedSize(), 1u);
    SendWriteBackDone(wb->op_ids[0]);
    SendDenoiseResult("r1", false);
    EXPECT_EQ(scheduler_->RetractedSize(), 1u);
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r1"), 14);
}

// Retract a request whose denoise pass is still IN FLIGHT, resume it, then
// deliver the discarded pass's result. The pre-retract result echoes a stale
// pass_epoch and must be dropped; only the restarted pass's result advances
// the canvas.
TEST_F(BlockDiffusionBackpressureTestSuite, Retraction_MidPass_StaleEpochResultDroppedAfterResume) {
    Submit(MakeDiffusionSpec("r1", 3, /*canvas_length=*/8, /*max_denoising_steps=*/4, /*max_new_tokens=*/16,
                             /*start=*/1));
    Submit(MakeDiffusionSpec("r2", 1, /*canvas_length=*/4, /*max_denoising_steps=*/4, /*max_new_tokens=*/16,
                             /*start=*/300));
    PlanOnce();  // prefills
    {
        auto plan = PlanOnce();  // both enter canvas 1
        EXPECT_EQ(DiffusionRows(*GetForwardOp(plan)).size(), 2u);
    }
    SendDenoiseResult("r1", true);
    SendDenoiseResult("r2", true);
    PlanOnce();  // both commit
    SendForwardDone("r1", MakeTokens(8, 100));  // r1 committed = 14
    SendForwardDone("r2", MakeTokens(4, 400));  // r2 committed = 6

    // r1 enters canvas 2 (its reservation still fits); r2's entry starves.
    std::int64_t in_flight_epoch = 0;
    {
        auto plan = PlanOnce();
        auto rows = DiffusionRows(*GetForwardOp(plan));
        ASSERT_EQ(rows.size(), 1u);
        EXPECT_EQ(rows[0].id, "r1");
        EXPECT_EQ(rows[0].kind, DiffusionKind::kDenoise);
        EXPECT_EQ(rows[0].steps_taken, 0);
        in_flight_epoch = rows[0].pass_epoch;
    }

    // r2 still cannot enter its canvas → KV pressure. r1 (the longest) is
    // retracted while its canvas-2 denoise pass is STILL IN FLIGHT.
    cache_op_id retract_op_id{};
    {
        auto plan = PlanOnce();
        const auto* fwd = GetForwardOp(plan);
        ASSERT_NE(fwd, nullptr);
        EXPECT_TRUE(fwd->request_ids.empty());
        const auto* wb = GetWriteBack(plan);
        ASSERT_NE(wb, nullptr);
        ASSERT_EQ(wb->op_ids.size(), 1u);
        retract_op_id = wb->op_ids[0];
        EXPECT_EQ(scheduler_->RetractedSize(), 1u);
    }
    SendWriteBackDone(retract_op_id);
    SendAbort("r2");  // clear the field so r1 can resume

    // r1 resumes with a fresh pass identity for the restarted canvas.
    std::int64_t resume_epoch = 0;
    {
        auto plan = PlanOnce();
        auto rows = DiffusionRows(*GetForwardOp(plan));
        ASSERT_EQ(rows.size(), 1u);
        EXPECT_EQ(rows[0].id, "r1");
        EXPECT_EQ(rows[0].steps_taken, 0);
        EXPECT_EQ(rows[0].committed_len, 14);
        EXPECT_GT(rows[0].pass_epoch, in_flight_epoch);
        resume_epoch = rows[0].pass_epoch;
        EXPECT_EQ(scheduler_->RetractedSize(), 0u);
    }

    // The OLD pass's result (scheduled pre-retract, discarded canvas) arrives
    // now: stale epoch → dropped; r1 keeps waiting for the restarted pass.
    SendDenoiseResult("r1", true, in_flight_epoch);
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r1"), 14);
    {
        auto plan = PlanOnce();
        const auto* fwd = GetForwardOp(plan);
        ASSERT_NE(fwd, nullptr);
        EXPECT_TRUE(fwd->request_ids.empty());  // still in flight, nothing to schedule
    }

    // The NEW pass's result is accepted and the request completes normally.
    SendDenoiseResult("r1", true, resume_epoch);
    {
        auto row = PlanSingleDiffusionRow("r1");
        EXPECT_EQ(row.kind, DiffusionKind::kCommit);
        EXPECT_EQ(row.steps_taken, 1);
        EXPECT_GT(row.pass_epoch, resume_epoch);
    }
    SendForwardDone("r1", MakeTokens(8, 200));
    SendFinish("r1");
    {
        auto plan = PlanOnce();
        const auto* wb = GetWriteBack(plan);
        ASSERT_NE(wb, nullptr);
        ASSERT_EQ(wb->op_ids.size(), 1u);
        SendWriteBackDone(wb->op_ids[0]);
    }
    PlanOnce();
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r1"), -1);
}

// ------------------------------------------------------------
//  Explicit write span with an unaligned prompt: the canvas starts mid-page,
//  so the commit's write span includes the page shared with committed
//  history while denoise passes still may not write anything.
// ------------------------------------------------------------

TEST_F(BlockDiffusionTestSuite, WriteSpan_UnalignedPrompt_CoversSharedTailPage) {
    RequestSpec spec{.request_id = "r1", .tokens = {1, 2, 3}};  // 3 tokens, page_size 2
    spec.block_diffusion = BlockDiffusionParams{
        .canvas_length = 8,
        .max_denoising_steps = 4,
        .max_new_tokens = 8,
    };
    Submit(spec);
    PlanOnce();  // prefill
    {
        auto row = PlanSingleDiffusionRow("r1");
        EXPECT_EQ(row.kind, DiffusionKind::kDenoise);
        EXPECT_EQ(row.committed_len, 3);
        // Canvas covers positions [3, 11) → pages 1..5 of the 6 occupied.
        EXPECT_EQ(row.occupied_pages.size(), 6u);
        EXPECT_EQ(row.write_page_begin, 1);
        EXPECT_EQ(row.write_page_count, 0);  // dark during denoise
    }
    SendDenoiseResult("r1", true);
    {
        auto row = PlanSingleDiffusionRow("r1");
        EXPECT_EQ(row.kind, DiffusionKind::kCommit);
        EXPECT_EQ(row.write_page_begin, 1);  // includes the shared tail page
        EXPECT_EQ(row.write_page_count, 5);
    }
    SendForwardDone("r1", MakeTokens(8, 100));
    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->GetRequestTokenSize("r1"), -1);
    EXPECT_EQ(scheduler_->ActiveKvPages(), 0u);
}

// ------------------------------------------------------------
//  Submit-time validation of BlockDiffusionParams.
// ------------------------------------------------------------

TEST_F(BlockDiffusionTestSuite, Submit_RejectsInvalidParams) {
    // canvas_length not a multiple of page_size.
    EXPECT_THROW(Submit(MakeDiffusionSpec("bad1", 2, /*canvas_length=*/7, 4, 16)), std::invalid_argument);
    // canvas_length exceeding max_scheduled_tokens could never be scheduled.
    EXPECT_THROW(Submit(MakeDiffusionSpec("bad2", 2, /*canvas_length=*/128, 4, 256)), std::invalid_argument);
    // Non-positive parameters.
    EXPECT_THROW(Submit(MakeDiffusionSpec("bad3", 2, 8, /*max_denoising_steps=*/0, 16)), std::invalid_argument);
    EXPECT_THROW(Submit(MakeDiffusionSpec("bad4", 2, 8, 4, /*max_new_tokens=*/0)), std::invalid_argument);
}

}  // namespace tokenspeed::test
