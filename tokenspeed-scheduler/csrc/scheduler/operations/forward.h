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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace tokenspeed {

struct ForwardOperationBase {
    std::string request_id;
    std::int32_t request_pool_index;
    std::int32_t input_length;
    // All pages currently occupied by this request (existing + newly allocated).
    std::vector<int32_t> occupied_pages;
    // Index into occupied_pages where newly allocated pages begin.
    std::int32_t begin;
    // Number of newly allocated pages (starting at occupied_pages[begin]).
    std::int32_t size;

    std::int32_t prefill_length;

    // Per-request model-defined paged cache pages. For sliding-window groups
    // the vector is COMPACT — it contains only live pages (released-from-front
    // entries are absent). Use paged_cache_page_base_offsets to recover
    // absolute logical-page indexing: column c here = absolute logical page
    // base_offset + c. For full-history groups base_offset is implicitly 0
    // and the key may be omitted from paged_cache_page_base_offsets.
    std::map<std::string, std::vector<std::int32_t>> paged_cache_pages;
    // Per-request, per-sliding-group base logical-page offset.
    std::map<std::string, std::int32_t> paged_cache_page_base_offsets;

    // Mamba extension (default: inactive)
    std::int32_t mamba_working_idx{-1};
    std::int32_t mamba_checkpoint_dst_idx{-1};
    std::int32_t mamba_cow_src_idx{-1};
    std::int32_t mamba_branching_seqlen{-1};
};

struct PrefillOperation : public ForwardOperationBase {
    std::vector<std::int32_t> input_ids;
    std::vector<std::int32_t> shifted_input_ids;
    std::int32_t extend_prefix_len;
};

struct DecodeOperation : public ForwardOperationBase {
    std::int32_t decode_input_id = -1;
    // For retraction recover
    std::int32_t hist_token_len = -1;
};

// Block-diffusion row kinds.
enum class DiffusionKind : std::int32_t {
    // Decoder mode (bidirectional, KV read-only) over the executor-resident
    // canvas; the executor must not write the canvas-reservation pages.
    kDenoise = 0,
    // Encoder mode (causal) over the argmax canvas, writing KV into the
    // reservation pages at positions committed_len … committed_len+canvas_len-1.
    kCommit = 1,
};

// Block-diffusion row: one denoise or commit pass over the current canvas.
// input_length == canvas_len (these are real forward rows; they consume
// forward-token budget like prefill chunks but kDenoise allocates nothing).
struct DiffusionOperation : public ForwardOperationBase {
    DiffusionKind kind{DiffusionKind::kDenoise};
    // Tokens in the current canvas (≤ configured canvas_length for the final
    // truncated canvas near max_new_tokens).
    std::int32_t canvas_len{};
    // Tokens already committed (prompt + prior canvases) = canvas start
    // position; drives canvas position ids and attention extents.
    std::int32_t committed_len{};
    // Scheduler's per-canvas step counter. 0 ⇒ the executor must (re)init the
    // canvas (fresh random canvas, zero self-conditioning — also after
    // retraction).
    std::int32_t steps_taken{};
};

using ForwardOperation = std::variant<PrefillOperation, DecodeOperation, DiffusionOperation>;

struct FlatForwardOperation {
    std::vector<std::string> request_ids;
    std::vector<std::int32_t> request_pool_indices;
    std::vector<std::int32_t> input_lengths;
    // Per-request total number of prompt tokens (Request::PrefillSize()).
    std::vector<std::int32_t> prefill_lengths;

    std::vector<std::vector<std::int32_t>> occupied_pages;
    std::vector<std::int32_t> begins;
    std::vector<std::int32_t> sizes;

    std::vector<std::int32_t> input_ids;
    std::vector<std::int32_t> shifted_input_ids;
    std::vector<std::int32_t> extend_prefix_lens;
    std::vector<std::int32_t> decode_input_ids;
    std::vector<std::int32_t> hist_token_lens;

    // Block-diffusion rows (SoA, len = num_diffusion(), indexed after the
    // decode rows: row = num_extends() + #decodes + i).
    std::vector<DiffusionKind> diffusion_kinds;
    std::vector<std::int32_t> diffusion_canvas_lens;
    std::vector<std::int32_t> diffusion_committed_lens;
    std::vector<std::int32_t> diffusion_steps_taken;

    // Mamba extension (SoA)
    std::vector<std::int32_t> mamba_working_indices;
    std::vector<std::int32_t> mamba_checkpoint_dst_indices;
    std::vector<std::int32_t> mamba_cow_src_indices;
    std::vector<std::int32_t> mamba_branching_seqlens;

    // Per-group paged cache block tables: dict[group_id] = [num_reqs,
    // max_live_pages_for_group_in_this_batch] padded with -1. For sliding
    // groups each row is COMPACT (released-from-front pages are absent);
    // pair with paged_cache_block_table_base_offsets to recover absolute
    // logical-page indexing. For full-history groups rows are absolute and
    // the offset is implicitly 0 (key omitted from the offsets map).
    std::map<std::string, std::vector<std::vector<std::int32_t>>> paged_cache_block_tables;
    // Per-group [num_reqs] base logical-page offsets, only present for
    // sliding-window groups. Missing key ⇔ offset is 0 for every row.
    std::map<std::string, std::vector<std::int32_t>> paged_cache_block_table_base_offsets;

    explicit FlatForwardOperation(std::vector<ForwardOperation> ops) {
        // Row partition invariant: [ extends | decodes | diffusion ]. The
        // prefill-first partition is the original AR invariant; the second
        // partition moves diffusion rows after the decodes and is a no-op
        // when no diffusion requests exist (AR plans stay byte-identical).
        std::stable_partition(ops.begin(), ops.end(),
                              [](const ForwardOperation& a) { return std::holds_alternative<PrefillOperation>(a); });
        std::stable_partition(ops.begin(), ops.end(),
                              [](const ForwardOperation& a) { return !std::holds_alternative<DiffusionOperation>(a); });
        for (auto& op : ops) {
            std::visit(
                [this](auto& inner) {
                    request_ids.push_back(std::move(inner.request_id));
                    request_pool_indices.push_back(inner.request_pool_index);
                    input_lengths.push_back(inner.input_length);
                    prefill_lengths.push_back(inner.prefill_length);
                    occupied_pages.push_back(std::move(inner.occupied_pages));
                    begins.push_back(inner.begin);
                    sizes.push_back(inner.size);
                    mamba_working_indices.push_back(inner.mamba_working_idx);
                    mamba_checkpoint_dst_indices.push_back(inner.mamba_checkpoint_dst_idx);
                    mamba_cow_src_indices.push_back(inner.mamba_cow_src_idx);
                    mamba_branching_seqlens.push_back(inner.mamba_branching_seqlen);
                    for (auto& [gid, pages] : inner.paged_cache_pages) {
                        paged_cache_block_tables[gid];
                    }
                    for (auto& [gid, _] : inner.paged_cache_page_base_offsets) {
                        paged_cache_block_table_base_offsets[gid];
                    }
                },
                op);
            if (auto* prefill = std::get_if<PrefillOperation>(&op)) {
                input_ids.insert(input_ids.end(), prefill->input_ids.begin(), prefill->input_ids.end());
                shifted_input_ids.insert(shifted_input_ids.end(), prefill->shifted_input_ids.begin(),
                                         prefill->shifted_input_ids.end());
                extend_prefix_lens.push_back(prefill->extend_prefix_len);
            } else if (auto* decode = std::get_if<DecodeOperation>(&op)) {
                decode_input_ids.push_back(decode->decode_input_id);
                hist_token_lens.push_back(decode->hist_token_len);
            } else if (auto* diffusion = std::get_if<DiffusionOperation>(&op)) {
                diffusion_kinds.push_back(diffusion->kind);
                diffusion_canvas_lens.push_back(diffusion->canvas_len);
                diffusion_committed_lens.push_back(diffusion->committed_len);
                diffusion_steps_taken.push_back(diffusion->steps_taken);
            }
        }
        const std::size_t num_reqs = request_ids.size();
        for (auto& [_, table] : paged_cache_block_tables) {
            table.assign(num_reqs, std::vector<std::int32_t>{});
        }
        for (auto& [_, offsets] : paged_cache_block_table_base_offsets) {
            offsets.assign(num_reqs, 0);
        }
        std::size_t row = 0;
        for (auto& op : ops) {
            std::visit(
                [&](auto& inner) {
                    for (auto& [gid, pages] : inner.paged_cache_pages) {
                        paged_cache_block_tables[gid][row] = std::move(pages);
                    }
                    for (auto& [gid, off] : inner.paged_cache_page_base_offsets) {
                        paged_cache_block_table_base_offsets[gid][row] = off;
                    }
                },
                op);
            ++row;
        }
        padRectangularMinusOne(paged_cache_block_tables);
    }

    bool empty() const { return request_ids.empty(); }
    std::size_t num_extends() const { return extend_prefix_lens.size(); }
    std::size_t num_diffusion() const { return diffusion_kinds.size(); }

private:
    template <typename Key>
    static void padRectangularMinusOne(std::map<Key, std::vector<std::vector<std::int32_t>>>& tables) {
        for (auto& [_, table] : tables) {
            std::int32_t max_cols = 0;
            for (const auto& row : table) {
                max_cols = std::max(max_cols, static_cast<std::int32_t>(row.size()));
            }
            for (auto& row : table) {
                row.resize(max_cols, -1);
            }
        }
    }
};

}  // namespace tokenspeed
