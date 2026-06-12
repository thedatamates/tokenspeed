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

#include <span>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tokenspeed {

// Block-diffusion generation mode (DiffusionGemma-style). Present on a
// RequestSpec, the request denoises fixed-length canvases instead of decoding
// token-by-token: prefill → repeat per canvas {N denoise passes, 1 commit
// pass} → finish. Absent ⇒ plain autoregressive request (unchanged).
struct BlockDiffusionParams {
    // Tokens per canvas (CL). Must be a positive multiple of page_size and
    // fit within max_scheduled_tokens.
    std::int32_t canvas_length{};
    // Scheduler-enforced backstop: a canvas commits after at most this many
    // denoise passes even if the executor never reports convergence.
    std::int32_t max_denoising_steps{};
    // Total generation budget; canvases = ceil(max_new_tokens / canvas_length),
    // the final canvas is truncated to the remaining budget.
    std::int32_t max_new_tokens{};
};

struct RequestSpec {
    std::string request_id;
    std::vector<std::int32_t> tokens;
    std::vector<std::string> rolling_hashes;
    std::int32_t storage_hit_pages{0};
    // Engaged ⇒ block-diffusion request; absent ⇒ autoregressive.
    std::optional<BlockDiffusionParams> block_diffusion{};
};

struct PrefillInfo {
    std::span<const std::int32_t> input_ids;
    std::vector<std::int32_t> shifted_input_ids;
    std::int32_t already_scheduled_len;
    std::int32_t extend_len;
};

struct StorageInfo {
    std::vector<std::string> rolling_hashes;
    std::int32_t hit_pages{0};
};

}  // namespace tokenspeed
