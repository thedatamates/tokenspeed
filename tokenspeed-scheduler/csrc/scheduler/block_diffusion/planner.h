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

// Block-diffusion planner policy, extracted from the generic forward planner
// (scheduler/operations/forward.cpp). The Scheduler methods implementing this
// policy live in block_diffusion/planner.cpp; the FSM transitions live in
// block_diffusion/lifecycle.cpp.

#include <cstdint>

namespace tokenspeed {

class Request;

namespace block_diffusion {

// Outcome of one planning attempt for a diffusion request
// (Scheduler::scheduleDiffusionPass).
enum class PlanOutcome : std::int8_t {
    // Nothing to schedule and nothing starved (e.g. the request was finished
    // at the max_new_tokens boundary inside the planner).
    kNoop,
    // A denoise/commit row was produced.
    kScheduled,
    // A pass/commit result is still in flight; the request is waiting on the
    // executor, not on resources (must not trigger the retraction heuristic).
    kWaitingOnExecutor,
    // The request failed to schedule on resource pressure (canvas entry or
    // commit could not fit); counts toward the retraction trigger.
    kStarved,
};

// KV-pressure retraction victim policy: Denoising progress is regenerable;
// Committing is retractable except while its commit pass is in flight
// (required to avoid deadlock at canvas-entry KV pressure).
bool RetractableUnderKvPressure(const Request& request);

}  // namespace block_diffusion
}  // namespace tokenspeed
