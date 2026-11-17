#pragma once
#include <chrono>
#include "simParams.h"

class FixedStepper {
    using Clock = std::chrono::steady_clock;
    Clock::time_point prev = Clock::now();
    double accumulator = 0.0;   // seconds of real time not yet simulated

public:
    // Call right before the first frame, so setup time isn't counted.
    void reset() { prev = Clock::now(); accumulator = 0.0; }

    int stepsThisFrame() {
        Clock::time_point now = Clock::now();
        accumulator += std::chrono::duration<double>(now - prev).count() * kTimeScale;
        prev = now;

        int steps = (int)(accumulator / kDt);
        if (steps > kMaxStepsPerFrame) {
            steps = kMaxStepsPerFrame;
            accumulator = 0.0;
        } else {
            accumulator -= steps * kDt;
        }
        return steps;
    }
};
