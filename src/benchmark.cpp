#include <stdio.h>
#include <chrono>
#include <cmath>
#include <vector>

#include "platformgl.h"
#include "benchmark.h"
#include "refRenderer.h"
#include "cudaRenderer.h"
#include "stream.h"

static void readPositions(SPH_sim& sim, GLuint vbo, std::vector<float>& pos) {
    sim.writeToVBO();
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, pos.size() * sizeof(float), pos.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

static double timeSim(SPH_sim& sim, int steps, std::vector<float>& pos) {
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vboBytes(sim.getNumParticles()), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    sim.setup(vbo);

    sim.step(kDt);
    readPositions(sim, vbo, pos);

    auto start = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; s++) sim.step(kDt);
    readPositions(sim, vbo, pos);
    std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start;
    return elapsed.count() / steps;
}

int runBenchmark(int n, float boxScale, int steps) {
    if (n <= 0 || boxScale <= 0 || steps <= 0) {
        fprintf(stderr, "usage: ./render --bench N BOXSCALE STEPS\n");
        return 1;
    }
    if (!makeOffscreenContext()) {
        fprintf(stderr, "bench: could not create an EGL OpenGL context\n");
        return 1;
    }

    RefRenderer ref(n, kSeed, boxScale);
    CudaRenderer cuda(n, kSeed, boxScale);
    std::vector<float> refPos(4 * n), cudaPos(4 * n);
    double refMs  = timeSim(ref, steps, refPos);
    double cudaMs = timeSim(cuda, steps, cudaPos);

    float maxDiff = 0.f;
    for (int i = 0; i < 4 * n; i++) {
        float diff = fabsf(refPos[i] - cudaPos[i]);
        if (diff > maxDiff || std::isnan(diff)) maxDiff = diff;
    }

    float w = kBoundsW * boxScale, h = kBoundsH * boxScale;
    printf("n=%d box=%gx%g density=%.1f steps=%d | ref %.3f ms, cuda %.3f ms, speedup %.1fx, max diff %g\n",
           n, w, h, n / (w * h), steps, refMs, cudaMs, refMs / cudaMs, maxDiff);
    return 0;
}
