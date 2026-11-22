#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <chrono>
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

int runBenchmark(int n, int steps) {
    if (n <= 0 || steps <= 0) {
        fprintf(stderr, "usage: ./render --bench N STEPS\n");
        return 1;
    }
    if (!makeOffscreenContext()) {
        fprintf(stderr, "bench: could not create an EGL OpenGL context\n");
        return 1;
    }

    RefRenderer ref(n, kSeed);
    CudaRenderer cuda(n, kSeed);
    std::vector<float> refPos(4 * n), cudaPos(4 * n);
    double refMs  = timeSim(ref, steps, refPos);
    double cudaMs = timeSim(cuda, steps, cudaPos);

    float maxDiff = 0.f;
    for (int i = 0; i < 4 * n; i++) maxDiff = std::max(maxDiff, fabsf(refPos[i] - cudaPos[i]));

    printf("%d particles, %d steps\n", n, steps);
    printf("ref:     %.4f ms/step\n", refMs);
    printf("cuda:    %.4f ms/step\n", cudaMs);
    printf("speedup: %.1fx\n", refMs / cudaMs);
    printf("max position diff: %g\n", maxDiff);
    return 0;
}
