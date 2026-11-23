#pragma once
#include <stddef.h>
#include <random>
#include "simParams.h"
#include "grid.h"

//   [ n x float4 position (x, y, 0, 1) | n x 4-byte colour (r, g, b, a) ]
inline size_t vboColorOffset(int n) { return (size_t)n * 4 * sizeof(float); }
inline size_t vboBytes(int n)       { return vboColorOffset(n) + (size_t)n * 4; }

class SPH_sim {

public:
    SPH_sim(unsigned int seed, float boxScale) : grid(boxScale), rng(seed) {}
    virtual ~SPH_sim() { };

    virtual void setup(unsigned int vbo) = 0;

    virtual void step(float dt) = 0;

    virtual void writeToVBO() = 0;

    virtual int getNumParticles() const = 0;

protected:
    Grid grid;
    // std::random_device rd;
    std::mt19937 rng;

    void spawnParticles(int n, float* pos, float* vel) {
        // static std::mt19937 gen(rd());

        const float hx = grid.boundsW / 2 - kParticleRadius, hy = grid.boundsH / 2 - kParticleRadius;
        std::uniform_real_distribution<float> x_pos_dist(-hx, hx);
        std::uniform_real_distribution<float> y_pos_dist(-hy, hy);

        for (int i = 0; i < n; i++) {
            float* p = &pos[4 * i]; float* v = &vel[4 * i];

            p[0] = x_pos_dist(rng);
            p[1] = y_pos_dist(rng);
            p[2] = 0.f; p[3] = 1.f;

            v[0] = kSpawnVx; v[1] = kSpawnVy; v[2] = 0.f; v[3] = (float)i;
        }
    }
};
