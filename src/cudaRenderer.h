#pragma once
#include "SPH_sim.h"

struct cudaGraphicsResource;

class CudaRenderer : public SPH_sim {
private:
    int numParticles;
    float* positions;       // n * float4
    float* velocities;      // n * float4
    float* densities;       // n * float
    cudaGraphicsResource* vboResource;

public:
    CudaRenderer(int numParticles, unsigned int seed);
    ~CudaRenderer();

    int  getNumParticles() const override { return numParticles; }
    void setup(unsigned int vbo) override;
    void step(float dt) override;
    void writeToVBO() override;
};