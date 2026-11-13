#pragma once
#include "SPH_sim.h"

struct cudaGraphicsResource;

class CudaRenderer : public SPH_sim {
private:
    int numParticles;
    float* positions;       // n * float4
    float* velocities;      // n * float4
    cudaGraphicsResource* vboResource;

public:
    CudaRenderer(int numParticles = 1);
    ~CudaRenderer();

    int  getNumParticles() const override { return numParticles; }
    void setup(unsigned int vbo) override;
    void step() override;
};