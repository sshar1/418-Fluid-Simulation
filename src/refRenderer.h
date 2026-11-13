// TODO NATHAN: ref renderer with openmp
#pragma once
#include <vector>
#include "SPH_sim.h"

class RefRenderer : public SPH_sim {
private:
    int n;
    unsigned int vbo;
    std::vector<float> pos, vel;   // 4 floats per particle
public:
    RefRenderer(int numParticles = 1);
    ~RefRenderer() {}
    int  getNumParticles() const override { return n; }
    void setup(unsigned int vbo) override;
    void step() override;
};