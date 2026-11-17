// TODO NATHAN: ref renderer with openmp
#pragma once
#include <vector>
#include "SPH_sim.h"

class RefRenderer : public SPH_sim {
private:
    int n;
    unsigned int vbo;
    std::vector<float> pos, vel;          // 4 floats per particle
    std::vector<unsigned char> colors;    // 4 bytes (RGBA) per particle, filled in writeToVBO
public:
    RefRenderer(int numParticles, unsigned int seed);
    ~RefRenderer() {}
    int  getNumParticles() const override { return n; }
    void setup(unsigned int vbo) override;
    void step(float dt) override;
    void writeToVBO() override;
};