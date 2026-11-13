#pragma once

struct Image;

class SPH_sim {

public:
    virtual ~SPH_sim() { };

    // Advance one time step
    virtual void setup(unsigned int vbo) = 0;

    // Called once after GL context and VBO created.
    // The vbo just holds all the float4 positions
    virtual void step() = 0;

    virtual int getNumParticles() const = 0;
};