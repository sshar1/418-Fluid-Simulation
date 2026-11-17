#include <cmath>
#include "platformgl.h"
#include "refRenderer.h"
#include "simParams.h"
#include "colormap.h"

RefRenderer::RefRenderer(int numParticles, unsigned int seed)
    : SPH_sim(seed), n(numParticles), vbo(0),
      pos(4 * numParticles), vel(4 * numParticles), colors(4 * numParticles) {}

void RefRenderer::setup(unsigned int v) {
    vbo = v;
    spawnParticles(n, pos.data(), vel.data());
}

void RefRenderer::step(float dt) {
    for (int i = 0; i < n; i++) {
        float* p = &pos[4 * i]; float* v = &vel[4 * i];
        v[1] += kGravity * dt;
        p[0] += v[0] * dt;
        p[1] += v[1] * dt;

        // bounce off the box walls: clamp inside and reflect with damping
        const float hx = kBoundsW / 2 - kParticleRadius, hy = kBoundsH / 2 - kParticleRadius;
        if (std::fabs(p[0]) > hx) { p[0] = std::copysign(hx, p[0]); v[0] *= -kCollisionDamping; }
        if (std::fabs(p[1]) > hy) { p[1] = std::copysign(hy, p[1]); v[1] *= -kCollisionDamping; }
    }
}

void RefRenderer::writeToVBO() {
    for (int i = 0; i < n; i++) {
        RGBA8 c = speedToColor(vel[4 * i], vel[4 * i + 1]);
        unsigned char* out = &colors[4 * i];
        out[0] = c.r; out[1] = c.g; out[2] = c.b; out[3] = c.a;
    }

    // Layout in SPH_sim.h: positions, then colours
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, pos.size() * sizeof(float), pos.data());
    glBufferSubData(GL_ARRAY_BUFFER, vboColorOffset(n), colors.size(), colors.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
