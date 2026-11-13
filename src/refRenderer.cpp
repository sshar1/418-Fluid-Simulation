#include "platformgl.h"
#include "refRenderer.h"

static const float kDt = 1e-3f, kGravity = -9.8f, kRadius = 0.05f, kRestitution = 0.9f;

RefRenderer::RefRenderer(int numParticles)
    : n(numParticles), vbo(0), pos(4 * numParticles), vel(4 * numParticles) {}

void RefRenderer::setup(unsigned int v) {
    vbo = v;
    for (int i = 0; i < n; i++) {
        float* p = &pos[4 * i]; float* u = &vel[4 * i];
        p[0] = 0.f;  p[1] = 0.5f; p[2] = 0.f; p[3] = 1.f;
        u[0] = 0.6f; u[1] = 0.f;  u[2] = 0.f; u[3] = 0.f;
    }
}

void RefRenderer::step() {
    for (int i = 0; i < n; i++) {
        float* p = &pos[4 * i]; float* v = &vel[4 * i];
        v[1] += kGravity * kDt;
        p[0] += v[0] * kDt;
        p[1] += v[1] * kDt;
        if (p[1] < -1.f + kRadius && v[1] < 0.f) { p[1] = -1.f + kRadius; v[1] = -v[1] * kRestitution; }
        if (p[0] < -1.f + kRadius && v[0] < 0.f) { p[0] = -1.f + kRadius; v[0] = -v[0]; }
        if (p[0] >  1.f - kRadius && v[0] > 0.f) { p[0] =  1.f - kRadius; v[0] = -v[0]; }
    }
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, pos.size() * sizeof(float), pos.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
