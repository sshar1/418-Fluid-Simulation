#pragma once

#include <math.h>
#include "hostDevice.h"
#include "simParams.h"

struct RGBA8 { unsigned char r, g, b, a; };

HOST_DEVICE inline RGBA8 speedToColor(float vx, float vy) {
    constexpr float kInvMaxSpeedSq = 1.f / (kColorMaxSpeed * kColorMaxSpeed);
    float t = fminf((vx * vx + vy * vy) * kInvMaxSpeedSq, 1.f);   // 0..1
    float u, r0, g0, b0, r1, g1, b1;
    if (t < 0.5f) { u = 2.f * t;       r0 = 30.f; g0 = 80.f;  b0 = 220.f; r1 = 60.f;  g1 = 210.f; b1 = 230.f; }
    else          { u = 2.f * t - 1.f; r0 = 60.f; g0 = 210.f; b0 = 230.f; r1 = 255.f; g1 = 255.f; b1 = 255.f; }

    RGBA8 c;
    c.r = (unsigned char)(r0 + (r1 - r0) * u);
    c.g = (unsigned char)(g0 + (g1 - g0) * u);
    c.b = (unsigned char)(b0 + (b1 - b0) * u);
    c.a = 255;
    return c;
}
