#pragma once

// ---- Time ----
constexpr float kDt = 1.f / 240.f;
constexpr int   kMaxStepsPerFrame = 8;
constexpr float kTimeScale = 1.f;

// ---- World ----
constexpr int   kNumParticles   = 1000;
constexpr float kBoundsW        = 16.f;
constexpr float kBoundsH        = 9.f;
constexpr float kParticleRadius = 0.05f;

// ---- Sim Params ----
constexpr float kTargetDensity  = 4.f;
constexpr float kPressureMult   = 50.f;

// ---- Forces ----
constexpr float kGravity          = -9.8f;
constexpr float kCollisionDamping = 0.9f;   // fraction of speed kept after a bounce
constexpr float kSmoothingRadius  = 2.f;   // smoothing radius for density

// ---- Rendering ----
constexpr float kColorMaxSpeed = 4.f;      // speed drawn white

// ---- Initial state ----
constexpr unsigned int kSeed = 418;
constexpr float kSpawnVx = 0.f, kSpawnVy = 0.f;

// ---- Numeric Constants ----
constexpr float kPi = 3.14159265f;
constexpr float kH4 = kSmoothingRadius * kSmoothingRadius * kSmoothingRadius * kSmoothingRadius;
constexpr float kSpikyNorm      = 6.f  / (kPi * kH4);   // W  = (h - d)^2 * kSpikyNorm
constexpr float kSpikyDerivNorm = 12.f / (kPi * kH4);   // W' = (d - h)   * kSpikyDerivNorm