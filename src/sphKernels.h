#pragma once

#include <math.h>
#include "hostDevice.h"
#include "simParams.h"

HOST_DEVICE inline float densityToPressure(float density) {
    return kPressureMult * (density - kTargetDensity);
}

HOST_DEVICE inline float smoothingKernel(float radius, float dst2) {
    float dst = sqrtf(dst2);
    if (dst >= radius) return 0;
    return (radius - dst) * (radius - dst) * kSpikyNorm;
}

HOST_DEVICE inline float smoothingKernelDerivative(float radius, float dst) {
    if (dst >= radius) return 0;
    return (dst - radius) * kSpikyDerivNorm;
}

HOST_DEVICE inline float calculateSharedPressure(float density1, float density2) {
    float pressure1 = densityToPressure(density1);
    float pressure2 = densityToPressure(density2);
    return (pressure1 + pressure2) / 2;
}
