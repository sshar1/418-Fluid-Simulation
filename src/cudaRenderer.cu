#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <math_constants.h>

#include "platformgl.h"          // must come before cuda_gl_interop.h
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "cudaRenderer.h"
#include "simParams.h"
#include "colormap.h"
#include "helper_math.h"

// Wrap CUDA API calls: print the error and exit instead of silently continuing.
#define cudaCheck(call) \
    do { \
        cudaError_t err_ = (call); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err_)); \
            exit(1); \
        } \
    } while (0)

////////////////////////////////////////////////////////////////////////////////////////

CudaRenderer::CudaRenderer(int numParticles, unsigned int seed)
    : SPH_sim(seed), numParticles(numParticles),
      positions(nullptr), velocities(nullptr), vboResource(nullptr) {}

CudaRenderer::~CudaRenderer() {
    if (vboResource) cudaGraphicsUnregisterResource(vboResource);
    cudaFree(positions);
    cudaFree(velocities);
}

void CudaRenderer::setup(unsigned int vbo) {
    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaCheck(cudaGetDeviceCount(&deviceCount));

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("NVIDIA GeForce RTX 2080") == 0)
        {
            isFastGPU = true;
        }

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    if (!isFastGPU)
    {
        printf("WARNING: "
               "You're not running on a fast GPU, please consider using "
               "NVIDIA RTX 2080.\n");
        printf("---------------------------------------------------------\n");
    }

    // Initial state is built on the host by SPH_sim (same code and RNG as the
    // ref sim), then copied to the GPU.
    std::vector<float4> pos(numParticles), vel(numParticles);
    spawnParticles(numParticles, (float*)pos.data(), (float*)vel.data());

    cudaCheck(cudaMalloc(&positions, numParticles * sizeof(float4)));
    cudaCheck(cudaMalloc(&velocities, numParticles * sizeof(float4)));
    cudaCheck(cudaMalloc(&densities, numParticles * sizeof(float)));
    cudaCheck(cudaMemcpy(positions, pos.data(), numParticles * sizeof(float4), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(velocities, vel.data(), numParticles * sizeof(float4), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(densities, 0, numParticles * sizeof(float)));

    cudaCheck(cudaGraphicsGLRegisterBuffer(&vboResource, vbo, cudaGraphicsMapFlagsWriteDiscard));
}

__device__ float densityToPressure(float density) {
    return kPressureMult * (density - kTargetDensity);
}

__device__ float smoothingKernel(float radius, float dst2) {
    // float volume = CUDART_PI_F * std::pow(radius, 8) / 4; // TODO make a constant
    // float value = max(0.f, radius * radius - dst2);
    // return value * value * value / volume;

    float dst = sqrt(dst2);
    if (dst >= radius) return 0;
    // float volume = CUDART_PI_F * std::pow(radius, 4) / 6;
    // return (radius - dst) * (radius - dst) / volume;
    return (radius - dst) * (radius - dst) * kSpikyNorm;
}

__device__ float smoothingKernelDerivative(float radius, float dst) {
    // if (dst >= radius) return 0;

    // float volume = CUDART_PI_F * std::pow(radius, 8) / 4; // TODO make a constant
    // float f = radius * radius - dst * dst;
    // return -6 * dst * f * f / volume;

    if (dst >= radius) return 0;
    // float scale = 12 / (std::pow(radius, 4) * CUDART_PI_F);
    // return (dst - radius) * scale;
    return (dst - radius) * kSpikyDerivNorm;
}

__device__ float calculateDensity(const float4* positions, float4 pos, int n) {
    float density = 0.f;
    for (int particle = 0; particle < n; particle++) {
        float4 d = positions[particle] - pos;
        density += smoothingKernel(kSmoothingRadius, dot(d, d));
    }

    return density;
}

// A = summation {A_j / p_j * ∇W} = summation {∇W}
__device__ float3 calculateDensityGradient(const float4* positions, float4 pos, float* densities, int n) {
    float4 grad = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int particle = 0; particle < n; particle++) {
        float4 d = positions[particle] - pos;

        float dst_squared = dot(d, d);
        if (dst_squared <= __FLT_EPSILON__) continue;

        float dst = sqrt(dst_squared);
        float4 dir = d / dst;

        grad += -dir * smoothingKernelDerivative(kSmoothingRadius, dst);
    }

    float3 res = make_float3(grad.x, grad.y, grad.z);
    return res; 
}

__device__ float calculateSharedPressure(float density1, float density2) {
    float pressure1 = densityToPressure(density1);
    float pressure2 = densityToPressure(density2);
    return (pressure1 + pressure2) / 2;
}

// A = summation {A_j / p_j * ∇W} = summation {∇W}
__device__ float4 calculatePressureForce(const float4* positions, float4 pos, float* densities, int particle, int n) {
    float4 pressure = make_float4(0.f, 0.f, 0.f, 0.f);
    float myDensity = densities[particle];
    for (int i = 0; i< n; i++) {
        float4 d = positions[i] - pos;

        float dst_squared = dot(d, d);
        if (dst_squared <= __FLT_EPSILON__) continue;

        float dst = sqrt(dst_squared);
        float4 dir = d / dst;

        float densityNeighbor = densities[i];
        float sharedPressure = calculateSharedPressure(myDensity, densityNeighbor);

        pressure += sharedPressure * dir * smoothingKernelDerivative(kSmoothingRadius, dst) / densityNeighbor;
    }

    // float3 res = make_float3(pressure.x, pressure.y, pressure.z);
    // return res; 
    return pressure;
}

__global__ void kernelDensities(const float4* pos, float* densities, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    densities[i] = calculateDensity(pos, pos[i], n);
}

__global__ void kernelPressures(float4* pos, float4* vel, float* densities, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float4 p = pos[i];

    // Apply pressure forces
    float4 pressureForce = calculatePressureForce(pos, p, densities, i, n);
    float4 pressureAccel = pressureForce / densities[i];
    vel[i] += pressureAccel * dt;
}

__global__ void kernelStep(float4* pos, float4* vel, float* densities, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float4 p = pos[i];
    float4 v = vel[i];

    v.y += kGravity * dt;
    p.x += v.x * dt;
    p.y += v.y * dt;

    // bounce off the box walls: clamp inside and reflect with damping
    const float hx = kBoundsW / 2 - kParticleRadius, hy = kBoundsH / 2 - kParticleRadius;
    if (fabsf(p.x) > hx) { p.x = copysignf(hx, p.x); v.x *= -kCollisionDamping; }
    if (fabsf(p.y) > hy) { p.y = copysignf(hy, p.y); v.y *= -kCollisionDamping; }

    pos[i] = p;
    vel[i] = v;
}

void CudaRenderer::step(float dt) {
    int threads = 256;
    int blocks  = (numParticles + threads - 1) / threads;
    kernelDensities<<<blocks, threads>>>((float4*)positions, densities, numParticles);
    kernelPressures<<<blocks, threads>>>((float4*)positions, (float4*)velocities, densities, numParticles, dt);
    kernelStep<<<blocks, threads>>>((float4*)positions, (float4*)velocities, densities, numParticles, dt);
    cudaCheck(cudaGetLastError());

    // static int k = 0;
    // if (k++ % 240 == 0) {
    //     float d0;
    //     cudaCheck(cudaMemcpy(&d0, densities + sizeof(float), sizeof(float), cudaMemcpyDeviceToHost));
    //     printf("density: %.4f\n", d0);
    // }
}

// Copy positions into the VBO and color each particle by speed. Once per frame.
__global__ void kernelWriteVBO(const float4* pos, const float4* vel,
                               float4* vboPos, uchar4* vboColor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    vboPos[i] = pos[i];
    float4 v = vel[i];
    RGBA8 c = speedToColor(v.x, v.y);
    vboColor[i] = make_uchar4(c.r, c.g, c.b, c.a);
}

void CudaRenderer::writeToVBO() {
    char* vboPtr;
    size_t bytes;
    cudaCheck(cudaGraphicsMapResources(1, &vboResource));
    cudaCheck(cudaGraphicsResourceGetMappedPointer((void**)&vboPtr, &bytes, vboResource));

    // Layout in SPH_sim.h: positions, then colors. The kernel writes the whole
    // buffer, which the WriteDiscard map flag set in setup() requires.
    int threads = 256;
    int blocks  = (numParticles + threads - 1) / threads;
    kernelWriteVBO<<<blocks, threads>>>((float4*)positions, (float4*)velocities, (float4*)vboPtr,
                                        (uchar4*)(vboPtr + vboColorOffset(numParticles)), numParticles);
    cudaCheck(cudaGetLastError());

    // Unmapping guarantees the kernel (and every one before it) finishes
    // before GL reads the buffer, so no cudaDeviceSynchronize() is needed.
    cudaCheck(cudaGraphicsUnmapResources(1, &vboResource));
}