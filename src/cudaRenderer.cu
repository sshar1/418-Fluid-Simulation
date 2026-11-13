#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "platformgl.h"          // must come before cuda_gl_interop.h
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "cudaRenderer.h"

#define SIM_DT      1e-3f
#define GRAVITY    -9.8f
#define RADIUS      0.05f
#define RESTITUTION 0.9f

////////////////////////////////////////////////////////////////////////////////////////

CudaRenderer::CudaRenderer(int numParticles)
    : numParticles(numParticles), positions(nullptr), velocities(nullptr), vboResource(nullptr) {}

CudaRenderer::~CudaRenderer() {
    if (vboResource) cudaGraphicsUnregisterResource(vboResource);
    cudaFree(positions);
    cudaFree(velocities);
}

void CudaRenderer::setup(unsigned int vbo) {
    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

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

    float4* pos = new float4[numParticles];
    float4* vel = new float4[numParticles];
    for (int i = 0; i < numParticles; i++) {
        pos[i] = make_float4(0.f, 0.5f, 0.f, 1.f);
        vel[i] = make_float4(0.2f, 0.0f, 0.f, 0.f);
    }

    cudaMalloc(&positions, numParticles * sizeof(float4));
    cudaMalloc(&velocities, numParticles * sizeof(float4));
    cudaMemcpy(positions, pos, numParticles * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(velocities, vel, numParticles * sizeof(float4), cudaMemcpyHostToDevice);

    delete[] pos;
    delete[] vel;

    cudaGraphicsGLRegisterBuffer(&vboResource, vbo, cudaGraphicsMapFlagsWriteDiscard);
}

__global__ void kernelStep(float4* pos, float4* vel, float4* vboOut, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float4 p = pos[i];
    float4 v = vel[i];

    v.y += GRAVITY * dt;
    p.x += v.x * dt;
    p.y += v.y * dt;

    // bounce off the window edges
    if (p.y < -1.f + RADIUS && v.y < 0.f) { p.y = -1.f + RADIUS; v.y = -v.y * RESTITUTION; }
    if (p.x < -1.f + RADIUS && v.x < 0.f) { p.x = -1.f + RADIUS; v.x = -v.x; }
    if (p.x >  1.f - RADIUS && v.x > 0.f) { p.x =  1.f - RADIUS; v.x = -v.x; }

    pos[i] = p;
    vel[i] = v;
    vboOut[i] = make_float4(p.x, p.y, p.z, 1.f);
}

void CudaRenderer::step() {
    float4* vboPtr;
    size_t bytes;
    cudaGraphicsMapResources(1, &vboResource);
    cudaGraphicsResourceGetMappedPointer((void**)&vboPtr, &bytes, vboResource);

    int threads = 256;
    int blocks  = (numParticles + threads - 1) / threads;
    kernelStep<<<blocks, threads>>>((float4*)positions, (float4*)velocities, vboPtr, numParticles, SIM_DT);
    cudaGetLastError();

    // Unmapping guarantees the kernel finishes before GL reads the buffer,
    // so no cudaDeviceSynchronize() is needed.
    cudaGraphicsUnmapResources(1, &vboResource);
}