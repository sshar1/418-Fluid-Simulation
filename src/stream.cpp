// Headless streaming mode, for watching the sim live when you can only ssh in.
//
//   ssh ghcXX 'cd .../src && ./render cuda --stream' | python3 ../tools/viewer.py
//
// Instead of a GLUT window, this creates an offscreen OpenGL context on the GPU
// with EGL, so the sims (including CUDA-GL interop) run completely unchanged.
// After each frame the VBO is read back and written to stdout as (all in world
// units; the viewer maps the box to its canvas):
//
//   uint32 magic 'SPH3' | uint32 n | float32 boundsW | float32 boundsH
//   | float32 radius | float32 xyzw[4n] | uint8 rgba[4n]
//
// Everything after the header is the VBO byte-for-byte (layout in SPH_sim.h).
//
// All other output (printf in the sims, etc.) is redirected to stderr.

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <chrono>
#include <thread>
#include <vector>

#include <EGL/egl.h>
#include "platformgl.h"
#include "stream.h"
#include "simParams.h"
#include "fixedStepper.h"

static const double kMaxFps = 60.0;       // like vsync; also keeps ssh bandwidth sane
static const unsigned int kMagic = 0x33485053;   // "SPH3"

static bool makeOffscreenContext() {
    unsetenv("DISPLAY");   // with ssh -X, EGL would otherwise try the forwarded display
    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, nullptr, nullptr)) return false;

    const EGLint cfgAttr[] = { EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                               EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE };
    EGLConfig cfg;
    EGLint numCfg = 0;
    if (!eglChooseConfig(dpy, cfgAttr, &cfg, 1, &numCfg) || numCfg < 1) return false;

    const EGLint pbAttr[] = { EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE };
    EGLSurface surf = eglCreatePbufferSurface(dpy, cfg, pbAttr);
    eglBindAPI(EGL_OPENGL_API);
    EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, nullptr);
    if (surf == EGL_NO_SURFACE || ctx == EGL_NO_CONTEXT) return false;
    if (!eglMakeCurrent(dpy, surf, surf, ctx)) return false;

    fprintf(stderr, "stream: EGL %s, OpenGL on %s\n",
            eglQueryString(dpy, EGL_VENDOR), (const char*)glGetString(GL_RENDERER));
    return true;
}

int runStream(SPH_sim* sim) {
    // stdout belongs to the frame stream from here on
    FILE* out = fdopen(dup(STDOUT_FILENO), "wb");
    dup2(STDERR_FILENO, STDOUT_FILENO);
    // stdout now goes to stderr, which over ssh is a pipe, so stdio would buffer
    // printf output in 4 KB blocks. Flush every line so debug prints show up live.
    setvbuf(stdout, nullptr, _IOLBF, 0);

    if (!makeOffscreenContext()) {
        fprintf(stderr, "stream: could not create an EGL OpenGL context (error 0x%x)\n", eglGetError());
        return 1;
    }

    // Same VBO setup as the windowed path in main.cpp
    const int n = sim->getNumParticles();
    const size_t bytes = vboBytes(n);
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, bytes, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    sim->setup(vbo);

    std::vector<unsigned char> frame(bytes);
    const unsigned int hdr[2] = { kMagic, (unsigned int)n };
    const float world[3] = { kBoundsW, kBoundsH, kParticleRadius };
    const auto minFrame = std::chrono::duration<double>(1.0 / kMaxFps);
    FixedStepper stepper;   // same real-time stepping as the GLUT window

    // Once a second, print to stderr where each frame's time went
    using Clock = std::chrono::steady_clock;
    auto secs = [](Clock::duration d) { return std::chrono::duration<double>(d).count(); };
    auto statsStart = Clock::now();
    int statFrames = 0, statSteps = 0;
    double simSecs = 0, sendSecs = 0;

    while (true) {
        auto start = Clock::now();
        int steps = stepper.stepsThisFrame();
        for (int s = 0; s < steps; s++) sim->step(kDt);
        sim->writeToVBO();

        // Kernel launches are async; this readback waits for all of them, so
        // start -> afterSim covers the whole sim + VBO write + readback.
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glGetBufferSubData(GL_ARRAY_BUFFER, 0, bytes, frame.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        auto afterSim = Clock::now();

        // Blocks when ssh/the network can't keep up (the pipe fills)
        fwrite(hdr, sizeof(hdr), 1, out);
        fwrite(world, sizeof(world), 1, out);
        fwrite(frame.data(), 1, frame.size(), out);
        if (fflush(out) != 0) return 0;   // viewer went away
        auto afterSend = Clock::now();

        simSecs += secs(afterSim - start);
        sendSecs += secs(afterSend - afterSim);
        statFrames++;
        statSteps += steps;
        double window = secs(afterSend - statsStart);
        if (window >= 1.0) {
            fprintf(stderr, "stream: %5.1f frames/s, %4.0f steps/s | per frame: sim %6.2f ms, send %6.2f ms, %zu bytes\n",
                    statFrames / window, statSteps / window,
                    1000 * simSecs / statFrames, 1000 * sendSecs / statFrames, sizeof(hdr) + sizeof(world) + bytes);
            statsStart = afterSend;
            statFrames = statSteps = 0;
            simSecs = sendSecs = 0;
        }

        auto elapsed = Clock::now() - start;
        if (elapsed < minFrame) std::this_thread::sleep_for(minFrame - elapsed);
    }
}
