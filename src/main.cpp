#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "platformgl.h"
#include "refRenderer.h"
#include "cudaRenderer.h"
#include "stream.h"
#include "benchmark.h"
#include "simParams.h"
#include "fixedStepper.h"

static const int kWindowW = 800;
static const int kWindowH = (int)(kWindowW * kBoundsH / kBoundsW);

static SPH_sim* gSim = nullptr;
static GLuint vbo = 0;
static FixedStepper gStepper;
static int frames = 0;
static int stepsSinceTitle = 0;
static int lastFrameTime = 0;

static void display() {
    int steps = gStepper.stepsThisFrame();
    for (int s = 0; s < steps; s++) gSim->step(kDt);
    if (steps > 0) gSim->writeToVBO();
    stepsSinceTitle += steps;

    const int n = gSim->getNumParticles();
    glClear(GL_COLOR_BUFFER_BIT);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(4, GL_FLOAT, 0, (const void*)0);
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, (const void*)vboColorOffset(n));
    glDrawArrays(GL_POINTS, 0, n);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glutSwapBuffers();

    frames++;
    int now = glutGet(GLUT_ELAPSED_TIME);   // ms
    if (now - lastFrameTime >= 1000) {
        float secs = (now - lastFrameTime) / 1000.f;
        char title[96];
        snprintf(title, sizeof(title), "SPH - %.1f FPS, %.0f steps/s",
                 frames / secs, stepsSinceTitle / secs);
        glutSetWindowTitle(title);
        frames = 0;
        stepsSinceTitle = 0;
        lastFrameTime = now;
    }
}

static void idle() { glutPostRedisplay(); }

static SPH_sim* makeSim(bool useCuda) {
    if (useCuda) return new CudaRenderer(kNumParticles, kSeed);
    return new RefRenderer(kNumParticles, kSeed);
}

int main(int argc, char** argv) {
    if (argc > 4 && strcmp(argv[1], "--bench") == 0)
        return runBenchmark(atoi(argv[2]), atof(argv[3]), atoi(argv[4]));

    bool useCuda = (argc > 1 && strcmp(argv[1], "cuda") == 0);

    if (argc > 2 && strcmp(argv[2], "--stream") == 0)   // headless, see stream.cpp
        return runStream(makeSim(useCuda));

    // Create gl context
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(kWindowW, kWindowH);
    glutCreateWindow("SPH");

    glClearColor(0.f, 0.f, 0.f, 1.f);
    glDisable(GL_DEPTH_TEST);

    // Map the sim box (world units) onto the window
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-kBoundsW / 2, kBoundsW / 2, -kBoundsH / 2, kBoundsH / 2, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Settings that make points look nice. Point size is in pixels.
    glPointSize(2.f * kParticleRadius * (kWindowW / kBoundsW));
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Create sim and vbo
    gSim = makeSim(useCuda);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vboBytes(gSim->getNumParticles()), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    gSim->setup(vbo);
    gSim->writeToVBO();   // so frames drawn before the first step aren't garbage

    glutDisplayFunc(display);
    glutIdleFunc(idle);
    lastFrameTime = glutGet(GLUT_ELAPSED_TIME);
    gStepper.reset();   // don't count GL/CUDA setup time as sim time
    glutMainLoop();
    return 0;
}
