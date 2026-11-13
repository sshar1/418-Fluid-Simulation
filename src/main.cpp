#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "platformgl.h"
#include "refRenderer.h"
#include "cudaRenderer.h"

static const int kWindowSize = 800;
static const float kRadiusNDC = 0.05f;   // keep in sync with RADIUS in the sims
static const int kNumPoints = 1;

static SPH_sim* gSim = nullptr;
static GLuint vbo = 0;
static int frames = 0;
static int lastFrameTime = 0;

static void display() {
    gSim->step();

    glClear(GL_COLOR_BUFFER_BIT);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glDrawArrays(GL_POINTS, 0, gSim->getNumParticles());
    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glutSwapBuffers();

    // FPS counter in the window title
    frames++;
    int now = glutGet(GLUT_ELAPSED_TIME);   // ms
    if (now - lastFrameTime >= 1000) {
        char title[64];
        snprintf(title, sizeof(title), "SPH - %.1f FPS",
                 frames * 1000.f / (now - lastFrameTime));
        glutSetWindowTitle(title);
        frames= 0;
        lastFrameTime = now;
    }
}

static void idle() { glutPostRedisplay(); }   // redraw as soon as the last frame is done

int main(int argc, char** argv) {
    bool useCuda = (argc > 1 && strcmp(argv[1], "cuda") == 0);

    // Create gl context
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(kWindowSize, kWindowSize);
    glutCreateWindow("SPH");

    glClearColor(0.f, 0.f, 0.f, 1.f);
    glDisable(GL_DEPTH_TEST);

    // Settings that make points look nice
    glPointSize(2.f * kRadiusNDC * (kWindowSize / 2.f));
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Create sim and vbo
    gSim = useCuda ? (SPH_sim*)new CudaRenderer(kNumPoints) : (SPH_sim*)new RefRenderer(kNumPoints);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, gSim->getNumParticles() * 4 * sizeof(float),
                 nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    gSim->setup(vbo);

    glutDisplayFunc(display);
    glutIdleFunc(idle);
    lastFrameTime = glutGet(GLUT_ELAPSED_TIME);
    glutMainLoop();
    return 0;
}
