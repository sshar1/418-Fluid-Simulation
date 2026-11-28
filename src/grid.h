#pragma once
#include "hostDevice.h"
#include "simParams.h"

struct Grid {
    float boundsW, boundsH;
    int cols, rows;

    explicit Grid(float boxScale)
        : boundsW(kBoundsW * boxScale), boundsH(kBoundsH * boxScale),
          cols((int)(boundsW / kSmoothingRadius) + 1),
          rows((int)(boundsH / kSmoothingRadius) + 1) {}

    int numCells() const { return cols * rows; }

    HOST_DEVICE static int clampCell(int c, int count) { return c < 0 ? 0 : c < count ? c : count - 1; }
    HOST_DEVICE int cellX(float x) const { return clampCell((int)((x + boundsW / 2) / kSmoothingRadius), cols); }
    HOST_DEVICE int cellY(float y) const { return clampCell((int)((y + boundsH / 2) / kSmoothingRadius), rows); }
    HOST_DEVICE int cellOf(float x, float y) const { return cellY(y) * cols + cellX(x); }

    template <class F>
    HOST_DEVICE void forEachNeighbor(float x, float y, const int* cellStart, F f) const {
        int cx = cellX(x), cy = cellY(y);
        int x0 = clampCell(cx - 1, cols), x1 = clampCell(cx + 1, cols);
        int y0 = clampCell(cy - 1, rows), y1 = clampCell(cy + 1, rows);
        for (int row = y0; row <= y1; row++) {
            int end = cellStart[row * cols + x1 + 1];
            for (int j = cellStart[row * cols + x0]; j < end; j++) f(j);
        }
    }
};
