/* preprocess.c - MNIST-style normalization of a drawing. */
#include "preprocess.h"

#include <math.h>
#include <string.h>

#define INK_THRESHOLD 0.05f   /* canvas values above this count as ink */
#define BOX 20                /* MNIST digits fit in a 20x20 box inside 28x28 */

/* Steps, matching how MNIST was built:
 * 1. bounding box: smallest rectangle containing all ink
 * 2. scale: s = 20 / max(width, height), keeping the aspect ratio
 * 3. resample: each output pixel = average of the canvas pixels it covers
 * 4. centre of mass: (cx, cy) = sum(p * (x, y)) / sum(p); shift so it lands at (14, 14) */
int preprocess_canvas(const float *canvas, int size, float out[MNIST_PIXELS])
{
    /* 1. Bounding box of the ink. */
    int x0 = size, y0 = size, x1 = -1, y1 = -1;
    for (int y = 0; y < size; y++)
        for (int x = 0; x < size; x++)
            if (canvas[y * size + x] > INK_THRESHOLD) {
                if (x < x0) x0 = x;
                if (x > x1) x1 = x;
                if (y < y0) y0 = y;
                if (y > y1) y1 = y;
            }
    if (x1 < 0)
        return -1;

    int w = x1 - x0 + 1, h = y1 - y0 + 1;
    float s = (float)BOX / (w > h ? w : h);   /* output pixels per canvas pixel */

    /* 2-3. Scale into a box of ow x oh pixels, placed in the middle of 28x28. */
    int ow = (int)lroundf(w * s), oh = (int)lroundf(h * s);
    if (ow < 1) ow = 1;
    if (oh < 1) oh = 1;
    int ox = (MNIST_COLS - ow) / 2, oy = (MNIST_ROWS - oh) / 2;

    float scaled[MNIST_PIXELS] = {0};
    for (int r = 0; r < oh; r++) {
        /* Canvas rows covered by output row r: [r/s, (r+1)/s), at least one row. */
        int sy0 = y0 + (int)(r / s), sy1 = y0 + (int)((r + 1) / s);
        if (sy1 <= sy0) sy1 = sy0 + 1;
        if (sy1 > y1 + 1) sy1 = y1 + 1;
        for (int c = 0; c < ow; c++) {
            int sx0 = x0 + (int)(c / s), sx1 = x0 + (int)((c + 1) / s);
            if (sx1 <= sx0) sx1 = sx0 + 1;
            if (sx1 > x1 + 1) sx1 = x1 + 1;

            float sum = 0.0f;
            for (int y = sy0; y < sy1; y++)
                for (int x = sx0; x < sx1; x++)
                    sum += canvas[y * size + x];
            int n = (sy1 - sy0) * (sx1 - sx0);
            scaled[(oy + r) * MNIST_COLS + ox + c] = n > 0 ? sum / n : 0.0f;
        }
    }

    /* 4. Centre of mass, then shift it to the image centre. */
    float total = 0.0f, mx = 0.0f, my = 0.0f;
    for (int r = 0; r < MNIST_ROWS; r++)
        for (int c = 0; c < MNIST_COLS; c++) {
            float v = scaled[r * MNIST_COLS + c];
            total += v;
            mx += v * c;
            my += v * r;
        }
    int dx = (int)lroundf(MNIST_COLS / 2.0f - mx / total);
    int dy = (int)lroundf(MNIST_ROWS / 2.0f - my / total);

    memset(out, 0, sizeof(float) * MNIST_PIXELS);
    for (int r = 0; r < MNIST_ROWS; r++)
        for (int c = 0; c < MNIST_COLS; c++) {
            int nr = r + dy, nc = c + dx;
            if (nr >= 0 && nr < MNIST_ROWS && nc >= 0 && nc < MNIST_COLS) {
                float v = scaled[r * MNIST_COLS + c];
                out[nr * MNIST_COLS + nc] = v > 1.0f ? 1.0f : v;
            }
        }
    return 0;
}
