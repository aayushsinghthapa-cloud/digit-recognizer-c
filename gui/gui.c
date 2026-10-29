/* gui.c - draw a digit with the mouse and see the network's live prediction.
 * Build with `make gui`, run from the project root after `./digits train-nn`. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"

#include "mnist.h"
#include "nn.h"
#include "preprocess.h"

#define CANVAS      280          /* drawing resolution: 10x MNIST */
#define VIEW_SCALE  2            /* canvas shown at 560x560 on screen */
#define MARGIN      20
#define PANEL_X     (MARGIN * 2 + CANVAS * VIEW_SCALE)
#define WIN_W       (PANEL_X + 320)
#define WIN_H       (CANVAS * VIEW_SCALE + MARGIN * 2)
#define BRUSH       11.0f        /* brush radius in canvas pixels (MNIST strokes are thick) */
#define PREVIEW     4            /* 28x28 preview drawn at 4x */

/* Paint a soft round dot: full ink inside radius*0.6, fading to 0 at the radius.
 * Ink only ever increases (max), so overlapping dots don't get darker than 1. */
static void paint_dot(float *canvas, float cx, float cy)
{
    int x0 = (int)(cx - BRUSH), x1 = (int)(cx + BRUSH);
    int y0 = (int)(cy - BRUSH), y1 = (int)(cy + BRUSH);
    for (int y = y0; y <= y1; y++)
        for (int x = x0; x <= x1; x++) {
            if (x < 0 || y < 0 || x >= CANVAS || y >= CANVAS)
                continue;
            float d = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
            float v = d < BRUSH * 0.6f ? 1.0f : 1.0f - (d - BRUSH * 0.6f) / (BRUSH * 0.4f);
            if (v > canvas[y * CANVAS + x])
                canvas[y * CANVAS + x] = v;
        }
}

/* Paint dots along the segment from (ax, ay) to (bx, by) so fast mouse
 * movements still leave a continuous stroke. */
static void paint_line(float *canvas, float ax, float ay, float bx, float by)
{
    float len = sqrtf((bx - ax) * (bx - ax) + (by - ay) * (by - ay));
    int steps = (int)(len / 2.0f) + 1;
    for (int i = 0; i <= steps; i++) {
        float t = (float)i / steps;
        paint_dot(canvas, ax + (bx - ax) * t, ay + (by - ay) * t);
    }
}

int main(void)
{
    NNModel *model = malloc(sizeof(*model));
    float *canvas = calloc(CANVAS * CANVAS, sizeof(float));
    Color *pixels = malloc(CANVAS * CANVAS * sizeof(Color));
    if (!model || !canvas || !pixels) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    if (nn_load(model, "models/nn.bin")) {
        free(model);
        free(canvas);
        free(pixels);
        return 1;
    }

    SetTraceLogLevel(LOG_WARNING);
    InitWindow(WIN_W, WIN_H, "Digit Recognizer - draw a digit");
    SetTargetFPS(60);

    Image blank = GenImageColor(CANVAS, CANVAS, BLACK);
    Texture2D tex = LoadTextureFromImage(blank);
    UnloadImage(blank);

    float input[MNIST_PIXELS] = {0};
    NNActivations act;
    int has_ink = 0, dirty = 1;
    Vector2 last = {0};
    int drawing = 0;

    while (!WindowShouldClose()) {
        /* --- input --- */
        Vector2 mouse = GetMousePosition();
        float cx = (mouse.x - MARGIN) / VIEW_SCALE, cy = (mouse.y - MARGIN) / VIEW_SCALE;
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            if (!drawing)
                last = (Vector2){cx, cy};
            paint_line(canvas, last.x, last.y, cx, cy);
            last = (Vector2){cx, cy};
            drawing = 1;
            dirty = 1;
        } else {
            drawing = 0;
        }
        if (IsKeyPressed(KEY_C) || IsKeyPressed(KEY_SPACE) ||
            IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) {
            memset(canvas, 0, CANVAS * CANVAS * sizeof(float));
            dirty = 1;
        }

        /* --- predict only when the drawing changed --- */
        if (dirty) {
            has_ink = preprocess_canvas(canvas, CANVAS, input) == 0;
            if (has_ink)
                nn_forward(model, input, &act);
            for (int i = 0; i < CANVAS * CANVAS; i++) {
                unsigned char g = (unsigned char)(canvas[i] * 255.0f);
                pixels[i] = (Color){g, g, g, 255};
            }
            UpdateTexture(tex, pixels);
            dirty = 0;
        }

        /* --- draw --- */
        BeginDrawing();
        ClearBackground((Color){24, 24, 28, 255});

        DrawTextureEx(tex, (Vector2){MARGIN, MARGIN}, 0.0f, VIEW_SCALE, WHITE);
        DrawRectangleLines(MARGIN - 1, MARGIN - 1, CANVAS * VIEW_SCALE + 2,
                           CANVAS * VIEW_SCALE + 2, GRAY);

        int y = MARGIN;
        if (has_ink) {
            int best = 0;
            for (int k = 1; k < NUM_CLASSES; k++)
                if (act.p[k] > act.p[best])
                    best = k;
            DrawText(TextFormat("%d", best), PANEL_X, y - 6, 96, RAYWHITE);
            DrawText(TextFormat("%.1f%% sure", 100.0f * act.p[best]), PANEL_X + 80, y + 50, 20, LIGHTGRAY);

            y += 110;
            for (int k = 0; k < NUM_CLASSES; k++) {
                Color c = k == best ? (Color){80, 200, 120, 255} : (Color){90, 110, 150, 255};
                DrawText(TextFormat("%d", k), PANEL_X, y + k * 26, 20, LIGHTGRAY);
                DrawRectangle(PANEL_X + 24, y + k * 26 + 2, (int)(200 * act.p[k]), 18, c);
                DrawText(TextFormat("%5.1f%%", 100.0f * act.p[k]), PANEL_X + 232, y + k * 26, 18, GRAY);
            }
        } else {
            DrawText("Draw a digit", PANEL_X, y + 20, 30, RAYWHITE);
            DrawText("left mouse: draw", PANEL_X, y + 70, 18, GRAY);
            DrawText("right mouse / C / space: clear", PANEL_X, y + 95, 18, GRAY);
            y += 110;
        }

        /* What the network actually sees: the 28x28 preprocessed input. */
        int py = MARGIN + CANVAS * VIEW_SCALE - MNIST_ROWS * PREVIEW;
        DrawText("network input (28x28):", PANEL_X, py - 24, 18, GRAY);
        for (int r = 0; r < MNIST_ROWS; r++)
            for (int c = 0; c < MNIST_COLS; c++) {
                unsigned char g = has_ink ? (unsigned char)(input[r * MNIST_COLS + c] * 255.0f) : 0;
                DrawRectangle(PANEL_X + c * PREVIEW, py + r * PREVIEW, PREVIEW, PREVIEW,
                              (Color){g, g, g, 255});
            }
        DrawRectangleLines(PANEL_X - 1, py - 1, MNIST_COLS * PREVIEW + 2, MNIST_ROWS * PREVIEW + 2, GRAY);

        EndDrawing();
    }

    UnloadTexture(tex);
    CloseWindow();
    free(model);
    free(canvas);
    free(pixels);
    return 0;
}
