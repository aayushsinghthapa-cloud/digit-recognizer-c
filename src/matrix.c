/* matrix.c - hand-written vector and matrix math. */
#include "matrix.h"

#include <math.h>

/* a . b = sum_i a[i] * b[i] */
float vec_dot(const float *a, const float *b, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

/* ||a - b||^2 = sum_i (a[i] - b[i])^2
 * The square root is skipped: it doesn't change which distance is smallest. */
float vec_sq_dist(const float *a, const float *b, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

/* y[r] = (sum_c W[r][c] * x[c]) + b[r]: each output is one dot product. */
void mat_vec_mul_add(const float *W, const float *x, const float *b,
                     float *y, int rows, int cols)
{
    for (int r = 0; r < rows; r++)
        y[r] = vec_dot(W + r * cols, x, cols) + b[r];
}

/* y[c] = sum_r W[r][c] * v[r]: send each row's signal back to the columns.
 * Loops row by row so memory is read in order (fast). */
void mat_t_vec_mul(const float *W, const float *v, float *y, int rows, int cols)
{
    for (int c = 0; c < cols; c++)
        y[c] = 0.0f;
    for (int r = 0; r < rows; r++)
        vec_add_scaled(y, W + r * cols, v[r], cols);
}

/* a[i] += scale * b[i] */
void vec_add_scaled(float *a, const float *b, float scale, int n)
{
    for (int i = 0; i < n; i++)
        a[i] += scale * b[i];
}

/* W[r][c] += scale * u[r] * v[c] */
void mat_add_outer(float *W, const float *u, const float *v,
                   float scale, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        float s = scale * u[r];
        if (s == 0.0f)
            continue;
        float *row = W + r * cols;
        for (int c = 0; c < cols; c++)
            row[c] += s * v[c];
    }
}

/* softmax(z)_i = e^(z_i) / sum_j e^(z_j)
 * Subtracting max(z) first gives the same result (it cancels top and bottom)
 * but stops e^z from overflowing to infinity for large z. */
void vec_softmax(float *z, int n)
{
    float max = z[vec_argmax(z, n)];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        z[i] = expf(z[i] - max);
        sum += z[i];
    }
    for (int i = 0; i < n; i++)
        z[i] /= sum;
}

/* Return i such that a[i] is the largest value. */
int vec_argmax(const float *a, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (a[i] > a[best])
            best = i;
    return best;
}
