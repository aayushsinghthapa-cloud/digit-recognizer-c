/* matrix.c - hand-written vector and matrix math. */
#include "matrix.h"

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
