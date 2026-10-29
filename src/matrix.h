/* matrix.h - hand-written vector and matrix math. */
#ifndef MATRIX_H
#define MATRIX_H

/* Dot product: a . b = sum_i a[i] * b[i]. */
float vec_dot(const float *a, const float *b, int n);

/* Squared Euclidean distance: ||a - b||^2 = sum_i (a[i] - b[i])^2. */
float vec_sq_dist(const float *a, const float *b, int n);

#endif
