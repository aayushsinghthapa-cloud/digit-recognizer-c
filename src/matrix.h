/* matrix.h - hand-written vector and matrix math.
 * Matrices are stored row-major: element (r, c) of a rows x cols matrix is m[r * cols + c]. */
#ifndef MATRIX_H
#define MATRIX_H

/* Dot product: a . b = sum_i a[i] * b[i]. */
float vec_dot(const float *a, const float *b, int n);

/* Squared Euclidean distance: ||a - b||^2 = sum_i (a[i] - b[i])^2. */
float vec_sq_dist(const float *a, const float *b, int n);

/* Matrix-vector product plus bias: y = W x + b, W is rows x cols. */
void mat_vec_mul_add(const float *W, const float *x, const float *b,
                     float *y, int rows, int cols);

/* Transposed matrix-vector product: y = W^T v, W is rows x cols, y has cols entries. */
void mat_t_vec_mul(const float *W, const float *v, float *y, int rows, int cols);

/* Scaled add: a += scale * b. */
void vec_add_scaled(float *a, const float *b, float scale, int n);

/* Add an outer product: W += scale * (u v^T), i.e. W[r][c] += scale * u[r] * v[c]. */
void mat_add_outer(float *W, const float *u, const float *v,
                   float scale, int rows, int cols);

/* In-place softmax: z[i] = e^(z[i] - max) / sum_j e^(z[j] - max). */
void vec_softmax(float *z, int n);

/* Index of the largest element. */
int vec_argmax(const float *a, int n);

#endif
