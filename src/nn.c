/* nn.c - two-layer neural network with hand-written backpropagation. */
#include "nn.h"
#include "matrix.h"
#include "rng.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NN_MAGIC "DGNN"   /* first 4 bytes of a weights file */

/* Fill w with normal(0, 1) * sqrt(2 / fan_in).
 * He initialization: keeps signal size steady through ReLU layers.
 * Random (not zero) so each hidden neuron starts different and learns
 * a different feature. */
static void he_init(float *w, int n, int fan_in)
{
    float scale = sqrtf(2.0f / fan_in);
    for (int i = 0; i < n; i++)
        w[i] = rng_normal() * scale;
}

void nn_init(NNModel *m)
{
    memset(m, 0, sizeof(*m));
    he_init(m->W1, NN_HIDDEN * MNIST_PIXELS, MNIST_PIXELS);
    he_init(m->W2, NUM_CLASSES * NN_HIDDEN, NN_HIDDEN);
}

/* h_pre = W1 x + b1
 * h     = ReLU(h_pre) = max(0, h_pre)
 * p     = softmax(W2 h + b2) */
int nn_forward(const NNModel *m, const float *x, NNActivations *a)
{
    mat_vec_mul_add(m->W1, x, m->b1, a->h_pre, NN_HIDDEN, MNIST_PIXELS);
    for (int j = 0; j < NN_HIDDEN; j++)
        a->h[j] = a->h_pre[j] > 0.0f ? a->h_pre[j] : 0.0f;
    mat_vec_mul_add(m->W2, a->h, m->b2, a->p, NUM_CLASSES, NN_HIDDEN);
    vec_softmax(a->p, NUM_CLASSES);
    return vec_argmax(a->p, NUM_CLASSES);
}

/* Backpropagation for one example, adding its gradients into g.
 * Chain rule, from the output back to the input:
 *   dz     = p - onehot(y)                 (softmax + cross-entropy)
 *   dW2   += dz h^T,        db2 += dz
 *   dh     = W2^T dz                       (blame passed back to hidden layer)
 *   dh_pre = dh * [h_pre > 0]              (ReLU passes gradient only where active)
 *   dW1   += dh_pre x^T,    db1 += dh_pre */
static void nn_backward(const NNModel *m, const float *x, int y,
                        const NNActivations *a, NNModel *g)
{
    float dz[NUM_CLASSES];
    float dh[NN_HIDDEN];

    for (int k = 0; k < NUM_CLASSES; k++)
        dz[k] = a->p[k] - (k == y ? 1.0f : 0.0f);

    mat_add_outer(g->W2, dz, a->h, 1.0f, NUM_CLASSES, NN_HIDDEN);
    vec_add_scaled(g->b2, dz, 1.0f, NUM_CLASSES);

    mat_t_vec_mul(m->W2, dz, dh, NUM_CLASSES, NN_HIDDEN);
    for (int j = 0; j < NN_HIDDEN; j++)
        if (a->h_pre[j] <= 0.0f)
            dh[j] = 0.0f;

    mat_add_outer(g->W1, dh, x, 1.0f, NN_HIDDEN, MNIST_PIXELS);
    vec_add_scaled(g->b1, dh, 1.0f, NN_HIDDEN);
}

/* theta -= step * gradient, for every parameter array. */
static void nn_update(NNModel *m, const NNModel *g, float step)
{
    vec_add_scaled(m->W1, g->W1, -step, NN_HIDDEN * MNIST_PIXELS);
    vec_add_scaled(m->b1, g->b1, -step, NN_HIDDEN);
    vec_add_scaled(m->W2, g->W2, -step, NUM_CLASSES * NN_HIDDEN);
    vec_add_scaled(m->b2, g->b2, -step, NUM_CLASSES);
}

/* Each epoch: shuffle, then for every mini-batch run forward + backward on
 * each image, average the gradients, and take one step downhill. */
void nn_train(NNModel *m, const Dataset *train, const Dataset *test,
              int epochs, int batch_size, float lr)
{
    int *order = malloc(train->count * sizeof(int));
    NNModel *g = malloc(sizeof(*g));
    if (!order || !g) {
        fprintf(stderr, "out of memory\n");
        free(order);
        free(g);
        return;
    }
    for (int i = 0; i < train->count; i++)
        order[i] = i;

    NNActivations a;
    for (int epoch = 1; epoch <= epochs; epoch++) {
        rng_shuffle(order, train->count);
        double total_loss = 0.0;
        int correct = 0;

        for (int start = 0; start < train->count; start += batch_size) {
            int end = start + batch_size < train->count ? start + batch_size : train->count;
            memset(g, 0, sizeof(*g));

            for (int i = start; i < end; i++) {
                const float *x = mnist_image(train, order[i]);
                int y = train->labels[order[i]];

                if (nn_forward(m, x, &a) == y)
                    correct++;
                total_loss += -logf(a.p[y] + 1e-7f);
                nn_backward(m, x, y, &a, g);
            }
            nn_update(m, g, lr / (end - start));
        }

        printf("epoch %2d | loss %.4f | train acc %.2f%% | test acc %.2f%%\n",
               epoch, total_loss / train->count, 100.0f * correct / train->count,
               100.0f * nn_accuracy(m, test));
        fflush(stdout);
    }

    free(order);
    free(g);
}

/* accuracy = (# images where the predicted digit == label) / count */
float nn_accuracy(const NNModel *m, const Dataset *ds)
{
    NNActivations a;
    int correct = 0;
    for (int i = 0; i < ds->count; i++)
        if (nn_forward(m, mnist_image(ds, i), &a) == ds->labels[i])
            correct++;
    return (float)correct / ds->count;
}

/* File layout: "DGNN", then int32 sizes (784, 128, 10) so a file from a
 * different network shape is rejected, then W1, b1, W2, b2 as raw floats.
 * Floats are stored in this machine's byte order. */
int nn_save(const NNModel *m, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path);
        return -1;
    }
    int dims[3] = {MNIST_PIXELS, NN_HIDDEN, NUM_CLASSES};
    int ok = fwrite(NN_MAGIC, 1, 4, f) == 4 &&
             fwrite(dims, sizeof(int), 3, f) == 3 &&
             fwrite(m, sizeof(*m), 1, f) == 1;
    if (fclose(f) != 0 || !ok) {
        fprintf(stderr, "failed writing %s\n", path);
        return -1;
    }
    return 0;
}

/* Read a file written by nn_save, checking the magic and sizes first. */
int nn_load(NNModel *m, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s (run ./digits train-nn first)\n", path);
        return -1;
    }
    char magic[4];
    int dims[3];
    int ok = fread(magic, 1, 4, f) == 4 && memcmp(magic, NN_MAGIC, 4) == 0 &&
             fread(dims, sizeof(int), 3, f) == 3 &&
             dims[0] == MNIST_PIXELS && dims[1] == NN_HIDDEN && dims[2] == NUM_CLASSES &&
             fread(m, sizeof(*m), 1, f) == 1;
    fclose(f);
    if (!ok) {
        fprintf(stderr, "%s is not a valid weights file for this network\n", path);
        return -1;
    }
    return 0;
}
