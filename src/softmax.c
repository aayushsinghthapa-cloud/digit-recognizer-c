/* softmax.c - softmax regression trained with mini-batch gradient descent. */
#include "softmax.h"
#include "matrix.h"
#include "rng.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Zero init is fine here: with a single layer, every weight still gets a
 * different gradient because each one sees a different pixel. */
void softmax_init(SoftmaxModel *m)
{
    memset(m, 0, sizeof(*m));
}

/* z = W x + b (10 scores, "logits"), then probs = softmax(z). */
void softmax_forward(const SoftmaxModel *m, const float *x, float *probs)
{
    mat_vec_mul_add(m->W, x, m->b, probs, NUM_CLASSES, MNIST_PIXELS);
    vec_softmax(probs, NUM_CLASSES);
}

/* For each mini-batch:
 *   forward:  p = softmax(W x + b)
 *   loss:     L = -log(p[y])                       (cross-entropy)
 *   gradient: dL/dz = p - onehot(y)
 *             dL/dW = (p - onehot(y)) x^T,  dL/db = p - onehot(y)
 *   update:   W -= lr * (average gradient over the batch), same for b
 * The training order is reshuffled every epoch. */
void softmax_train(SoftmaxModel *m, const Dataset *train, const Dataset *test,
                   int epochs, int batch_size, float lr)
{
    int *order = malloc(train->count * sizeof(int));
    float *grad_W = malloc(sizeof(m->W));
    if (!order || !grad_W) {
        fprintf(stderr, "out of memory\n");
        free(order);
        free(grad_W);
        return;
    }
    for (int i = 0; i < train->count; i++)
        order[i] = i;

    float grad_b[NUM_CLASSES];
    float p[NUM_CLASSES];

    for (int epoch = 1; epoch <= epochs; epoch++) {
        rng_shuffle(order, train->count);
        double total_loss = 0.0;
        int correct = 0;

        for (int start = 0; start < train->count; start += batch_size) {
            int end = start + batch_size < train->count ? start + batch_size : train->count;
            memset(grad_W, 0, sizeof(m->W));
            memset(grad_b, 0, sizeof(grad_b));

            for (int i = start; i < end; i++) {
                const float *x = mnist_image(train, order[i]);
                int y = train->labels[order[i]];

                softmax_forward(m, x, p);
                total_loss += -logf(p[y] + 1e-7f);   /* tiny epsilon avoids log(0) */
                if (vec_argmax(p, NUM_CLASSES) == y)
                    correct++;

                p[y] -= 1.0f;   /* p is now dL/dz = p - onehot(y) */
                mat_add_outer(grad_W, p, x, 1.0f, NUM_CLASSES, MNIST_PIXELS);
                for (int k = 0; k < NUM_CLASSES; k++)
                    grad_b[k] += p[k];
            }

            /* Step downhill by lr times the batch-average gradient. */
            float step = lr / (end - start);
            for (int j = 0; j < NUM_CLASSES * MNIST_PIXELS; j++)
                m->W[j] -= step * grad_W[j];
            for (int k = 0; k < NUM_CLASSES; k++)
                m->b[k] -= step * grad_b[k];
        }

        printf("epoch %2d | loss %.4f | train acc %.2f%% | test acc %.2f%%\n",
               epoch, total_loss / train->count, 100.0f * correct / train->count,
               100.0f * softmax_accuracy(m, test));
    }

    free(order);
    free(grad_W);
}

/* accuracy = (# images where argmax(probs) == label) / count */
float softmax_accuracy(const SoftmaxModel *m, const Dataset *ds)
{
    float p[NUM_CLASSES];
    int correct = 0;
    for (int i = 0; i < ds->count; i++) {
        softmax_forward(m, mnist_image(ds, i), p);
        if (vec_argmax(p, NUM_CLASSES) == ds->labels[i])
            correct++;
    }
    return (float)correct / ds->count;
}
