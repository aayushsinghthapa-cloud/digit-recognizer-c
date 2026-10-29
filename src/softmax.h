/* softmax.h - softmax regression: one linear layer 784 -> 10 plus softmax. */
#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "mnist.h"

/* The model's learnable parameters. */
typedef struct {
    float W[NUM_CLASSES * MNIST_PIXELS];   /* 10 x 784 weights */
    float b[NUM_CLASSES];                  /* 10 biases */
} SoftmaxModel;

/* Set all weights and biases to zero. */
void softmax_init(SoftmaxModel *m);

/* Forward pass: probs = softmax(W x + b). probs has 10 entries summing to 1. */
void softmax_forward(const SoftmaxModel *m, const float *x, float *probs);

/* Train with mini-batch gradient descent, printing loss and accuracy each epoch. */
void softmax_train(SoftmaxModel *m, const Dataset *train, const Dataset *test,
                   int epochs, int batch_size, float lr);

/* Fraction of ds the model classifies correctly. */
float softmax_accuracy(const SoftmaxModel *m, const Dataset *ds);

#endif
