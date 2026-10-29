/* nn.h - two-layer neural network: 784 -> 128 (ReLU) -> 10 (softmax). */
#ifndef NN_H
#define NN_H

#include "mnist.h"

#define NN_HIDDEN 128

/* All learnable parameters. The same struct also holds gradients during training. */
typedef struct {
    float W1[NN_HIDDEN * MNIST_PIXELS];   /* 128 x 784: input -> hidden */
    float b1[NN_HIDDEN];
    float W2[NUM_CLASSES * NN_HIDDEN];    /* 10 x 128: hidden -> output */
    float b2[NUM_CLASSES];
} NNModel;

/* Values computed by the forward pass, kept for the backward pass. */
typedef struct {
    float h_pre[NN_HIDDEN];   /* hidden layer before ReLU: W1 x + b1 */
    float h[NN_HIDDEN];       /* hidden layer after ReLU */
    float p[NUM_CLASSES];     /* output probabilities */
} NNActivations;

/* Random weights (He initialization), zero biases. */
void nn_init(NNModel *m);

/* Forward pass for one image; fills a. Returns the predicted digit. */
int nn_forward(const NNModel *m, const float *x, NNActivations *a);

/* Train with mini-batch gradient descent, printing loss and accuracy each epoch. */
void nn_train(NNModel *m, const Dataset *train, const Dataset *test,
              int epochs, int batch_size, float lr);

/* Fraction of ds the network classifies correctly. */
float nn_accuracy(const NNModel *m, const Dataset *ds);

/* Write / read all weights to a binary file. Return 0 on success, -1 on error. */
int nn_save(const NNModel *m, const char *path);
int nn_load(NNModel *m, const char *path);

#endif
