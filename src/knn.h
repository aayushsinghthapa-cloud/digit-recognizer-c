/* knn.h - k-nearest-neighbour classifier (no training, just comparison). */
#ifndef KNN_H
#define KNN_H

#include "mnist.h"

#define KNN_MAX_K 32

/* Predict the digit in image by majority vote of its k closest training images. */
int knn_predict(const Dataset *train, const float *image, int k);

/* Classify the first n test images and return the fraction predicted correctly. */
float knn_evaluate(const Dataset *train, const Dataset *test, int n, int k);

#endif
