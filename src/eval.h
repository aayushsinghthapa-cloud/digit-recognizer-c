/* eval.h - evaluation reports: confusion matrix, confused pairs, mistakes.
 * Works from arrays of predictions, so it doesn't depend on any one model. */
#ifndef EVAL_H
#define EVAL_H

#include "mnist.h"

/* conf[true][predicted] = how many images of digit `true` were predicted as `predicted`. */
void eval_confusion(const unsigned char *labels, const int *pred, int n,
                    int conf[NUM_CLASSES][NUM_CLASSES]);

/* Print the confusion matrix with per-digit accuracy. */
void eval_print_confusion(int conf[NUM_CLASSES][NUM_CLASSES]);

/* Print the `top` largest off-diagonal cells: the most common mistakes. */
void eval_print_confused_pairs(int conf[NUM_CLASSES][NUM_CLASSES], int top);

/* Print up to `count` misclassified images as ASCII art, most confident mistakes first.
 * confidence[i] is the probability the model gave its (wrong) prediction. */
void eval_print_mistakes(const Dataset *ds, const int *pred, const float *confidence,
                         int count);

#endif
