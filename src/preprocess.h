/* preprocess.h - turn a free-hand drawing into an MNIST-style 28x28 image. */
#ifndef PREPROCESS_H
#define PREPROCESS_H

#include "mnist.h"

/* Crop the ink in a size x size canvas (values 0..1), scale it so its longer
 * side is 20 pixels, then place it in a 28x28 image centred on its centre of
 * mass, the same way the MNIST digits were prepared.
 * Returns 0 on success, -1 if the canvas has no ink. */
int preprocess_canvas(const float *canvas, int size, float out[MNIST_PIXELS]);

#endif
