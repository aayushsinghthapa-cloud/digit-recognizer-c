/* mnist.h - load MNIST IDX files and display digits as ASCII art. */
#ifndef MNIST_H
#define MNIST_H

#define MNIST_ROWS   28
#define MNIST_COLS   28
#define MNIST_PIXELS (MNIST_ROWS * MNIST_COLS)   /* 784 */
#define NUM_CLASSES  10                          /* digits 0-9 */

/* A set of images and their labels.
 * images is count x 784 floats, row-major: image i starts at images[i * 784].
 * Each pixel is in [0, 1] (0 = black background, 1 = full ink). */
typedef struct {
    int count;
    float *images;
    unsigned char *labels;   /* each in 0..9 */
} Dataset;

/* Load an image file and its matching label file. Returns 0 on success, -1 on error. */
int mnist_load(const char *images_path, const char *labels_path, Dataset *ds);

/* Free the memory owned by ds and zero it. */
void mnist_free(Dataset *ds);

/* Return a pointer to the 784 pixels of image i. */
const float *mnist_image(const Dataset *ds, int i);

/* Print one 28x28 image to stdout as ASCII art. */
void mnist_print(const float *image);

#endif
