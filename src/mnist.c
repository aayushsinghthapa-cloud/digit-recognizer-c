/* mnist.c - IDX file loader and ASCII renderer for MNIST. */
#include "mnist.h"

#include <stdio.h>
#include <stdlib.h>

#define IMAGES_MAGIC 2051   /* 0x00000803: unsigned bytes, 3 dimensions */
#define LABELS_MAGIC 2049   /* 0x00000801: unsigned bytes, 1 dimension  */

/* Read a 4-byte big-endian unsigned integer.
 * Big-endian means the most significant byte comes first:
 * value = b0 * 2^24 + b1 * 2^16 + b2 * 2^8 + b3. */
static int read_be32(FILE *f, unsigned int *out)
{
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4)
        return -1;
    *out = ((unsigned int)b[0] << 24) | ((unsigned int)b[1] << 16) |
           ((unsigned int)b[2] << 8)  |  (unsigned int)b[3];
    return 0;
}

/* Read an image file into a newly allocated float array, scaling pixels
 * from 0..255 to 0..1 (x = byte / 255). Sets *count. Returns NULL on error. */
static float *load_images(const char *path, int *count)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s (did you run `make data`?)\n", path);
        return NULL;
    }

    unsigned int magic, n, rows, cols;
    if (read_be32(f, &magic) || read_be32(f, &n) ||
        read_be32(f, &rows)  || read_be32(f, &cols) ||
        magic != IMAGES_MAGIC || rows != MNIST_ROWS || cols != MNIST_COLS) {
        fprintf(stderr, "%s: bad image header\n", path);
        fclose(f);
        return NULL;
    }

    size_t total = (size_t)n * MNIST_PIXELS;
    unsigned char *raw = malloc(total);
    float *images = malloc(total * sizeof(float));
    if (!raw || !images || fread(raw, 1, total, f) != total) {
        fprintf(stderr, "%s: failed to read pixels\n", path);
        free(raw);
        free(images);
        fclose(f);
        return NULL;
    }

    for (size_t i = 0; i < total; i++)
        images[i] = raw[i] / 255.0f;

    free(raw);
    fclose(f);
    *count = (int)n;
    return images;
}

/* Read a label file into a newly allocated byte array. Sets *count. Returns NULL on error. */
static unsigned char *load_labels(const char *path, int *count)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s (did you run `make data`?)\n", path);
        return NULL;
    }

    unsigned int magic, n;
    if (read_be32(f, &magic) || read_be32(f, &n) || magic != LABELS_MAGIC) {
        fprintf(stderr, "%s: bad label header\n", path);
        fclose(f);
        return NULL;
    }

    unsigned char *labels = malloc(n);
    if (!labels || fread(labels, 1, n, f) != n) {
        fprintf(stderr, "%s: failed to read labels\n", path);
        free(labels);
        fclose(f);
        return NULL;
    }

    fclose(f);
    *count = (int)n;
    return labels;
}

/* Load images and labels, checking that their counts match. */
int mnist_load(const char *images_path, const char *labels_path, Dataset *ds)
{
    int n_images = 0, n_labels = 0;
    ds->images = load_images(images_path, &n_images);
    ds->labels = load_labels(labels_path, &n_labels);

    if (!ds->images || !ds->labels || n_images != n_labels) {
        if (n_images != n_labels)
            fprintf(stderr, "image/label count mismatch: %d vs %d\n", n_images, n_labels);
        mnist_free(ds);
        return -1;
    }
    ds->count = n_images;
    return 0;
}

/* Free the dataset's arrays. */
void mnist_free(Dataset *ds)
{
    free(ds->images);
    free(ds->labels);
    ds->images = NULL;
    ds->labels = NULL;
    ds->count = 0;
}

/* Image i starts 784 floats after image i-1. */
const float *mnist_image(const Dataset *ds, int i)
{
    return ds->images + (size_t)i * MNIST_PIXELS;
}

/* Map each pixel's brightness to a character from light to dark ink.
 * Each pixel is printed twice so the digit isn't squashed horizontally. */
void mnist_print(const float *image)
{
    static const char ramp[] = " .:-=+*#%@";
    const int levels = (int)sizeof(ramp) - 2;   /* last index, 9 */

    for (int r = 0; r < MNIST_ROWS; r++) {
        for (int c = 0; c < MNIST_COLS; c++) {
            char ch = ramp[(int)(image[r * MNIST_COLS + c] * levels + 0.5f)];
            putchar(ch);
            putchar(ch);
        }
        putchar('\n');
    }
}
