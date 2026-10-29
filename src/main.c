/* main.c - command-line entry point: dispatches ./digits <command>. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "knn.h"
#include "mnist.h"

#define TRAIN_IMAGES "data/train-images-idx3-ubyte"
#define TRAIN_LABELS "data/train-labels-idx1-ubyte"
#define TEST_IMAGES  "data/t10k-images-idx3-ubyte"
#define TEST_LABELS  "data/t10k-labels-idx1-ubyte"

/* Print the list of available commands. */
static void usage(void)
{
    printf("usage: ./digits <command>\n"
           "  show <index> [--test]   print a training (or test) digit as ASCII art\n"
           "  knn [n_test] [k]        k-nearest-neighbour accuracy on the first n_test\n"
           "                          test images (default 1000, k=3; 10000 = full set)\n"
           "  help                    show this message\n");
}

/* ./digits show <index> [--test]: load a dataset and draw one image. */
static int cmd_show(int argc, char **argv)
{
    if (argc < 3) {
        usage();
        return 1;
    }
    int use_test = argc > 3 && strcmp(argv[3], "--test") == 0;

    Dataset ds = {0};
    if (use_test ? mnist_load(TEST_IMAGES, TEST_LABELS, &ds)
                 : mnist_load(TRAIN_IMAGES, TRAIN_LABELS, &ds))
        return 1;

    int i = atoi(argv[2]);
    if (i < 0 || i >= ds.count) {
        fprintf(stderr, "index must be between 0 and %d\n", ds.count - 1);
        mnist_free(&ds);
        return 1;
    }

    printf("%s image #%d, label: %d\n", use_test ? "test" : "train", i, ds.labels[i]);
    mnist_print(mnist_image(&ds, i));
    mnist_free(&ds);
    return 0;
}

/* ./digits knn [n_test] [k]: measure k-NN accuracy on the test set. */
static int cmd_knn(int argc, char **argv)
{
    int n = argc > 2 ? atoi(argv[2]) : 1000;
    int k = argc > 3 ? atoi(argv[3]) : 3;
    if (k < 1 || k > KNN_MAX_K) {
        fprintf(stderr, "k must be between 1 and %d\n", KNN_MAX_K);
        return 1;
    }

    Dataset train = {0}, test = {0};
    if (mnist_load(TRAIN_IMAGES, TRAIN_LABELS, &train))
        return 1;
    if (mnist_load(TEST_IMAGES, TEST_LABELS, &test)) {
        mnist_free(&train);
        return 1;
    }
    if (n < 1 || n > test.count)
        n = test.count;

    printf("k-NN with k=%d on %d test images (each compared to %d training images)\n",
           k, n, train.count);
    float acc = knn_evaluate(&train, &test, n, k);
    printf("accuracy: %.2f%%\n", 100.0f * acc);

    mnist_free(&train);
    mnist_free(&test);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2 || strcmp(argv[1], "help") == 0) {
        usage();
        return 0;
    }
    if (strcmp(argv[1], "show") == 0)
        return cmd_show(argc, argv);
    if (strcmp(argv[1], "knn") == 0)
        return cmd_knn(argc, argv);

    fprintf(stderr, "unknown command: %s\n", argv[1]);
    usage();
    return 1;
}
