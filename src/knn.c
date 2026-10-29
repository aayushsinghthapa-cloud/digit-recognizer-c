/* knn.c - k-nearest-neighbour classifier. */
#include "knn.h"
#include "matrix.h"

#include <float.h>
#include <stdio.h>

/* Compare image against every training image, keeping the k smallest
 * distances in sorted order (insertion sort into a short list).
 * Then each neighbour votes for its label; ties go to the label whose
 * member is closest, since neighbours are visited nearest-first. */
int knn_predict(const Dataset *train, const float *image, int k)
{
    float best_dist[KNN_MAX_K];
    int best_label[KNN_MAX_K];
    for (int j = 0; j < k; j++) {
        best_dist[j] = FLT_MAX;
        best_label[j] = -1;
    }

    for (int i = 0; i < train->count; i++) {
        float d = vec_sq_dist(image, mnist_image(train, i), MNIST_PIXELS);
        if (d >= best_dist[k - 1])
            continue;
        /* Shift worse neighbours down and insert this one in order. */
        int j = k - 1;
        while (j > 0 && best_dist[j - 1] > d) {
            best_dist[j] = best_dist[j - 1];
            best_label[j] = best_label[j - 1];
            j--;
        }
        best_dist[j] = d;
        best_label[j] = train->labels[i];
    }

    int votes[10] = {0};
    for (int j = 0; j < k; j++)
        votes[best_label[j]]++;

    int winner = best_label[0];
    for (int j = 1; j < k; j++)
        if (votes[best_label[j]] > votes[winner])
            winner = best_label[j];
    return winner;
}

/* accuracy = correct predictions / n */
float knn_evaluate(const Dataset *train, const Dataset *test, int n, int k)
{
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if (knn_predict(train, mnist_image(test, i), k) == test->labels[i])
            correct++;
        if ((i + 1) % 500 == 0 || i + 1 == n) {
            printf("\r  classified %5d / %d   running accuracy %.2f%%",
                   i + 1, n, 100.0f * correct / (i + 1));
            fflush(stdout);
        }
    }
    printf("\n");
    return (float)correct / n;
}
