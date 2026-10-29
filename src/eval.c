/* eval.c - evaluation reports. */
#include "eval.h"

#include <stdio.h>
#include <string.h>

/* Count every (true label, predicted label) combination. */
void eval_confusion(const unsigned char *labels, const int *pred, int n,
                    int conf[NUM_CLASSES][NUM_CLASSES])
{
    memset(conf, 0, sizeof(int) * NUM_CLASSES * NUM_CLASSES);
    for (int i = 0; i < n; i++)
        conf[labels[i]][pred[i]]++;
}

/* Rows = true digit, columns = predicted digit. The diagonal is correct answers.
 * Per-digit accuracy = conf[d][d] / (sum of row d). */
void eval_print_confusion(int conf[NUM_CLASSES][NUM_CLASSES])
{
    printf("confusion matrix (rows = true digit, columns = predicted)\n\n");
    printf("true\\pred");
    for (int p = 0; p < NUM_CLASSES; p++)
        printf("%7d", p);
    printf("   | accuracy\n");

    int total = 0, correct = 0;
    for (int t = 0; t < NUM_CLASSES; t++) {
        int row = 0;
        printf("%9d", t);
        for (int p = 0; p < NUM_CLASSES; p++) {
            char cell[16];
            row += conf[t][p];
            if (p == t)
                snprintf(cell, sizeof(cell), "[%d]", conf[t][p]);
            else if (conf[t][p] == 0)
                snprintf(cell, sizeof(cell), ".");
            else
                snprintf(cell, sizeof(cell), "%d", conf[t][p]);
            printf("%7s", cell);
        }
        printf("   | %6.2f%%\n", row ? 100.0f * conf[t][t] / row : 0.0f);
        total += row;
        correct += conf[t][t];
    }
    printf("\noverall: %d / %d correct = %.2f%%\n", correct, total, 100.0f * correct / total);
}

/* Repeatedly pick the largest remaining off-diagonal cell (selection, 90 cells). */
void eval_print_confused_pairs(int conf[NUM_CLASSES][NUM_CLASSES], int top)
{
    int used[NUM_CLASSES][NUM_CLASSES] = {{0}};
    printf("\nmost-confused pairs:\n");
    for (int k = 0; k < top; k++) {
        int bt = -1, bp = -1, best = 0;
        for (int t = 0; t < NUM_CLASSES; t++)
            for (int p = 0; p < NUM_CLASSES; p++)
                if (t != p && !used[t][p] && conf[t][p] > best) {
                    best = conf[t][p];
                    bt = t;
                    bp = p;
                }
        if (bt < 0)
            break;
        used[bt][bp] = 1;
        printf("  %2d. true %d predicted as %d : %3d times\n", k + 1, bt, bp, best);
    }
}

/* Repeatedly pick the highest-confidence mistake not yet shown. */
void eval_print_mistakes(const Dataset *ds, const int *pred, const float *confidence,
                         int count)
{
    int shown[64] = {0};
    if (count > 64)
        count = 64;

    for (int k = 0; k < count; k++) {
        int best = -1;
        for (int i = 0; i < ds->count; i++) {
            if (pred[i] == ds->labels[i])
                continue;
            int seen = 0;
            for (int j = 0; j < k; j++)
                if (shown[j] == i)
                    seen = 1;
            if (!seen && (best < 0 || confidence[i] > confidence[best]))
                best = i;
        }
        if (best < 0)
            break;
        shown[k] = best;
        printf("\ntest image #%d: true %d, predicted %d (%.1f%% confident)\n",
               best, ds->labels[best], pred[best], 100.0f * confidence[best]);
        mnist_print(mnist_image(ds, best));
    }
}
