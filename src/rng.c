/* rng.c - xorshift64* random number generator. */
#include "rng.h"

#include <math.h>

static uint64_t state = 88172645463325252ULL;

/* A zero state would stay zero forever, so replace it. */
void rng_seed(uint64_t seed)
{
    state = seed ? seed : 88172645463325252ULL;
}

/* xorshift64*: scramble the state with shifts and XORs, then multiply. */
static uint64_t rng_next(void)
{
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 2685821657736338717ULL;
}

/* Take the top 24 bits and divide by 2^24 -> [0, 1). */
float rng_uniform(void)
{
    return (rng_next() >> 40) / 16777216.0f;
}

/* Box-Muller transform: two uniform samples u1, u2 -> one normal sample
 * z = sqrt(-2 ln u1) * cos(2 pi u2). u1 is kept above 0 so ln(u1) is finite. */
float rng_normal(void)
{
    float u1 = rng_uniform();
    float u2 = rng_uniform();
    if (u1 < 1e-7f)
        u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318531f * u2);
}

/* Fisher-Yates: walk backwards, swapping each slot with a random earlier one.
 * Every ordering is equally likely. */
void rng_shuffle(int *arr, int n)
{
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(rng_next() % (uint64_t)(i + 1));
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}
