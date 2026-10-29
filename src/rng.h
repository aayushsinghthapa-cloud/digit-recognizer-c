/* rng.h - small deterministic random number generator.
 * Our own generator (instead of rand()) gives identical results on every OS. */
#ifndef RNG_H
#define RNG_H

#include <stdint.h>

/* Set the generator's starting state. Same seed -> same sequence. */
void rng_seed(uint64_t seed);

/* Uniform random float in [0, 1). */
float rng_uniform(void);

/* Shuffle arr[0..n-1] in place (Fisher-Yates). */
void rng_shuffle(int *arr, int n);

#endif
