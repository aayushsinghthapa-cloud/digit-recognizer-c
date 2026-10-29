# Digit Recognizer in C

A handwritten digit recognizer (MNIST) written from scratch in C99 using only the standard library and `math.h`. No ML or matrix libraries.

## Status
Work in progress, built in stages: data loader ✅ → k-NN → softmax regression → neural network → evaluation.

## Build & run
```sh
make data      # download MNIST into data/
make           # build ./digits
make debug     # build ./digits-debug with AddressSanitizer
./digits help
./digits show 42           # draw training image #42 as ASCII art
./digits show 42 --test    # same, from the test set
make leakcheck ARGS="show 42"   # confirm no memory leaks (macOS)
```

## Project structure
```
src/        C source files
scripts/    helper scripts (dataset download)
docs/       architecture, math, and explanation notes
data/       MNIST files (downloaded, not committed)
models/     saved weights (not committed)
```
