# Digit Recognizer in C

A handwritten digit recognizer (MNIST) written from scratch in C99 using only the standard library and `math.h`. No ML or matrix libraries.

## Status
Work in progress, built in stages: data loader ✅ → k-NN ✅ → softmax regression ✅ → neural network → evaluation.

## Results
Accuracy on the 10,000-image MNIST test set:

| Model | Test accuracy | Notes |
|---|---|---|
| k-nearest-neighbour (k=3) | 97.17% | No training; ~7 min to classify 10k images |
| Softmax regression (784→10) | 92.28% | 10 epochs, batch 64, lr 0.1; ~5 s to train |

## Build & run
```sh
make data      # download MNIST into data/
make           # build ./digits
make debug     # build ./digits-debug with AddressSanitizer
./digits help
./digits show 42           # draw training image #42 as ASCII art
./digits show 42 --test    # same, from the test set
./digits knn              # k-NN on the first 1000 test images (~40 s)
./digits knn 10000 3      # full test set, k=3 (~7 min)
./digits train-softmax    # train softmax regression, 10 epochs (~5 s)
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
