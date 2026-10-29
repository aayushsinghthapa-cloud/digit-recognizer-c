# Digit Recognizer in C

A handwritten digit recognizer (MNIST) written from scratch in C99 using only the standard library and `math.h`. No ML or matrix libraries.

Three classifiers are built up step by step:
1. **k-nearest-neighbour**: no learning, just "which training images look most similar?"
2. **Softmax regression**: one learned linear layer (784 → 10).
3. **Neural network**: 784 → 128 (ReLU) → 10, with backpropagation written by hand.

Plus an optional **draw-a-digit window** (raylib) that shows the network's live prediction as you draw.

Everything is hand-written, including matrix math, softmax, cross-entropy, backprop, random number generation and file parsing.

## Results
Accuracy on the 10,000-image MNIST test set:

| Model | Test accuracy | Notes |
|---|---|---|
| k-nearest-neighbour (k=3) | 97.17% | No training; ~7 min to classify 10k images |
| Softmax regression (784→10) | 92.28% | 10 epochs, batch 64, lr 0.1; ~5 s to train |
| Neural network (784→128→10) | **97.51%** | ReLU hidden layer, 10 epochs, batch 64, lr 0.1; ~60 s to train |

### Where the network goes wrong
From `./digits eval`, the most common mistakes are **4 → 9** (20 times), **7 → 9** (15), **5 → 3** (11) and **7 → 2** (11). These are the pairs that differ by one stroke: a 4 with a closed top looks like a 9, and a 7 with a hooked top looks like a 9. Several of the most confident mistakes are digits a human would also struggle with.

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
./digits train-nn         # train the neural network, save to models/nn.bin (~60 s)
./digits test-nn          # reload saved weights, report test accuracy
./digits eval             # confusion matrix + most-confused pairs + worst mistakes
make gui && ./digits-gui  # optional: draw digits with the mouse (needs `brew install raylib`)
make leakcheck ARGS="show 42"   # confirm no memory leaks (macOS)
```

## Project structure
```
src/
  main.c      command-line interface
  mnist.c     IDX loader + ASCII viewer
  matrix.c    hand-written vector/matrix math
  rng.c       seeded random numbers, shuffle
  knn.c       k-nearest-neighbour baseline
  softmax.c   softmax regression
  nn.c        neural network + backprop + save/load
  eval.c      confusion matrix + error analysis
  preprocess.c  drawing → MNIST-style 28×28 input
gui/
  gui.c       optional raylib drawing window
scripts/    helper scripts (dataset download)
docs/       ARCHITECTURE.md, MATH.md, EXPLAIN_IT.md
data/       MNIST files (downloaded, not committed)
models/     saved weights (not committed)
```

## Documentation
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): module diagram, training-loop diagram, data shapes, role of every file
- [docs/MATH.md](docs/MATH.md): dot products → backprop, each linked to the function that implements it
- [docs/EXPLAIN_IT.md](docs/EXPLAIN_IT.md): 30-second pitch, 2-minute explanation, 15 interview questions with answers

## What I learned
- **Similarity is not learning.** k-NN scores 97% by brute force, but it has to keep all 60,000 images and compares against every one of them for every prediction. A trained network packs what it learned into 101,770 weights and predicts in microseconds.
- **Linear models have a ceiling.** Softmax regression gets one template per digit and stalls around 92%. A hidden layer with a non-linearity (ReLU) lets the model combine features, which pushes it past 97%.
- **Backprop is just the chain rule**, applied layer by layer from the loss back to each weight. Writing it by hand made `dz = p − onehot(y)`, `dW = dz·xᵀ` and `dh = Wᵀ·dz` concrete.
- **Small engineering details matter.** You have to read IDX headers byte by byte because they're big-endian. Softmax needs the max subtracted first or it overflows. Random init breaks symmetry. A fixed seed makes results reproducible. AddressSanitizer and `leaks` catch memory bugs that C would otherwise hide.
- **The input has to match the training data.** A mouse drawing has to be cropped, scaled and centred exactly the way MNIST was, or the network sees something unlike anything it trained on.
- **Look at the errors, not only the accuracy.** The confusion matrix shows *which* digits get mixed up, and some "mistakes" are genuinely ambiguous handwriting.
