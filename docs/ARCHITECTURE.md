# Architecture

## Module structure

Each arrow means "uses". `main.c` is the only file that knows about the command line; every other module is a small library.

```mermaid
flowchart TD
    main["main.c<br/>CLI: show, knn, train-softmax, train-nn, test-nn, eval"]
    mnist["mnist.c<br/>IDX loader, normalization, ASCII art"]
    knn["knn.c<br/>k-nearest-neighbour"]
    softmax["softmax.c<br/>softmax regression 784→10"]
    nn["nn.c<br/>neural network 784→128→10"]
    eval["eval.c<br/>confusion matrix, mistakes"]
    matrix["matrix.c<br/>dot, mat-vec, outer product, softmax"]
    rng["rng.c<br/>seeded RNG, shuffle, normal samples"]

    main --> mnist
    main --> knn
    main --> softmax
    main --> nn
    main --> rng
    main --> eval
    eval --> mnist
    knn --> mnist
    knn --> matrix
    softmax --> mnist
    softmax --> matrix
    softmax --> rng
    nn --> mnist
    nn --> matrix
    nn --> rng
```

## Training loop (neural network)

```mermaid
flowchart LR
    A["Shuffle training order"] --> B["Take next mini-batch<br/>(64 images)"]
    B --> C["Forward pass<br/>x → h = ReLU(W1x+b1)<br/>→ p = softmax(W2h+b2)"]
    C --> D["Loss<br/>L = −log p[y]"]
    D --> E["Backward pass<br/>chain rule: dz → dW2, dh → dW1"]
    E --> F{"More images<br/>in batch?"}
    F -- yes --> C
    F -- no --> G["Update<br/>θ −= lr · mean gradient"]
    G --> H{"More batches?"}
    H -- yes --> B
    H -- no --> I["Report loss, train acc,<br/>test acc"]
    I --> J{"More epochs?"}
    J -- yes --> A
    J -- no --> K["Save weights<br/>models/nn.bin"]
```

## Data shapes at each step

For one image through the neural network ([`nn_forward`](../src/nn.c#L34)):

| Step | Formula | Shape |
|---|---|---|
| Raw file | bytes 0–255 | 28 × 28 |
| Input `x` | byte / 255 | 784 |
| Hidden pre-activation `h_pre` | `W1 · x + b1` | (128×784)·(784) + 128 → **128** |
| Hidden `h` | `max(0, h_pre)` | **128** |
| Logits `z` | `W2 · h + b2` | (10×128)·(128) + 10 → **10** |
| Probabilities `p` | `softmax(z)` | **10**, sums to 1 |
| Prediction | `argmax(p)` | one digit 0–9 |

Backward pass shapes ([`nn_backward`](../src/nn.c#L51)): `dz` 10 → `dW2` 10×128 → `dh` 128 → `dW1` 128×784. Each gradient has exactly the shape of the thing it adjusts.

Parameter count: 128·784 + 128 + 10·128 + 10 = **101,770**. Saved as 407,096 bytes (16-byte header + 101,770 × 4-byte floats).

## Role of every file

| File | Role |
|---|---|
| [src/main.c](../src/main.c) | Parses the command, loads datasets, calls the right module, frees everything. Holds file paths and the fixed seed (42). |
| [src/mnist.c](../src/mnist.c) / [.h](../src/mnist.h) | Reads big-endian IDX headers, validates them, converts pixels to floats in [0, 1], and prints images as ASCII. Defines `Dataset`. |
| [src/matrix.c](../src/matrix.c) / [.h](../src/matrix.h) | All the math primitives, hand-written: dot product, squared distance, matrix·vector, transposed matrix·vector, outer-product accumulate, scaled add, softmax, argmax. |
| [src/rng.c](../src/rng.c) / [.h](../src/rng.h) | xorshift64* generator, so results are identical on every machine. Uniform and normal samples and the Fisher–Yates shuffle. |
| [src/knn.c](../src/knn.c) / [.h](../src/knn.h) | Baseline with no learning: the k closest training images vote. |
| [src/softmax.c](../src/softmax.c) / [.h](../src/softmax.h) | One linear layer + softmax, trained with mini-batch gradient descent. |
| [src/nn.c](../src/nn.c) / [.h](../src/nn.h) | Two-layer network: He init, forward pass, hand-written backprop, update, save/load. |
| [src/eval.c](../src/eval.c) / [.h](../src/eval.h) | Confusion matrix, most-confused pairs, ASCII prints of the most confident mistakes. Takes plain arrays of predictions, so it works with any model. |
| [Makefile](../Makefile) | `make`, `make debug` (AddressSanitizer), `make leakcheck`, `make data`. |
| [scripts/download_mnist.sh](../scripts/download_mnist.sh) | Downloads and unzips MNIST into `data/`. |

## Memory layout

Every matrix is one flat `float` array in **row-major** order: element (r, c) of a matrix with `cols` columns is `m[r * cols + c]`. The dataset is the same: image `i` starts at `images[i * 784]`. This keeps memory reads sequential, which is what makes the plain C loops fast.
