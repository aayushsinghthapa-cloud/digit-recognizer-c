# The math, explained from scratch

Each section explains one idea, then links to the exact C function that implements it.

---

## 1. Images as vectors

A 28×28 image is 784 numbers. Lay them out in one row and the image becomes a **vector**: a point in 784-dimensional space. Similar-looking images are points that sit close together. Every model here works on these 784-number vectors.

Pixels are scaled from 0–255 down to 0–1 (`x = byte / 255`) so all the numbers are small and on the same scale.
→ [`load_images`](../src/mnist.c#L25)

---

## 2. Distance (used by k-NN)

How different are two images? Subtract them pixel by pixel, square each difference so negatives don't cancel out, and add:

$$d(a, b) = \sum_{i=1}^{784} (a_i - b_i)^2$$

This is the squared Euclidean distance. The square root is skipped because it never changes which distance is smallest.
→ [`vec_sq_dist`](../src/matrix.c#L17), used by [`knn_predict`](../src/knn.c#L12)

---

## 3. Dot product

Multiply two vectors element by element and add up the results:

$$a \cdot b = \sum_i a_i b_i$$

**Intuition:** the dot product measures how well two vectors line up. Suppose `w` is a vector of weights, with positive values where a "1" usually has ink and negative values elsewhere. Then `w · x` is large when image `x` has ink in the "1" places. That turns a dot product into a pattern detector.
→ [`vec_dot`](../src/matrix.c#L7)

---

## 4. Matrix–vector multiply

A matrix is a stack of rows. Multiplying a matrix by a vector just means **one dot product per row**:

$$y_r = \sum_c W_{rc} x_c + b_r$$

With `W` of shape 10×784, you get 10 scores: one pattern detector per digit. In the network, `W1` (128×784) runs 128 detectors over the image at once.
→ [`mat_vec_mul_add`](../src/matrix.c#L28)

The **transpose** product `y = Wᵀv` runs the matrix "backwards": each column collects a weighted sum of the rows. Backprop uses this to send error signals from the output back toward the input.
→ [`mat_t_vec_mul`](../src/matrix.c#L37)

---

## 5. Softmax: scores to probabilities

The 10 raw scores ("logits") can be any numbers. Softmax turns them into probabilities that are all positive and sum to 1:

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Using `e^z` makes everything positive and stretches the gaps between scores, so the top score dominates. To avoid overflow, we compute `e^(z_i − max(z))` instead. Subtracting the same amount from every score cancels between the top and bottom of the fraction, so the answer is identical, but the largest exponent is now `e⁰ = 1`.
→ [`vec_softmax`](../src/matrix.c#L69)

---

## 6. Cross-entropy loss: how wrong are we?

If the true digit is `y`, the loss is:

$$L = -\log p_y$$

| Probability on the right answer | Loss |
|---|---|
| 0.99 | 0.01 |
| 0.5 | 0.69 |
| 0.01 | 4.6 |

It's small when the model is confidently right and huge when it's confidently wrong. Training means making the average loss smaller. The code adds a tiny `1e-7` inside the log so `log(0)` can never happen.
→ in [`softmax_train`](../src/softmax.c#L32) and [`nn_train`](../src/nn.c#L83)

---

## 7. Gradient descent: walking downhill

Think of the loss as a landscape where every weight is a direction you can move. The **gradient** `∂L/∂w` says which way is uphill and how steep it is. So step the other way:

$$w \leftarrow w - \eta \frac{\partial L}{\partial w}$$

`η` (eta) is the **learning rate**, the size of each step. Too large and you overshoot the valley; too small and training crawls.

**Mini-batches:** computing the gradient over all 60,000 images per step is slow, and computing it from one image is noisy. We average it over 64 images, which is a good-enough estimate at a fraction of the cost. **Shuffling** every epoch keeps each batch a fair random sample.
→ [`nn_update`](../src/nn.c#L73), [`rng_shuffle`](../src/rng.c#L42), [`vec_add_scaled`](../src/matrix.c#L46)

---

## 8. The softmax + cross-entropy gradient

Working through the calculus for softmax followed by cross-entropy gives a remarkably simple answer:

$$\frac{\partial L}{\partial z_k} = p_k - [k = y]$$

That's "what you predicted minus what you should have predicted". If the true digit is 3 and `p₃ = 0.7`, the error on score 3 is −0.3 (push it up), and every other score's error is `p_k` (push it down).

For a weight `W_kc` linking pixel `c` to score `k`, the gradient is `(p_k − [k=y]) · x_c`. For the whole matrix, that's an **outer product**, `dW = dz · xᵀ`. A pixel that's 0 contributes nothing, so its weight gets no blame.
→ [`mat_add_outer`](../src/matrix.c#L53), used in [`softmax_train`](../src/softmax.c#L32) and [`nn_backward`](../src/nn.c#L51)

---

## 9. ReLU and why we need a hidden layer

Softmax regression has one template per digit, which isn't enough because there are many ways to write a 7. The neural network adds a hidden layer of 128 **feature detectors**: `h = ReLU(W1·x + b1)`.

$$\text{ReLU}(a) = \max(0, a)$$

ReLU keeps positive signals and zeroes out negative ones. Without it, `W2·(W1·x)` is just one big matrix `(W2W1)·x`, which is still a linear model, so the extra layer would add nothing. The bend at 0 is what lets the network combine features, e.g. "has a loop on top AND a vertical stroke on the right → 9".
→ [`nn_forward`](../src/nn.c#L34)

---

## 10. The chain rule

If `L` depends on `z`, and `z` depends on `w`, then:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w}$$

In words: *how much L changes per unit of w* = *(how much L changes per unit of z)* × *(how much z changes per unit of w)*. You multiply the local slopes along the path.

---

## 11. Backpropagation: the chain rule applied layer by layer

Backprop applies the chain rule from the output back to the input, reusing each result for the next step:

| Step | Formula | Meaning |
|---|---|---|
| Output error | `dz = p − onehot(y)` | how wrong each score was |
| Output weights | `dW2 = dz · hᵀ`, `db2 = dz` | blame on W2 = error × hidden activation |
| Back to hidden | `dh = W2ᵀ · dz` | each hidden neuron's share of the blame |
| Through ReLU | `dh_pre = dh` where `h_pre > 0`, else `0` | ReLU's slope is 1 when on, 0 when off |
| Input weights | `dW1 = dh_pre · xᵀ`, `db1 = dh_pre` | blame on W1 = hidden error × pixel |

The forward pass saves `h_pre`, `h` and `p` in `NNActivations` so the backward pass can reuse them. Gradients from all 64 images in a batch are added up, then [`nn_update`](../src/nn.c#L73) takes one step.
→ [`nn_backward`](../src/nn.c#L51)

---

## 12. Weight initialization

If every weight started at 0, all 128 hidden neurons would compute the same thing and get the same gradient, so they'd stay identical forever. Random starting values break that symmetry.

How large should they be? **He initialization** draws each weight from a normal distribution with standard deviation `√(2 / fan_in)`, where `fan_in` is the number of inputs (784 for W1). This keeps the size of the signal about the same from layer to layer. The factor 2 makes up for ReLU zeroing out roughly half its inputs.
→ [`he_init`](../src/nn.c#L17), [`rng_normal`](../src/rng.c#L31) (Box–Muller: `z = √(−2 ln u₁) · cos(2π u₂)`)
