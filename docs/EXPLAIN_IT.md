# How to explain this project

## 30-second pitch

> I built a handwritten digit recognizer from scratch in pure C, with no ML libraries and no matrix libraries. It reads the raw MNIST dataset of 70,000 handwritten digits, and I trained three models of increasing power: a nearest-neighbour baseline, a linear softmax classifier, and a two-layer neural network. I wrote the matrix math and the backpropagation by hand. The network reaches **97.5% accuracy** on 10,000 images it never saw during training, trains in about a minute on a laptop, and I can show you exactly which digits it confuses and why.

## 2-minute explanation

**The problem.** Each MNIST image is a 28×28 grid of grayscale pixels. Flatten it and you get a list of 784 numbers between 0 and 1. The task is to map those 784 numbers to one of 10 digits.

**Step 1: a baseline with no learning.** k-nearest-neighbour compares a new image with all 60,000 training images, finds the 3 most similar by pixel distance and takes a vote. That gets 97.2%, which shows that similar-looking images usually share a label. But every prediction does 47 million operations and needs the whole training set kept in memory.

**Step 2: the first model that learns.** Softmax regression learns one 784-weight "template" per digit. It scores an image against each template with a dot product, softmax turns the 10 scores into probabilities, and cross-entropy loss measures how much probability went to the right answer. Gradient descent then nudges the weights to reduce that loss, 64 images at a time, in a freshly shuffled order every epoch. It reaches 92%. It's limited because one template can't cover every way of writing a 7.

**Step 3: a neural network.** I added a hidden layer of 128 neurons with ReLU between the input and the output: 784 → 128 → 10. Each hidden neuron learns a feature, such as a stroke or a loop, and the output layer combines those features. Training uses backpropagation: the chain rule applied layer by layer, sending the output error backwards to work out how much each of the 101,770 weights contributed. It reaches 97.5%, beating k-NN, and each prediction is about 100,000 operations instead of 47 million.

**Step 4: understanding the errors.** A confusion matrix shows the network mostly confuses 4 with 9 and 7 with 9, which are shapes one stroke apart. Its most confident mistakes include digits a human would struggle with.

**Step 5: a live demo.** An optional raylib window lets you draw a digit with the mouse. The drawing is cropped, scaled to 20×20 and centred by centre of mass in 28×28, just as MNIST was prepared, then fed to the network, which updates its 10 probability bars as you draw.

**Engineering.** It's C99 with one module per job: the loader, matrix math, RNG, each model, evaluation and the CLI. Results are reproducible thanks to a fixed seed and a custom RNG. I checked memory with AddressSanitizer and the macOS `leaks` tool, and trained weights save to a validated binary file.

## 16 likely questions, with model answers

**1. Why write it in C without libraries?**
To understand what libraries like PyTorch do for you. With no library, every step has to be written out: the IDX parsing, the matrix multiply, softmax, the gradient formulas. If I can write backprop myself, I understand it. As a side effect, the whole thing has no dependencies beyond a C compiler.

**2. Walk me through what happens when the network sees one image.**
The 784 pixels are multiplied by a 128×784 weight matrix, and a bias is added, giving 128 numbers. ReLU zeroes out the negative ones. Those 128 values are multiplied by a 10×128 matrix plus a bias, giving 10 scores. Softmax turns the scores into probabilities that sum to 1, and the highest one is the prediction. See `nn_forward` in `src/nn.c`.

**3. What is backpropagation?**
The chain rule from calculus, applied efficiently. The output error is `p − onehot(y)`, predicted minus correct. That error gives the gradient for the output weights. It's then sent backwards through the transposed output weights to find each hidden neuron's share of the blame, zeroed wherever ReLU was off, and turned into the gradient for the first-layer weights. Each step reuses the step before it, which is why backprop is fast. See `nn_backward`.

**4. Why softmax and cross-entropy?**
Softmax turns arbitrary scores into a probability distribution. Cross-entropy, `−log(p_correct)`, rewards confidence in the right answer and heavily penalizes confidence in a wrong one. Together, their gradient simplifies to `p − onehot(y)`, which is cheap to compute and stable.

**5. Why do you need ReLU? Why not just two linear layers?**
Two linear layers multiply out to one linear layer, so you'd be back to the 92% softmax regression. ReLU, `max(0, x)`, adds the non-linearity that lets hidden neurons act as on/off feature detectors and lets the network combine features.

**6. Why did k-NN beat softmax regression but lose to the neural network?**
k-NN effectively has 60,000 templates, every training image, so it handles many writing styles. Softmax regression has only one template per digit. The neural network learns 128 reusable features and combines them, so it's more flexible than one template per digit, and far more compact and faster than storing every image.

**7. How do you know you're not overfitting?**
I report accuracy on the 10,000 test images, which the model never trains on. Train accuracy ended at 98.4% and test at 97.5%. That small gap shows mild overfitting, so the model mostly generalizes. If I trained much longer, train accuracy would approach 100% while test accuracy stalled or dropped.

**8. What is the learning rate, and how did you pick it?**
It's the step size in `w ← w − lr · gradient`. Too large and the loss bounces around or diverges; too small and training crawls. 0.1 with batch size 64 is a standard starting point for this kind of network, and the loss dropped smoothly every epoch, so I kept it.

**9. Why mini-batches and shuffling?**
The gradient from one image is noisy, and the gradient over all 60,000 images is expensive and gives only one update per epoch. A batch of 64 gives a good estimate cheaply, so we get about 940 updates per epoch. Shuffling each epoch keeps every batch a fair random sample and stops the model from learning the order of the data.

**10. Why initialize weights randomly, and why that particular scale?**
With all-zero weights, every hidden neuron gets the same gradient and they stay identical forever. Random values break that symmetry. He initialization scales them by √(2/fan_in) so the signal doesn't explode or fade from layer to layer. The 2 makes up for ReLU zeroing out about half the values.

**11. How did you read the MNIST files?**
They use the IDX format: a header of big-endian 32-bit integers (magic number, count, rows, columns), then raw pixel bytes. Most machines are little-endian, so I build each integer from its 4 bytes with bit shifts. I also validate the magic number and dimensions before trusting the data.

**12. How do you make results reproducible?**
A fixed seed (42) and my own xorshift random number generator, instead of `rand()`, whose sequence differs between operating systems. The same seed gives the same initial weights and the same shuffle order, so the same accuracy every run.

**13. How did you make sure there are no memory bugs?**
Every allocation has a matching free, including on error paths. I build a debug version with AddressSanitizer, which catches out-of-bounds access and use-after-free, and I run the macOS `leaks` tool on each command. All of them report 0 leaks.

**14. What are the most common mistakes, and why?**
4 → 9 (20 times), 7 → 9 (15), 5 → 3 (11) and 7 → 2 (11). These pairs differ by one stroke: a 4 with a closed top is nearly a 9. Some of the most confident mistakes are honestly ambiguous: the network was 99.7% sure test image #2654 is a 1, and it does look like one.

**15. Why do you preprocess the mouse drawing before predicting?**
The network only learned from MNIST digits, which are cropped, scaled to fit 20×20 and centred by centre of mass in a 28×28 image. A raw mouse drawing is bigger, off-centre and a different size, so it looks nothing like the training data. Repeating MNIST's exact preparation steps fixes that. I tested it by enlarging and moving all the test digits, then preprocessing them back: accuracy only changed from 97.51% to 97.48%.

**16. How would you improve it?**
- **A convolutional neural network (CNN):** it shares weights across positions, so it recognizes a stroke anywhere in the image. That typically gets above 99%.
- **Data augmentation:** small shifts and rotations of the training images.
- **A better optimizer:** momentum or Adam.
- **A learning-rate schedule** that lowers the rate over time.
- **Regularization:** dropout or weight decay, to shrink the train/test gap.
- **Speed:** batching the matrix multiplies, or using SIMD and multiple threads.
