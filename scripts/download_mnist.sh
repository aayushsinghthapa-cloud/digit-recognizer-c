#!/bin/sh
# Download the four MNIST IDX files into data/ and unzip them.
set -e
MIRROR="https://raw.githubusercontent.com/fgnt/mnist/master"
DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$DIR"

for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
         t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    if [ -f "$DIR/$f" ]; then
        echo "already have $f"
        continue
    fi
    echo "downloading $f ..."
    curl -fsSL "$MIRROR/$f.gz" -o "$DIR/$f.gz"
    gunzip -f "$DIR/$f.gz"
done
echo "done:"
ls -l "$DIR"
