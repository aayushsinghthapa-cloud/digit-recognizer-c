# Build the `digits` program from every .c file in src/.
CC      = cc
CFLAGS  = -std=c99 -Wall -Wextra -O2
LDLIBS  = -lm
SRC     = $(wildcard src/*.c)
BIN     = digits

.PHONY: all debug clean data

# Default: optimised build.
all: $(BIN)

$(BIN): $(SRC) $(wildcard src/*.h)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

# Debug build: symbols + AddressSanitizer (catches leaks and out-of-bounds bugs).
debug: CFLAGS = -std=c99 -Wall -Wextra -g -O0 -fsanitize=address
debug: $(SRC) $(wildcard src/*.h)
	$(CC) $(CFLAGS) -o $(BIN)-debug $(SRC) $(LDLIBS)

# Download MNIST into data/.
data:
	./scripts/download_mnist.sh

clean:
	rm -rf $(BIN) $(BIN)-debug *.dSYM
