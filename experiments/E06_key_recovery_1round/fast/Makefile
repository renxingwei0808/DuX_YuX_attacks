CC      ?= gcc
CFLAGS  ?= -O3 -march=native -funroll-loops -fopenmp -Wall -Wextra

all: mom

mom: mom.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f mom

.PHONY: all clean
