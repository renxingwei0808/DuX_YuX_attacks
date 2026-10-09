CC      ?= gcc
CFLAGS  ?= -O3 -march=native -funroll-loops -fopenmp -Wall -Wextra

all: zs

zs: zs.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f zs

.PHONY: all clean
