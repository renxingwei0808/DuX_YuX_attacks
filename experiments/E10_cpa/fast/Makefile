CC      ?= gcc
CFLAGS  ?= -O3 -march=native -funroll-loops -fopenmp -Wall -Wextra

all: mom_cp

mom_cp: mom_cp.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f mom_cp

.PHONY: all clean
