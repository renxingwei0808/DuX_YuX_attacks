CC      ?= gcc
CFLAGS  ?= -O3 -march=native -funroll-loops -fopenmp -Wall -Wextra

all: gf2nsolve libmom2n.so

gf2nsolve: gf2nsolve.c
	$(CC) $(CFLAGS) -o $@ $<

libmom2n.so: mom2n.c
	$(CC) $(CFLAGS) -shared -fPIC -o $@ $<

clean:
	rm -f gf2nsolve libmom2n.so

.PHONY: all clean
