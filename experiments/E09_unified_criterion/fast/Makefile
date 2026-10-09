CC      ?= gcc
CFLAGS  ?= -O3 -march=native -funroll-loops -fopenmp -Wall -Wextra

all: zsx zsx_yux

zsx: zsx.c
	$(CC) $(CFLAGS) -o $@ $<

zsx_yux: zsx.c
	$(CC) $(CFLAGS) -DCIPHER_YUX -o $@ $<

clean:
	rm -f zsx zsx_yux

.PHONY: all clean
