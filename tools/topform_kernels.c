/* O14 (R8) kernels for the YuX full-block constant sums over F_p.
 *
 *   topform_kernels direct     <p> <layers> <seed> <linv.txt> [amax]
 *       sum over y in F_p^4 (block 0 after the Lemma-2 substitution, i.e. the
 *       first S^{-1} layer of block 0 is skipped; blocks 1-3 are random
 *       constants) of all 16 words after <layers> layers, with independent
 *       random round keys drawn from <seed>.  p = 257, layers = 6 is the 2^32
 *       verification of the projective prediction (86, 231, 222, 244).
 *       [amax] restricts the outermost loop (timing only).
 *   topform_kernels projective <p> <layers> <linv.txt>
 *       c_i(p) = -(sum over the p^3+p^2+p+1 projective points of the top forms);
 *       reported for every word whose formal degree is >= T = 4(p-1) and a
 *       multiple of p-1 (the exact-hit words), 0 for words with D < T.
 *       p = 65537, layers = 10 is 2^48 points: ~2^56 simple ops, ~1 day on
 *       ~100 cores; p = 257 takes seconds.
 *
 * linv.txt = 16 lines x 16 ints, the matrix of L^{-1}, produced by
 *   python tools/topform_constants.py --p <p> --export-linv linv.txt
 * (the rotation direction is taken from the repository, never hard-coded).
 * Build: gcc -O3 -march=native -fopenmp -o topform_kernels tools/topform_kernels.c
 * The Python tool (tools/topform_constants.py) is the reference; this file
 * only makes p = 257 / 65537 affordable.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef uint64_t u64;
static u64 P;
static int M[16][16];
static int ROWS[16][16], NR[16];           /* non-zero columns per row */

static inline u64 md(u64 x) { return x % P; }

static void read_linv(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    for (int i = 0; i < 16; i++) for (int j = 0; j < 16; j++)
        if (fscanf(f, "%d", &M[i][j]) != 1) { fprintf(stderr, "bad linv\n"); exit(1); }
    fclose(f);
    for (int i = 0; i < 16; i++) { NR[i] = 0; for (int j = 0; j < 16; j++) if (M[i][j]) ROWS[i][NR[i]++] = j; }
}

/* ---- real cipher components (F_p, YuX): S^{-1} = Pf^4, L^{-1} = M ---- */
static inline void sinv(u64 *x, u64 alpha) {
    for (int r = 0; r < 4; r++) {
        u64 y3 = md(x[0] + x[1] * x[2] + x[3] + alpha);
        x[0] = x[1]; x[1] = x[2]; x[2] = x[3]; x[3] = y3;
    }
}
static inline void linv(const u64 *x, u64 *out) {
    for (int i = 0; i < 16; i++) { u64 s = 0; for (int k = 0; k < NR[i]; k++) s += (u64)M[i][ROWS[i][k]] * x[ROWS[i][k]]; out[i] = md(s); }
}

/* ---- top forms with degree tracking (mirrors tools/topform_constants.py) ---- */
typedef struct { u64 v; long d; } tf;
static inline tf top3(tf a, tf b, tf c) {
    long dm = a.d; if (b.d > dm) dm = b.d; if (c.d > dm) dm = c.d;
    u64 s = 0; if (a.d == dm) s += a.v; if (b.d == dm) s += b.v; if (c.d == dm) s += c.v;
    tf r = { md(s), dm }; return r;
}
static inline void sinv_top(const tf *u, tf *y) {
    tf m;
    m.v = md(u[1].v * u[2].v); m.d = u[1].d + u[2].d; y[0] = top3(u[0], m, u[3]);
    m.v = md(u[2].v * u[3].v); m.d = u[2].d + u[3].d; y[1] = top3(u[1], m, y[0]);
    m.v = md(u[3].v * y[0].v); m.d = u[3].d + y[0].d; y[2] = top3(u[2], m, y[1]);
    m.v = md(y[0].v * y[1].v); m.d = y[0].d + y[1].d; y[3] = top3(u[3], m, y[2]);
}
static void topforms(const u64 *y, int layers, tf *st) {
    for (int i = 0; i < 16; i++) { st[i].v = (i < 4) ? y[i] : 0; st[i].d = (i < 4) ? 1 : 0; }
    tf u[16], nw[16];
    for (int l = 2; l <= layers; l++) {
        for (int i = 0; i < 16; i++) {
            long dm = -1; for (int k = 0; k < NR[i]; k++) if (st[ROWS[i][k]].d > dm) dm = st[ROWS[i][k]].d;
            u64 s = 0; for (int k = 0; k < NR[i]; k++) { int j = ROWS[i][k]; if (st[j].d == dm) s += (u64)M[i][j] * st[j].v; }
            u[i].v = md(s); u[i].d = dm;
        }
        for (int b = 0; b < 4; b++) sinv_top(u + 4 * b, nw + 4 * b);
        memcpy(st, nw, sizeof(nw));
    }
}

static int mode_direct(int argc, char **argv) {
    if (argc < 6) return 1;
    P = strtoull(argv[2], 0, 10); int layers = atoi(argv[3]); unsigned seed = atoi(argv[4]); read_linv(argv[5]);
    u64 amax = argc > 6 ? strtoull(argv[6], 0, 10) : P;
    u64 alpha = 205 % P;
    srand(seed);
    u64 rk[32][16], cst[16];
    for (int l = 0; l < 32; l++) for (int i = 0; i < 16; i++) rk[l][i] = rand() % P;
    for (int i = 0; i < 16; i++) cst[i] = rand() % P;
    /* layer-1 outputs of blocks 1..3 (constants); block 0 = free y */
    u64 base[16];
    for (int i = 0; i < 16; i++) base[i] = md(cst[i] + P - rk[0][i]);
    for (int b = 1; b < 4; b++) sinv(base + 4 * b, alpha);
    u64 sums[16] = {0};
#pragma omp parallel
    {
        u64 loc[16] = {0};
#pragma omp for schedule(dynamic, 1)
        for (long long a = 0; a < (long long)amax; a++) for (u64 b = 0; b < P; b++) {
            u64 acc[16] = {0};
            for (u64 c = 0; c < P; c++) for (u64 d = 0; d < P; d++) {
                u64 x[16], u[16];
                memcpy(x, base, sizeof(x)); x[0] = a; x[1] = b; x[2] = c; x[3] = d;
                for (int l = 1; l < layers; l++) {
                    for (int i = 0; i < 16; i++) x[i] = md(x[i] + P - rk[l][i]);
                    linv(x, u);
                    for (int blk = 0; blk < 4; blk++) sinv(u + 4 * blk, alpha);
                    memcpy(x, u, sizeof(x));
                }
                for (int i = 0; i < 16; i++) acc[i] += x[i];
            }
            for (int i = 0; i < 16; i++) loc[i] = md(loc[i] + md(acc[i]));
        }
#pragma omp critical
        for (int i = 0; i < 16; i++) sums[i] = md(sums[i] + loc[i]);
    }
    printf("direct p=%llu layers=%d seed=%u amax=%llu sums:", (unsigned long long)P, layers, seed, (unsigned long long)amax);
    for (int i = 0; i < 16; i++) printf(" %llu", (unsigned long long)sums[i]);
    printf("\n");
    return 0;
}

static int mode_projective(int argc, char **argv) {
    if (argc < 5) return 1;
    P = strtoull(argv[2], 0, 10); int layers = atoi(argv[3]); read_linv(argv[4]);
    u64 T = 4 * (P - 1);
    u64 sums[16] = {0}; long deg[16];
    { u64 y[4] = {1, 0, 0, 0}; tf st[16]; topforms(y, layers, st); for (int i = 0; i < 16; i++) deg[i] = st[i].d; }
#pragma omp parallel
    {
        u64 loc[16] = {0}; tf st[16]; u64 y[4];
#pragma omp for schedule(dynamic, 1)
        for (long long a = -1; a < (long long)P; a++) {       /* a = -1 encodes the (0,1,b,c) chart */
            u64 acc[16] = {0};
            for (u64 b = 0; b < P; b++) for (u64 c = 0; c < P; c++) {
                if (a < 0) { y[0] = 0; y[1] = 1; } else { y[0] = 1; y[1] = (u64)a; }
                y[2] = b; y[3] = c;
                topforms(y, layers, st);
                for (int i = 0; i < 16; i++) acc[i] += st[i].v;
            }
            for (int i = 0; i < 16; i++) loc[i] = md(loc[i] + md(acc[i]));
        }
#pragma omp critical
        for (int i = 0; i < 16; i++) sums[i] = md(sums[i] + loc[i]);
    }
    /* charts (0,0,1,c) and (0,0,0,1) */
    { tf st[16]; u64 y[4];
      for (u64 c = 0; c < P; c++) { y[0] = 0; y[1] = 0; y[2] = 1; y[3] = c; topforms(y, layers, st); for (int i = 0; i < 16; i++) sums[i] = md(sums[i] + st[i].v); }
      y[0] = 0; y[1] = 0; y[2] = 0; y[3] = 1; topforms(y, layers, st); for (int i = 0; i < 16; i++) sums[i] = md(sums[i] + st[i].v); }
    printf("projective p=%llu layers=%d T=%llu\nformal degrees:", (unsigned long long)P, layers, (unsigned long long)T);
    for (int i = 0; i < 16; i++) printf(" %ld", deg[i]);
    printf("\npredicted sums:");
    for (int i = 0; i < 16; i++) {
        if ((u64)deg[i] < T) printf(" 0");
        else if (deg[i] % (long)(P - 1) == 0) printf(" %llu", (unsigned long long)md(P - sums[i]));
        else printf(" ?");
    }
    printf("\n");
    return 0;
}

int main(int argc, char **argv) {
    int rc = 1;
    if (argc >= 2 && !strcmp(argv[1], "direct")) rc = mode_direct(argc, argv);
    else if (argc >= 2 && !strcmp(argv[1], "projective")) rc = mode_projective(argc, argv);
    if (rc) fprintf(stderr, "usage: %s direct <p> <layers> <seed> <linv.txt> [amax] | projective <p> <layers> <linv.txt>\n", argv[0]);
    return rc;
}
