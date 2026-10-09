/* zs.c -- fast zero-sum (higher-order differential) evaluation for DuX(2^n).
 *
 * Reads a binary job file produced by run_fast.py (which builds the round
 * keys, the ciphertext structure and the field log/antilog tables with the
 * Python reference implementation in dux/), enumerates the whole product
 * structure and prints, for every key and every S^{-1} layer, the XOR of the
 * 16 state words over the structure.
 *
 * The cipher is the DECRYPTION direction of DuX (paper Sect. 4, Algorithm 3
 * inverted); `layers` is the number of S^{-1} layers of dux.cipher.DuX
 * .decrypt_layers.  Field arithmetic uses the exact tables exported from
 * dux/field.py, so the C and Python results must agree bit for bit
 * (tests/test_fast_zs.py).
 *
 * Two code paths:
 *   generic  -- rebuild the ciphertext and run all `layers` layers;
 *   folded   -- when every active word sits in a different S-box block, the
 *               first S^{-1} layer and the following L^{-1} are precomputed
 *               per active word (table U), because block b of the layer-1
 *               output depends on a single active word and L^{-1} is linear.
 *               Layer 1 sums are then obtained in closed form (each state
 *               word is hit 2^{dim*(s-1)} times, so the XOR sum vanishes for
 *               s >= 2 and equals the per-word XOR for s == 1).
 * Both paths are covered by tests/test_fast_zs.py; set DUXZS_GENERIC=1 to
 * force the generic one.
 *
 * Build:  make            (gcc -O3 -march=native -fopenmp)
 * Run  :  ./zs job.bin  [> out.txt]
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define WORDS 16

static uint32_t N, POLY, ALPHA, ROUNDS, LAYERS, NACT, DIM, NKEYS;
static uint32_t ORD;                 /* 2^n - 1 */
static uint32_t ACTIVE[WORDS];
static uint16_t *LOGT;               /* size 1<<n            */
static uint16_t *EXPT;               /* size 2*(2^n-1)       */
static uint16_t *RK;                 /* NKEYS * (ROUNDS+1) * 16 */
static uint16_t *CONST;              /* NKEYS * 16           */
static uint16_t *VALS;               /* NKEYS * NACT * (1<<DIM) */

static const int ROTS[2][5] = {{1, 4, 8, 9, 13}, {1, 5, 8, 12, 13}};

/* branch-free: LOGT[0] == 0, so the index stays in range even for a == 0 */
static inline uint16_t gmul(uint16_t a, uint16_t b)
{
    uint32_t s = (uint32_t)LOGT[a] + (uint32_t)LOGT[b];
    uint16_t r = EXPT[s];
    return (a && b) ? r : 0;
}

/* S^{-1} = R^4, closed form (dux/sbox.py):
 *   f = x0*x1 + x3 + a ; g = x1*x2 + x0 + a
 *   y = (g, x2*f + x1 + a, f*g + x2 + a, f)                                  */
static inline void sinv(uint16_t *x, uint16_t a)
{
    uint16_t f = (uint16_t)(gmul(x[0], x[1]) ^ x[3] ^ a);
    uint16_t g = (uint16_t)(gmul(x[1], x[2]) ^ x[0] ^ a);
    uint16_t y1 = (uint16_t)(gmul(x[2], f) ^ x[1] ^ a);
    uint16_t y2 = (uint16_t)(gmul(f, g) ^ x[2] ^ a);
    x[0] = g; x[1] = y1; x[2] = y2; x[3] = f;
}

static inline void sl_inv(uint16_t *st, uint16_t a)
{
    sinv(st + 0, a); sinv(st + 4, a); sinv(st + 8, a); sinv(st + 12, a);
}

static inline void l_inv(uint16_t *st, int t)
{
    uint16_t o[WORDS];
    const int *r = ROTS[t];
    for (int i = 0; i < WORDS; i++)
        o[i] = (uint16_t)(st[(i + r[0]) & 15] ^ st[(i + r[1]) & 15] ^
                          st[(i + r[2]) & 15] ^ st[(i + r[3]) & 15] ^
                          st[(i + r[4]) & 15]);
    memcpy(st, o, sizeof(o));
}

static void rd(FILE *f, void *p, size_t n)
{
    if (fread(p, 1, n, f) != n) { fprintf(stderr, "short read\n"); exit(2); }
}

static void load_job(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    char magic[8];
    rd(f, magic, 8);
    if (memcmp(magic, "DUXZSJ01", 8)) { fprintf(stderr, "bad magic\n"); exit(2); }
    uint32_t h[8];
    rd(f, h, sizeof(h));
    N = h[0]; POLY = h[1]; ALPHA = h[2]; ROUNDS = h[3];
    LAYERS = h[4]; NACT = h[5]; DIM = h[6]; NKEYS = h[7];
    ORD = (1u << N) - 1u;
    rd(f, ACTIVE, NACT * sizeof(uint32_t));
    LOGT = malloc((size_t)(1u << N) * 2);
    EXPT = malloc((size_t)2 * ORD * 2);
    rd(f, LOGT, (size_t)(1u << N) * 2);
    rd(f, EXPT, (size_t)2 * ORD * 2);
    RK = malloc((size_t)NKEYS * (ROUNDS + 1) * WORDS * 2);
    CONST = malloc((size_t)NKEYS * WORDS * 2);
    VALS = malloc((size_t)NKEYS * NACT * ((size_t)1 << DIM) * 2);
    rd(f, RK, (size_t)NKEYS * (ROUNDS + 1) * WORDS * 2);
    rd(f, CONST, (size_t)NKEYS * WORDS * 2);
    rd(f, VALS, (size_t)NKEYS * NACT * ((size_t)1 << DIM) * 2);
    fclose(f);
}

/* ---------------------------------------------------------------- generic */
static void run_generic(uint16_t *acc, const unsigned char *TX)
{
    const uint16_t a = (uint16_t)ALPHA;
    const uint64_t total = (uint64_t)1 << ((uint64_t)DIM * NACT);
    const uint64_t mask = ((uint64_t)1 << DIM) - 1;
    const size_t accn = (size_t)NKEYS * LAYERS * WORDS;

#pragma omp parallel
    {
        uint16_t *loc = calloc(accn, 2);
        uint16_t st[WORDS];
#pragma omp for schedule(static)
        for (uint64_t idx = 0; idx < total; idx++) {
            for (uint32_t k = 0; k < NKEYS; k++) {
                const uint16_t *rk = RK + (size_t)k * (ROUNDS + 1) * WORDS;
                const unsigned char *tx = TX + (size_t)k * (ROUNDS + 1);
                memcpy(st, CONST + (size_t)k * WORDS, sizeof(st));
                for (uint32_t j = 0; j < NACT; j++) {
                    uint64_t sub = (idx >> ((uint64_t)DIM * j)) & mask;
                    st[ACTIVE[j]] = VALS[((size_t)k * NACT + j) * ((size_t)1 << DIM) + sub];
                }
                const uint16_t *rkr = rk + (size_t)ROUNDS * WORDS;
                for (int i = 0; i < WORDS; i++) st[i] ^= rkr[i];
                sl_inv(st, a);
                uint16_t *p = loc + (size_t)k * LAYERS * WORDS;
                for (int i = 0; i < WORDS; i++) p[i] ^= st[i];
                for (uint32_t l = 1; l < LAYERS; l++) {
                    uint32_t ri = ROUNDS - l;
                    const uint16_t *rki = rk + (size_t)ri * WORDS;
                    for (int i = 0; i < WORDS; i++) st[i] ^= rki[i];
                    l_inv(st, tx[ri]);
                    sl_inv(st, a);
                    uint16_t *q = loc + ((size_t)k * LAYERS + l) * WORDS;
                    for (int i = 0; i < WORDS; i++) q[i] ^= st[i];
                }
            }
        }
#pragma omp critical
        for (size_t i = 0; i < accn; i++) acc[i] ^= loc[i];
        free(loc);
    }
}

/* ----------------------------------------------------------------- folded */
/* U[k][j][v][0..15] : L^{-1}_{t(rk^{r-1})} applied to the layer-1 output of
 * the block of active word j (other blocks zeroed).  BASE[k][0..15] holds the
 * contribution of the constant blocks plus L^{-1}(rk^{r-1}).                */
static uint16_t *U, *BASE;

static void build_folded(const unsigned char *TX)
{
    const uint16_t a = (uint16_t)ALPHA;
    const size_t nv = (size_t)1 << DIM;
    U = malloc((size_t)NKEYS * NACT * nv * WORDS * 2);
    BASE = calloc((size_t)NKEYS * WORDS, 2);
    for (uint32_t k = 0; k < NKEYS; k++) {
        const uint16_t *rk = RK + (size_t)k * (ROUNDS + 1) * WORDS;
        const uint16_t *rkr = rk + (size_t)ROUNDS * WORDS;
        const uint16_t *rk1 = rk + (size_t)(ROUNDS - 1) * WORDS;
        int t1 = TX[(size_t)k * (ROUNDS + 1) + (ROUNDS - 1)];
        int is_active_block[4] = {0, 0, 0, 0};
        for (uint32_t j = 0; j < NACT; j++) is_active_block[ACTIVE[j] / 4] = 1;

        /* constant blocks + round key rk^{r-1}, all pushed through L^{-1} */
        uint16_t z[WORDS];
        memset(z, 0, sizeof(z));
        for (int b = 0; b < 4; b++) {
            if (is_active_block[b]) continue;
            uint16_t blk[4];
            for (int i = 0; i < 4; i++)
                blk[i] = (uint16_t)(CONST[(size_t)k * WORDS + 4 * b + i] ^ rkr[4 * b + i]);
            sinv(blk, a);
            for (int i = 0; i < 4; i++) z[4 * b + i] = blk[i];
        }
        for (int i = 0; i < WORDS; i++) z[i] ^= rk1[i];
        l_inv(z, t1);
        memcpy(BASE + (size_t)k * WORDS, z, sizeof(z));

        for (uint32_t j = 0; j < NACT; j++) {
            uint32_t w = ACTIVE[j], b = w / 4, p = w % 4;
            for (size_t v = 0; v < nv; v++) {
                uint16_t blk[4];
                for (int i = 0; i < 4; i++)
                    blk[i] = (uint16_t)(CONST[(size_t)k * WORDS + 4 * b + i] ^ rkr[4 * b + i]);
                blk[p] = (uint16_t)(VALS[((size_t)k * NACT + j) * nv + v] ^ rkr[w]);
                sinv(blk, a);
                uint16_t y[WORDS];
                memset(y, 0, sizeof(y));
                for (int i = 0; i < 4; i++) y[4 * b + i] = blk[i];
                l_inv(y, t1);
                memcpy(U + (((size_t)k * NACT + j) * nv + v) * WORDS, y, sizeof(y));
            }
        }
    }
}

/* exact layer-1 XOR sums, in closed form */
static void layer1_sums(uint16_t *acc)
{
    const uint16_t a = (uint16_t)ALPHA;
    const size_t nv = (size_t)1 << DIM;
    if (NACT != 1) return;                         /* multiplicity even -> 0 */
    for (uint32_t k = 0; k < NKEYS; k++) {
        uint16_t *p = acc + (size_t)k * LAYERS * WORDS;
        const uint16_t *rk = RK + (size_t)k * (ROUNDS + 1) * WORDS;
        const uint16_t *rkr = rk + (size_t)ROUNDS * WORDS;
        uint32_t w = ACTIVE[0], b = w / 4, q = w % 4;
        for (size_t v = 0; v < nv; v++) {
            uint16_t blk[4];
            for (int i = 0; i < 4; i++)
                blk[i] = (uint16_t)(CONST[(size_t)k * WORDS + 4 * b + i] ^ rkr[4 * b + i]);
            blk[q] = (uint16_t)(VALS[(size_t)k * NACT * nv + v] ^ rkr[w]);
            sinv(blk, a);
            for (int i = 0; i < 4; i++) p[4 * b + i] ^= blk[i];
        }
        /* constant blocks: hit 2^dim times, even -> 0 */
    }
}

static void run_folded(uint16_t *acc, const unsigned char *TX)
{
    const uint16_t a = (uint16_t)ALPHA;
    const uint64_t total = (uint64_t)1 << ((uint64_t)DIM * NACT);
    const uint64_t mask = ((uint64_t)1 << DIM) - 1;
    const size_t accn = (size_t)NKEYS * LAYERS * WORDS;
    const size_t nv = (size_t)1 << DIM;

    build_folded(TX);
    layer1_sums(acc);

#pragma omp parallel
    {
        uint16_t *loc = calloc(accn, 2);
        uint16_t st[WORDS];
#pragma omp for schedule(static)
        for (uint64_t idx = 0; idx < total; idx++) {
            for (uint32_t k = 0; k < NKEYS; k++) {
                const uint16_t *rk = RK + (size_t)k * (ROUNDS + 1) * WORDS;
                const unsigned char *tx = TX + (size_t)k * (ROUNDS + 1);
                memcpy(st, BASE + (size_t)k * WORDS, sizeof(st));
                for (uint32_t j = 0; j < NACT; j++) {
                    uint64_t sub = (idx >> ((uint64_t)DIM * j)) & mask;
                    const uint16_t *u = U + (((size_t)k * NACT + j) * nv + sub) * WORDS;
                    for (int i = 0; i < WORDS; i++) st[i] ^= u[i];
                }
                sl_inv(st, a);                       /* this is layer 2 */
                uint16_t *p = loc + ((size_t)k * LAYERS + 1) * WORDS;
                for (int i = 0; i < WORDS; i++) p[i] ^= st[i];
                for (uint32_t l = 2; l < LAYERS; l++) {
                    uint32_t ri = ROUNDS - l;
                    const uint16_t *rki = rk + (size_t)ri * WORDS;
                    for (int i = 0; i < WORDS; i++) st[i] ^= rki[i];
                    l_inv(st, tx[ri]);
                    sl_inv(st, a);
                    uint16_t *q = loc + ((size_t)k * LAYERS + l) * WORDS;
                    for (int i = 0; i < WORDS; i++) q[i] ^= st[i];
                }
            }
        }
#pragma omp critical
        for (size_t i = 0; i < accn; i++) acc[i] ^= loc[i];
        free(loc);
    }
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s job.bin\n", argv[0]); return 1; }
    load_job(argv[1]);

    (void)POLY;
    const size_t accn = (size_t)NKEYS * LAYERS * WORDS;
    uint16_t *acc = calloc(accn, 2);

    unsigned char *TX = malloc((size_t)NKEYS * (ROUNDS + 1));
    for (uint32_t k = 0; k < NKEYS; k++)
        for (uint32_t i = 0; i <= ROUNDS; i++) {
            uint16_t w = RK[((size_t)k * (ROUNDS + 1) + i) * WORDS + 15];
            TX[(size_t)k * (ROUNDS + 1) + i] = (unsigned char)((w & 1u) ^ ((w >> 1) & 1u));
        }

    int blocks_distinct = 1;
    int seen[4] = {0, 0, 0, 0};
    for (uint32_t j = 0; j < NACT; j++) {
        if (seen[ACTIVE[j] / 4]) blocks_distinct = 0;
        seen[ACTIVE[j] / 4] = 1;
    }
    const char *force = getenv("DUXZS_GENERIC");
    int folded = blocks_distinct && LAYERS >= 2 && !(force && force[0] == '1');

    if (folded) run_folded(acc, TX);
    else        run_generic(acc, TX);

    printf("# DUXZS n=%u rounds=%u layers=%u nactive=%u dim=%u keys=%u "
           "data_log2=%u path=%s\n",
           N, ROUNDS, LAYERS, NACT, DIM, NKEYS, DIM * NACT,
           folded ? "folded" : "generic");
    for (uint32_t k = 0; k < NKEYS; k++)
        for (uint32_t l = 0; l < LAYERS; l++) {
            printf("S %u %u", k, l + 1);
            for (int i = 0; i < WORDS; i++)
                printf(" %u", acc[((size_t)k * LAYERS + l) * WORDS + i]);
            printf("\n");
        }
    return 0;
}
