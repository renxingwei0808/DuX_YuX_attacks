/* mom.c -- plaintext moments of a chosen-ciphertext structure for DuX(2^n).
 *
 * The r_KR = 1 key recovery from a PARTIAL zero-sum
 * (experiments/E06_key_recovery_1round/attack_1round_partial.py) needs, for
 * every structure and every S-box block j, the moments
 *
 *      M_j[e] = sum_{C in structure}  prod_{i<4} P_{4j+i}^{e_i},
 *      P = Dec_r(C)  (full r-round decryption)
 *
 * over a small list of exponent vectors e.  In numpy this costs hours per
 * 2^32-ciphertext structure; here it is a flat OpenMP loop.
 *
 * The job file is written by run_moments.py, which also supplies the field
 * log/antilog tables exported from dux/field.py, so the arithmetic is
 * bit-identical to the Python reference (tests/test_fast_zs.py::test_moments).
 *
 * Build:  make            (gcc -O3 -march=native -fopenmp)
 * Run  :  ./mom job.bin
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define WORDS 16

static uint32_t N, POLY, ALPHA, ROUNDS, NACT, DIM, NSTRUCT, NMOM, MAXE;
static uint32_t ORD;
static uint32_t ACTIVE[WORDS];
static uint16_t *LOGT, *EXPT, *RK, *CONST, *VALS;
static unsigned char *MOMEXP;          /* NMOM * 4 */
static unsigned char TX[64];

static const int ROTS[2][5] = {{1, 4, 8, 9, 13}, {1, 5, 8, 12, 13}};

static inline uint16_t gmul(uint16_t a, uint16_t b)
{
    uint32_t s = (uint32_t)LOGT[a] + (uint32_t)LOGT[b];
    uint16_t r = EXPT[s];
    return (a && b) ? r : 0;
}

/* S = (R^{-1})^4, closed form (dux/sbox.py S_closed) -- encryption direction */
static inline void s_enc(uint16_t *x, uint16_t al)
{
    uint16_t a = (uint16_t)(x[2] ^ gmul(x[0], x[3]) ^ al);
    uint16_t b = (uint16_t)(x[1] ^ gmul(x[3], a) ^ al);
    uint16_t c = (uint16_t)(x[0] ^ gmul(a, b) ^ al);
    uint16_t y3 = (uint16_t)(x[3] ^ gmul(b, c) ^ al);
    x[0] = c; x[1] = b; x[2] = a; x[3] = y3;
}

/* S^{-1} = R^4 -- decryption direction */
static inline void sinv(uint16_t *x, uint16_t al)
{
    uint16_t f = (uint16_t)(gmul(x[0], x[1]) ^ x[3] ^ al);
    uint16_t g = (uint16_t)(gmul(x[1], x[2]) ^ x[0] ^ al);
    uint16_t y1 = (uint16_t)(gmul(x[2], f) ^ x[1] ^ al);
    uint16_t y2 = (uint16_t)(gmul(f, g) ^ x[2] ^ al);
    x[0] = g; x[1] = y1; x[2] = y2; x[3] = f;
}

static inline void sl_inv(uint16_t *st, uint16_t al)
{
    sinv(st + 0, al); sinv(st + 4, al); sinv(st + 8, al); sinv(st + 12, al);
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
    if (memcmp(magic, "DUXMOMJ1", 8)) { fprintf(stderr, "bad magic\n"); exit(2); }
    uint32_t h[9];
    rd(f, h, sizeof(h));
    N = h[0]; POLY = h[1]; ALPHA = h[2]; ROUNDS = h[3];
    NACT = h[4]; DIM = h[5]; NSTRUCT = h[6]; NMOM = h[7]; MAXE = h[8];
    ORD = (1u << N) - 1u;
    rd(f, ACTIVE, NACT * sizeof(uint32_t));
    LOGT = malloc((size_t)(1u << N) * 2);
    EXPT = malloc((size_t)2 * ORD * 2);
    rd(f, LOGT, (size_t)(1u << N) * 2);
    rd(f, EXPT, (size_t)2 * ORD * 2);
    RK = malloc((size_t)(ROUNDS + 1) * WORDS * 2);
    rd(f, RK, (size_t)(ROUNDS + 1) * WORDS * 2);
    CONST = malloc((size_t)NSTRUCT * WORDS * 2);
    rd(f, CONST, (size_t)NSTRUCT * WORDS * 2);
    VALS = malloc((size_t)NSTRUCT * NACT * ((size_t)1 << DIM) * 2);
    rd(f, VALS, (size_t)NSTRUCT * NACT * ((size_t)1 << DIM) * 2);
    MOMEXP = malloc((size_t)NMOM * 4);
    rd(f, MOMEXP, (size_t)NMOM * 4);
    fclose(f);
    (void)POLY;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s job.bin\n", argv[0]); return 1; }
    load_job(argv[1]);
    if (MAXE > 16) { fprintf(stderr, "MAXE too large\n"); return 2; }
    const uint16_t al = (uint16_t)ALPHA;
    const uint64_t total = (uint64_t)1 << ((uint64_t)DIM * NACT);
    const uint64_t mask = ((uint64_t)1 << DIM) - 1;
    for (uint32_t i = 0; i <= ROUNDS; i++) {
        uint16_t w = RK[(size_t)i * WORDS + 15];
        TX[i] = (unsigned char)((w & 1u) ^ ((w >> 1) & 1u));
    }
    const size_t accn = (size_t)4 * NMOM;
    uint16_t *out = calloc((size_t)NSTRUCT * accn, 2);

    for (uint32_t st = 0; st < NSTRUCT; st++) {
        uint16_t *res = out + (size_t)st * accn;
#pragma omp parallel
        {
            uint16_t *loc = calloc(accn, 2);
            uint16_t s[WORDS];
            uint16_t pw[4][17];
#pragma omp for schedule(static)
            for (uint64_t idx = 0; idx < total; idx++) {
                memcpy(s, CONST + (size_t)st * WORDS, sizeof(s));
                for (uint32_t j = 0; j < NACT; j++) {
                    uint64_t sub = (idx >> ((uint64_t)DIM * j)) & mask;
                    s[ACTIVE[j]] = VALS[((size_t)st * NACT + j) * ((size_t)1 << DIM) + sub];
                }
                /* full r-round decryption */
                const uint16_t *rkr = RK + (size_t)ROUNDS * WORDS;
                for (int i = 0; i < WORDS; i++) s[i] ^= rkr[i];
                sl_inv(s, al);
                for (uint32_t r = ROUNDS - 1; r >= 1; r--) {
                    const uint16_t *rki = RK + (size_t)r * WORDS;
                    for (int i = 0; i < WORDS; i++) s[i] ^= rki[i];
                    l_inv(s, TX[r]);
                    sl_inv(s, al);
                }
                for (int i = 0; i < WORDS; i++) s[i] ^= RK[i];
                /* moments, per block */
                for (int b = 0; b < 4; b++) {
                    for (int i = 0; i < 4; i++) {
                        pw[i][0] = 1;
                        for (uint32_t e = 1; e <= MAXE; e++)
                            pw[i][e] = gmul(pw[i][e - 1], s[4 * b + i]);
                    }
                    for (uint32_t m = 0; m < NMOM; m++) {
                        const unsigned char *e = MOMEXP + 4 * (size_t)m;
                        uint16_t v = 1;
                        for (int i = 0; i < 4; i++)
                            if (e[i]) v = gmul(v, pw[i][e[i]]);
                        loc[(size_t)b * NMOM + m] ^= v;
                    }
                }
            }
#pragma omp critical
            for (size_t i = 0; i < accn; i++) res[i] ^= loc[i];
            free(loc);
        }
        fprintf(stderr, "structure %u/%u done\n", st + 1, NSTRUCT);
    }

    printf("# DUXMOM n=%u rounds=%u nactive=%u dim=%u structures=%u moments=%u\n",
           N, ROUNDS, NACT, DIM, NSTRUCT, NMOM);
    for (uint32_t st = 0; st < NSTRUCT; st++)
        for (int b = 0; b < 4; b++) {
            printf("M %u %d", st, b);
            for (uint32_t m = 0; m < NMOM; m++)
                printf(" %u", out[(size_t)st * accn + (size_t)b * NMOM + m]);
            printf("\n");
        }
    /* silence unused-function warning when s_enc is not needed */
    (void)s_enc;
    return 0;
}
