/* zsx.c -- generic zero-sum kernel for DuX (and, with -DCIPHER_YUX, YuX):
 * both fields, both directions.
 *
 * experiments/E04_zero_sum_F2n/fast/zs.c covers exactly one case (F_{2^n},
 * decryption, one affine subspace per active word).  S6 needs three more:
 * F_p full-field multi-word structures, F_p signed subgroup-coset structures,
 * and the ENCRYPTION direction (the designer's CPA model, E10).  Rather than
 * four kernels this one is driven by a job file that describes the structure
 * explicitly:
 *
 *   every active ciphertext (resp. plaintext) word j carries a list of
 *   SIZES[j] field values together with a sign in {+1, -1}; the structure is
 *   the product set and the weight of a point is the product of its signs.
 *
 * That covers all three thresholds of theorem O7 uniformly:
 *   full field / affine subspace : all signs +1
 *   signed 2^s-coset difference  : each word is  coset_0 (+1) u coset_1 (-1).
 *
 * Arithmetic: F_{2^n} through the log/antilog tables exported from
 * dux/field.py; F_p through Barrett reduction (p < 2^17, so every product is
 * below 2^34 and one 64-bit multiply-shift suffices).
 *
 * Cipher selection (S11).  The job-file format is cipher-independent (the F_p
 * forward circulant row is read from the job, the log/antilog tables too), so
 * the only cipher-specific things compiled in are the S-box closed forms and
 * the two rotation sets.  `make` builds BOTH binaries:
 *
 *   zsx       DuX   (Wu et al.)   S^{-1} = R^4,   ROT_INV = {1,4,8,9,13} (t=0)
 *                                                        {1,5,8,12,13} (t=1)
 *   zsx_yux   YuX   (Liu et al.)  S^{-1} = Pf^4,  ROT_INV = {0,3,4,8,9,12,14}
 *
 * YuX has NO key-dependent linear layer (yux/linear.py: `t` is accepted and
 * ignored), so both t slots hold the same rotation set and the per-round t bit
 * is forced to 0.
 *
 * Build:  make
 * Run  :  ./zsx job.bin   /   ./zsx_yux job.bin
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define WORDS 16

static uint32_t CHAR2, NBITS, P, ALPHA, ROUNDS, LAYERS, NACT, NKEYS, DIRECTION;
static uint32_t ORD;                       /* 2^n - 1                        */
static uint64_t PMAGIC;                    /* floor(2^40 / p)                */
static uint32_t ACTIVE[WORDS], SIZES[WORDS], OFFS[WORDS];
static uint32_t VTOT;                      /* sum of SIZES                   */
static uint16_t *LOGT, *EXPT;
static uint32_t FWD[2][WORDS];             /* F_p forward circulant rows     */
static uint32_t *RK, *CONSTW, *VALS;
static int8_t *SGNS;

#ifdef CIPHER_YUX
/* yux/params.py: ROT_INV (7 terms, both fields) and ROT_FWD_BIN (11 terms). */
#define CIPHER_NAME "YUX"
#define NROT_INV 7
static const int ROTS_INV[2][NROT_INV] = {{0, 3, 4, 8, 9, 12, 14},
                                          {0, 3, 4, 8, 9, 12, 14}};
static const int ROTS_FWD2[2][11] = {{1, 2, 3, 5, 6, 7, 8, 12, 13, 14, 15},
                                     {1, 2, 3, 5, 6, 7, 8, 12, 13, 14, 15}};
#else
#define CIPHER_NAME "DUX"
#define NROT_INV 5
static const int ROTS_INV[2][NROT_INV] = {{1, 4, 8, 9, 13}, {1, 5, 8, 12, 13}};
static const int ROTS_FWD2[2][11] = {{0, 1, 3, 4, 7, 8, 9, 10, 12, 14, 15},
                                     {0, 3, 4, 5, 6, 8, 10, 11, 12, 13, 15}};
#endif

/* ------------------------------------------------------------ arithmetic */
static inline uint32_t fadd(uint32_t a, uint32_t b)
{
    if (CHAR2) return a ^ b;
    uint32_t r = a + b;
    return r >= P ? r - P : r;
}

static inline uint32_t fsub(uint32_t a, uint32_t b)
{
    if (CHAR2) return a ^ b;
    return a >= b ? a - b : a + P - b;
}

static inline uint32_t fmul(uint32_t a, uint32_t b)
{
    if (CHAR2) {
        uint32_t s = (uint32_t)LOGT[a] + (uint32_t)LOGT[b];
        uint32_t r = EXPT[s];
        return (a && b) ? r : 0;
    }
    uint64_t prod = (uint64_t)a * (uint64_t)b;
    uint64_t q = (prod * PMAGIC) >> 40;
    uint64_t r = prod - q * (uint64_t)P;
    while (r >= (uint64_t)P) r -= (uint64_t)P;
    return (uint32_t)r;
}

#ifdef CIPHER_YUX
/* S^{-1} = Pf^4, closed form (yux/sbox.py S_inv_closed); degrees (2,2,3,4) */
static inline void sinv(uint32_t *x, uint32_t a)
{
    uint32_t y0 = fadd(fadd(fadd(x[0], fmul(x[1], x[2])), x[3]), a);
    uint32_t y1 = fadd(fadd(fadd(x[1], fmul(x[2], x[3])), y0), a);
    uint32_t y2 = fadd(fadd(fadd(x[2], fmul(x[3], y0)), y1), a);
    uint32_t y3 = fadd(fadd(fadd(x[3], fmul(y0, y1)), y2), a);
    x[0] = y0; x[1] = y1; x[2] = y2; x[3] = y3;
}

/* S = Pf^{-4}, closed form (yux/sbox.py S_closed): out = (z3, z2, z1, z0),
 * degrees (8,5,3,2) at positions 0..3 */
static inline void sfwd(uint32_t *x, uint32_t a)
{
    uint32_t z0 = fsub(fsub(fsub(x[3], fmul(x[0], x[1])), x[2]), a);
    uint32_t z1 = fsub(fsub(fsub(x[2], fmul(z0, x[0])), x[1]), a);
    uint32_t z2 = fsub(fsub(fsub(x[1], fmul(z1, z0)), x[0]), a);
    uint32_t z3 = fsub(fsub(fsub(x[0], fmul(z2, z1)), z0), a);
    x[0] = z3; x[1] = z2; x[2] = z1; x[3] = z0;
}
#else
/* S^{-1} = R^4, closed form (dux/sbox.py) */
static inline void sinv(uint32_t *x, uint32_t a)
{
    uint32_t f = fadd(fadd(fmul(x[0], x[1]), x[3]), a);
    uint32_t g = fadd(fadd(fmul(x[1], x[2]), x[0]), a);
    uint32_t y1 = fadd(fadd(fmul(x[2], f), x[1]), a);
    uint32_t y2 = fadd(fadd(fmul(f, g), x[2]), a);
    x[0] = g; x[1] = y1; x[2] = y2; x[3] = f;
}

/* S = (R^{-1})^4, closed form (dux/sbox.py): out = (c, b, a2, y3) */
static inline void sfwd(uint32_t *x, uint32_t a)
{
    uint32_t a2 = fsub(fsub(x[2], fmul(x[0], x[3])), a);
    uint32_t b  = fsub(fsub(x[1], fmul(x[3], a2)), a);
    uint32_t c  = fsub(fsub(x[0], fmul(a2, b)), a);
    uint32_t y3 = fsub(fsub(x[3], fmul(b, c)), a);
    x[0] = c; x[1] = b; x[2] = a2; x[3] = y3;
}
#endif

static inline void sl_inv(uint32_t *st, uint32_t a)
{
    sinv(st + 0, a); sinv(st + 4, a); sinv(st + 8, a); sinv(st + 12, a);
}

static inline void sl_fwd(uint32_t *st, uint32_t a)
{
    sfwd(st + 0, a); sfwd(st + 4, a); sfwd(st + 8, a); sfwd(st + 12, a);
}

static inline void l_inv(uint32_t *st, int t)
{
    uint32_t o[WORDS];
    const int *r = ROTS_INV[t];
    for (int i = 0; i < WORDS; i++) {
        uint32_t acc = st[(i + r[0]) & 15];
        for (int j = 1; j < NROT_INV; j++) acc = fadd(acc, st[(i + r[j]) & 15]);
        o[i] = acc;
    }
    memcpy(st, o, sizeof(o));
}

static inline void l_fwd(uint32_t *st, int t)
{
    uint32_t o[WORDS];
    if (CHAR2) {
        const int *r = ROTS_FWD2[t];
        for (int i = 0; i < WORDS; i++) {
            uint32_t acc = 0;
            for (int j = 0; j < 11; j++) acc ^= st[(i + r[j]) & 15];
            o[i] = acc;
        }
    } else {
        const uint32_t *row = FWD[t];
        for (int i = 0; i < WORDS; i++) {
            uint32_t acc = 0;
            for (int j = 0; j < WORDS; j++)
                if (row[j]) acc = fadd(acc, fmul(st[(i + j) & 15], row[j]));
            o[i] = acc;
        }
    }
    memcpy(st, o, sizeof(o));
}

/* ------------------------------------------------------------------- job */
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
    if (memcmp(magic, "DUXZSX01", 8)) { fprintf(stderr, "bad magic\n"); exit(2); }
    uint32_t h[9];
    rd(f, h, sizeof(h));
    CHAR2 = h[0]; NBITS = h[1]; P = h[2]; ALPHA = h[3]; ROUNDS = h[4];
    LAYERS = h[5]; NACT = h[6]; NKEYS = h[7]; DIRECTION = h[8];
    ORD = CHAR2 ? (1u << NBITS) - 1u : 0u;
    PMAGIC = CHAR2 ? 0 : ((uint64_t)1 << 40) / (uint64_t)P;
    rd(f, ACTIVE, NACT * sizeof(uint32_t));
    rd(f, SIZES, NACT * sizeof(uint32_t));
    VTOT = 0;
    for (uint32_t j = 0; j < NACT; j++) { OFFS[j] = VTOT; VTOT += SIZES[j]; }
    if (CHAR2) {
        LOGT = malloc((size_t)(1u << NBITS) * 2);
        EXPT = malloc((size_t)2 * ORD * 2);
        rd(f, LOGT, (size_t)(1u << NBITS) * 2);
        rd(f, EXPT, (size_t)2 * ORD * 2);
    }
    rd(f, FWD, sizeof(FWD));
    RK = malloc((size_t)NKEYS * (ROUNDS + 1) * WORDS * 4);
    CONSTW = malloc((size_t)NKEYS * WORDS * 4);
    VALS = malloc((size_t)NKEYS * VTOT * 4);
    SGNS = malloc((size_t)NKEYS * VTOT);
    rd(f, RK, (size_t)NKEYS * (ROUNDS + 1) * WORDS * 4);
    rd(f, CONSTW, (size_t)NKEYS * WORDS * 4);
    rd(f, VALS, (size_t)NKEYS * VTOT * 4);
    rd(f, SGNS, (size_t)NKEYS * VTOT);
    fclose(f);
}

/* ------------------------------------------------------------------- run */
int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s job.bin\n", argv[0]); return 1; }
    load_job(argv[1]);

    unsigned char *TX = malloc((size_t)NKEYS * (ROUNDS + 1));
    for (uint32_t k = 0; k < NKEYS; k++)
        for (uint32_t i = 0; i <= ROUNDS; i++) {
#ifdef CIPHER_YUX
            /* YuX has no key-dependent linear layer (yux/linear.py). */
            TX[(size_t)k * (ROUNDS + 1) + i] = 0;
#else
            uint32_t w = RK[((size_t)k * (ROUNDS + 1) + i) * WORDS + 15];
            TX[(size_t)k * (ROUNDS + 1) + i] = (unsigned char)((w & 1u) ^ ((w >> 1) & 1u));
#endif
        }

    uint64_t total = 1;
    for (uint32_t j = 0; j < NACT; j++) total *= (uint64_t)SIZES[j];
    const size_t accn = (size_t)NKEYS * LAYERS * WORDS;
    int64_t *acc = calloc(accn, sizeof(int64_t));
    const uint32_t a = ALPHA;

#pragma omp parallel
    {
        int64_t *loc = calloc(accn, sizeof(int64_t));
        uint32_t st[WORDS];
        uint32_t sub[WORDS];
        /* The product set is walked as an odometer so that no integer
         * division appears in the hot loop: the outermost active word is the
         * OpenMP loop variable and the remaining ones are carried by hand. */
        uint64_t rest = 1;
        for (uint32_t j = 1; j < NACT; j++) rest *= (uint64_t)SIZES[j];
#pragma omp for schedule(static)
        for (uint32_t i0 = 0; i0 < SIZES[0]; i0++) {
            sub[0] = i0;
            for (uint32_t j = 1; j < NACT; j++) sub[j] = 0;
            for (uint64_t cnt = 0; cnt < rest; cnt++) {
                for (uint32_t k = 0; k < NKEYS; k++) {
                    const uint32_t *rk = RK + (size_t)k * (ROUNDS + 1) * WORDS;
                    const unsigned char *tx = TX + (size_t)k * (ROUNDS + 1);
                    const uint32_t *vk = VALS + (size_t)k * VTOT;
                    const int8_t *sk = SGNS + (size_t)k * VTOT;
                    memcpy(st, CONSTW + (size_t)k * WORDS, sizeof(st));
                    int64_t w = 1;
                    for (uint32_t j = 0; j < NACT; j++) {
                        st[ACTIVE[j]] = vk[OFFS[j] + sub[j]];
                        w *= (int64_t)sk[OFFS[j] + sub[j]];
                    }
                    int64_t *base = loc + (size_t)k * LAYERS * WORDS;
                    if (DIRECTION == 0) {                    /* decryption */
                        const uint32_t *rkr = rk + (size_t)ROUNDS * WORDS;
                        for (int i = 0; i < WORDS; i++) st[i] = fsub(st[i], rkr[i]);
                        sl_inv(st, a);
                        if (CHAR2) for (int i = 0; i < WORDS; i++) base[i] ^= (int64_t)st[i];
                        else       for (int i = 0; i < WORDS; i++) base[i] += w * (int64_t)st[i];
                        for (uint32_t l = 1; l < LAYERS; l++) {
                            uint32_t ri = ROUNDS - l;
                            const uint32_t *rki = rk + (size_t)ri * WORDS;
                            for (int i = 0; i < WORDS; i++) st[i] = fsub(st[i], rki[i]);
                            l_inv(st, tx[ri]);
                            sl_inv(st, a);
                            int64_t *q = base + (size_t)l * WORDS;
                            if (CHAR2) for (int i = 0; i < WORDS; i++) q[i] ^= (int64_t)st[i];
                            else       for (int i = 0; i < WORDS; i++) q[i] += w * (int64_t)st[i];
                        }
                    } else {                                 /* encryption */
                        for (int i = 0; i < WORDS; i++) st[i] = fadd(st[i], rk[i]);
                        sl_fwd(st, a);
                        if (CHAR2) for (int i = 0; i < WORDS; i++) base[i] ^= (int64_t)st[i];
                        else       for (int i = 0; i < WORDS; i++) base[i] += w * (int64_t)st[i];
                        for (uint32_t l = 1; l < LAYERS; l++) {
                            const uint32_t *rki = rk + (size_t)l * WORDS;
                            l_fwd(st, tx[l]);
                            for (int i = 0; i < WORDS; i++) st[i] = fadd(st[i], rki[i]);
                            sl_fwd(st, a);
                            int64_t *q = base + (size_t)l * WORDS;
                            if (CHAR2) for (int i = 0; i < WORDS; i++) q[i] ^= (int64_t)st[i];
                            else       for (int i = 0; i < WORDS; i++) q[i] += w * (int64_t)st[i];
                        }
                    }
                }
                for (int32_t j = (int32_t)NACT - 1; j >= 1; j--) {
                    if (++sub[j] < SIZES[j]) break;
                    sub[j] = 0;
                }
            }
        }
#pragma omp critical
        {
            if (CHAR2) for (size_t i = 0; i < accn; i++) acc[i] ^= loc[i];
            else       for (size_t i = 0; i < accn; i++) acc[i] += loc[i];
        }
        free(loc);
    }

    printf("# " CIPHER_NAME "ZSX char2=%u n=%u p=%u rounds=%u layers=%u nactive=%u keys=%u "
           "dir=%s points=%llu\n",
           CHAR2, NBITS, P, ROUNDS, LAYERS, NACT, NKEYS,
           DIRECTION ? "enc" : "dec", (unsigned long long)total);
    for (uint32_t k = 0; k < NKEYS; k++)
        for (uint32_t l = 0; l < LAYERS; l++) {
            printf("S %u %u", k, l + 1);
            for (int i = 0; i < WORDS; i++) {
                int64_t v = acc[((size_t)k * LAYERS + l) * WORDS + i];
                if (!CHAR2) { v %= (int64_t)P; if (v < 0) v += (int64_t)P; }
                printf(" %lld", (long long)v);
            }
            printf("\n");
        }
    return 0;
}
