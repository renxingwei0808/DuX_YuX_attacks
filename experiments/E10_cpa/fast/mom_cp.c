/* mom_cp.c -- CIPHERTEXT moments of a chosen-PLAINTEXT structure for DuX(2^n).
 *
 * Mirror image of experiments/E06_key_recovery_1round/fast/mom.c.  There the
 * structure lives on the ciphertext side and the kernel decrypts; here the
 * structure lives on the plaintext side (designer CPA model, experiments/E10_cpa)
 * and the kernel ENCRYPTS.  For every structure and every S-box block b it
 * accumulates
 *
 *      M_b[e] = sum_{P in structure}  prod_{i<4} C_{4b+i}^{e_i},
 *      C = Enc_r(P)  (full r-round encryption)
 *
 * over the list of exponent vectors that the last-round equation of
 * experiments/E10_cpa/attack_cp_lastround.py needs (17 of them for the
 * single-position `--coords 2` system, max exponent 2, total degree <= 4).
 *
 * The 8-round DuX(2^16) attack of S8 needs 20 structures of 2^32 chosen
 * plaintexts per key; the numpy path of attack_cp_lastround.py would take
 * weeks, this takes minutes per structure.
 *
 * The job file is written by attack_cp_lastround_fast.py, which also supplies
 * the field log/antilog tables exported from dux/field.py, so the arithmetic is
 * bit-identical to the Python reference (tests/test_mom_cp.py, and the
 * --verify flag of the driver).
 *
 * Build:  make            (gcc -O3 -march=native -fopenmp)
 * Run  :  ./mom_cp job.bin
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define WORDS 16
/* Points per batch.  The state is held transposed (one array of VB points per
 * state word) so that the linear layer -- 57% of the scalar run time, and pure
 * XOR -- becomes vector code; the S-box keeps its scalar table lookups.  32
 * uint16 = one 64-byte cache line / two AVX2 registers per word. */
#define VB 128

static uint32_t N, POLY, ALPHA, ROUNDS, NACT, DIM, NSTRUCT, NMOM, MAXE;
static uint32_t ORD;
static uint32_t ACTIVE[WORDS];
static uint16_t *LOGT, *EXPT, *RK, *CONST, *VALS;
static unsigned char *MOMEXP;          /* NMOM * 4 */
static unsigned char TX[64];

/* Forward linear layer over F_{2^n}.  The paper's tap sets (dux/params.py
 * ROT_FWD_BIN) have 11 of the 16 offsets, so it is cheaper to XOR the five
 * MISSING ones into the total: over F_2 and with an even number of words,
 *      T = XOR_j X_j                    (the same for every output word)
 *      L(X)_i = T ^ XOR_{c in COMPL} X_{(i+c) mod 16}
 * which is 5 loads instead of 11 per output word.  Exact, not an
 * approximation -- verified against dux/linear.py by the driver's --verify. */
static const int COMPL_FWD[2][5] = {{2, 5, 6, 11, 13}, {1, 2, 7, 9, 14}};

/* EXPT as shipped has 2*ORD entries so that a sum of two logs can be looked up
 * directly.  Reducing the exponent by hand instead keeps every access inside
 * the first ORD entries, which halves the table's cache footprint -- and the
 * moment loop below needs a reduction anyway (sums of up to four logs). */
static inline uint16_t gexp(uint32_t s)
{
    while (s >= ORD) s -= ORD;
    return EXPT[s];
}

static inline uint16_t gmul(uint16_t a, uint16_t b)
{
    uint32_t s = (uint32_t)LOGT[a] + (uint32_t)LOGT[b];
    uint16_t r = gexp(s);
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

static inline void sl_b(uint16_t s[WORDS][VB], uint16_t al)
{
    for (int b = 0; b < 4; b++) {
        uint16_t *x0 = s[4 * b], *x1 = s[4 * b + 1],
                 *x2 = s[4 * b + 2], *x3 = s[4 * b + 3];
        for (unsigned k = 0; k < VB; k++) {
            uint16_t a  = (uint16_t)(x2[k] ^ gmul(x0[k], x3[k]) ^ al);
            uint16_t bb = (uint16_t)(x1[k] ^ gmul(x3[k], a) ^ al);
            uint16_t c  = (uint16_t)(x0[k] ^ gmul(a, bb) ^ al);
            uint16_t y3 = (uint16_t)(x3[k] ^ gmul(bb, c) ^ al);
            x0[k] = c; x1[k] = bb; x2[k] = a; x3[k] = y3;
        }
    }
}

static inline void l_fwd(uint16_t *st, int t)
{
    uint16_t o[WORDS];
    const int *r = COMPL_FWD[t];
    uint16_t tot = 0;
    for (int i = 0; i < WORDS; i++) tot ^= st[i];
    for (int i = 0; i < WORDS; i++)
        o[i] = (uint16_t)(tot ^ st[(i + r[0]) & 15] ^ st[(i + r[1]) & 15] ^
                          st[(i + r[2]) & 15] ^ st[(i + r[3]) & 15] ^
                          st[(i + r[4]) & 15]);
    memcpy(st, o, sizeof(o));
}

/* Writes into a second buffer (the caller swaps) so that no state copy is
 * needed per round. */
static inline void l_fwd_b(uint16_t s[WORDS][VB], uint16_t o[WORDS][VB], int t)
{
    uint16_t tot[VB];
    const int *r = COMPL_FWD[t];
    for (unsigned k = 0; k < VB; k++) tot[k] = 0;
    for (int i = 0; i < WORDS; i++)
        for (unsigned k = 0; k < VB; k++) tot[k] ^= s[i][k];
    for (int i = 0; i < WORDS; i++) {
        const uint16_t *a0 = s[(i + r[0]) & 15], *a1 = s[(i + r[1]) & 15],
                       *a2 = s[(i + r[2]) & 15], *a3 = s[(i + r[3]) & 15],
                       *a4 = s[(i + r[4]) & 15];
        for (unsigned k = 0; k < VB; k++)
            o[i][k] = (uint16_t)(tot[k] ^ a0[k] ^ a1[k] ^ a2[k] ^ a3[k] ^ a4[k]);
    }
}

static inline void ark_b(uint16_t s[WORDS][VB], const uint16_t *rk)
{
    for (int i = 0; i < WORDS; i++)
        for (unsigned k = 0; k < VB; k++) s[i][k] ^= rk[i];
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
    if (memcmp(magic, "DUXCPMJ1", 8)) { fprintf(stderr, "bad magic\n"); exit(2); }
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
    /* t_xor(rk^i) selects L0/L1 in round i; O2 makes the choice irrelevant to
     * the SET of equations, but the kernel must reproduce the real cipher. */
    for (uint32_t i = 0; i <= ROUNDS; i++) {
        uint16_t w = RK[(size_t)i * WORDS + 15];
        TX[i] = (unsigned char)((w & 1u) ^ ((w >> 1) & 1u));
    }
    const uint64_t nbatch = (total + VB - 1) / VB;
    const size_t accn = (size_t)4 * NMOM;
    uint16_t *out = calloc((size_t)NSTRUCT * accn, 2);

    for (uint32_t st = 0; st < NSTRUCT; st++) {
        uint16_t *res = out + (size_t)st * accn;
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
#pragma omp parallel
        {
            uint16_t *loc = calloc(accn, 2);
            uint16_t buf[2][WORDS][VB];
            uint32_t lg[4];
            unsigned zf[4];
#pragma omp for schedule(static)
            for (uint64_t bi = 0; bi < nbatch; bi++) {
                const uint64_t base = bi * (uint64_t)VB;
                const unsigned nb = (unsigned)((total - base < (uint64_t)VB)
                                               ? (total - base) : VB);
                uint16_t (*s)[VB] = buf[0], (*o)[VB] = buf[1];
                for (int i = 0; i < WORDS; i++) {
                    uint16_t cst = CONST[(size_t)st * WORDS + i];
                    for (unsigned k = 0; k < VB; k++) s[i][k] = cst;
                }
                for (uint32_t j = 0; j < NACT; j++) {
                    uint16_t *dst = s[ACTIVE[j]];
                    const uint16_t *tab =
                        VALS + ((size_t)st * NACT + j) * ((size_t)1 << DIM);
                    for (unsigned k = 0; k < nb; k++)
                        dst[k] = tab[(size_t)(((base + k) >> ((uint64_t)DIM * j))
                                              & mask)];
                }
                /* full r-round encryption (dux/cipher.py DuX.encrypt):
                 *   x = P + rk^0
                 *   for i = 1..r-1:  x = SL(x); x = L_{t(rk^i)}(x); x += rk^i
                 *   x = SL(x); C = x + rk^r                                */
                ark_b(s, RK);
                for (uint32_t r = 1; r < ROUNDS; r++) {
                    uint16_t (*tmp)[VB];
                    sl_b(s, al);
                    l_fwd_b(s, o, TX[r]);
                    tmp = s; s = o; o = tmp;
                    ark_b(s, RK + (size_t)r * WORDS);
                }
                sl_b(s, al);
                ark_b(s, RK + (size_t)ROUNDS * WORDS);
                /* Ciphertext moments, per block.  A monomial prod C_i^{e_i}
                 * is one antilog lookup of sum_i e_i*log(C_i), so the four
                 * logs are taken once per block and every monomial costs a
                 * few adds plus a single gather (instead of up to three
                 * multiplications).  C_i = 0 is tracked with a flag rather
                 * than a sentinel log, which keeps EXPT small. */
                for (unsigned k = 0; k < nb; k++)
                    for (int b = 0; b < 4; b++) {
                        for (int i = 0; i < 4; i++) {
                            uint16_t c = s[4 * b + i][k];
                            lg[i] = LOGT[c];
                            zf[i] = (c == 0);
                        }
                        for (uint32_t m = 0; m < NMOM; m++) {
                            const unsigned char *e = MOMEXP + 4 * (size_t)m;
                            uint32_t acc = 0;
                            unsigned z = 0;
                            for (int i = 0; i < 4; i++)
                                if (e[i]) { acc += e[i] * lg[i]; z |= zf[i]; }
                            loc[(size_t)b * NMOM + m] ^= z ? 0 : gexp(acc);
                        }
                    }
            }
#pragma omp critical
            for (size_t i = 0; i < accn; i++) res[i] ^= loc[i];
            free(loc);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "structure %u/%u done (%.1f s)\n", st + 1, NSTRUCT,
                (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec));
    }

    printf("# DUXCPMOM n=%u rounds=%u nactive=%u dim=%u structures=%u moments=%u\n",
           N, ROUNDS, NACT, DIM, NSTRUCT, NMOM);
    for (uint32_t st = 0; st < NSTRUCT; st++)
        for (int b = 0; b < 4; b++) {
            printf("M %u %d", st, b);
            for (uint32_t m = 0; m < NMOM; m++)
                printf(" %u", out[(size_t)st * accn + (size_t)b * NMOM + m]);
            printf("\n");
        }
    return 0;
}
