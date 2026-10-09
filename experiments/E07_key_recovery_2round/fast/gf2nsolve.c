/* gf2nsolve.c -- blocked Gauss-Jordan over F_{2^n} for the r_KR = 2 systems.
 *
 * W12 solved the characteristic-2 systems with `modp_solve.solve_gf2n`, a
 * numpy rank-1 update per pivot.  That touches the whole trailing sub-matrix
 * once per pivot, so the 4 400 x 18 026 system of the four-unknown-block case
 * needs ~4 300 passes over 160 MB and was measured at ~4.7 h.
 *
 * This is the same panel algorithm as modp_solve.rref_blocked, in C, with XOR
 * for addition and log/antilog tables for multiplication:
 *
 *   * inside a panel of `block` columns only the m x block sub-matrix is
 *     touched, and the composed row operation is accumulated in D with
 *         row_i(current) = row_i(original) + sum_j D[i][j] row_{prow[j]}(original)
 *   * the trailing columns are then updated once per panel with
 *         trail[i] ^= sum_j D[i][j] * trail[prow[j]]
 *     which is one pass over the trailing block per panel instead of one per
 *     pivot, and is the loop that OpenMP parallelises over rows.
 *
 * Input / output are binary; experiments/E07_key_recovery_2round/gf2n_solve.py
 * writes the job and parses the result.
 *
 * Build:  make gf2nsolve
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static uint32_t N, M, NCOLS, BLOCK, ORD, LD;
static uint16_t *LOGT, *EXPT, *AUG, *D;

static inline uint16_t gmul(uint16_t a, uint16_t b)
{
    uint32_t s = (uint32_t)LOGT[a] + (uint32_t)LOGT[b];
    uint16_t r = EXPT[s];
    return (a && b) ? r : 0;
}

static uint16_t ginv(uint16_t a)
{
    return EXPT[ORD - LOGT[a]];          /* a != 0 */
}

static void rd(FILE *f, void *p, size_t n)
{
    if (fread(p, 1, n, f) != n) { fprintf(stderr, "short read\n"); exit(2); }
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s job.bin [out.bin]\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    char magic[8];
    rd(f, magic, 8);
    if (memcmp(magic, "DUXGF2N1", 8)) { fprintf(stderr, "bad magic\n"); return 2; }
    uint32_t h[4];
    rd(f, h, sizeof(h));
    N = h[0]; M = h[1]; NCOLS = h[2]; BLOCK = h[3];
    ORD = (1u << N) - 1u;
    LD = NCOLS + 1;                       /* augmented: A | b */
    LOGT = malloc((size_t)(1u << N) * 2);
    EXPT = malloc((size_t)2 * ORD * 2);
    rd(f, LOGT, (size_t)(1u << N) * 2);
    rd(f, EXPT, (size_t)2 * ORD * 2);
    AUG = malloc((size_t)M * LD * 2);
    if (!AUG) { fprintf(stderr, "out of memory\n"); return 2; }
    rd(f, AUG, (size_t)M * LD * 2);
    fclose(f);

    D = calloc((size_t)M * BLOCK, 2);
    uint32_t *prow = malloc((size_t)BLOCK * 4);
    uint32_t *piv_col = malloc((size_t)(NCOLS + 1) * 4);
    uint32_t *piv_row = malloc((size_t)(NCOLS + 1) * 4);
    unsigned char *used = calloc(M, 1);
    uint16_t *buf = malloc((size_t)BLOCK * LD * 2);
    uint32_t npiv = 0;

    for (uint32_t c0 = 0; c0 < NCOLS && npiv < M; c0 += BLOCK) {
        uint32_t c1 = c0 + BLOCK < NCOLS ? c0 + BLOCK : NCOLS;
        uint32_t w = c1 - c0, k = 0;
        memset(D, 0, (size_t)M * BLOCK * 2);
        for (uint32_t c = c0; c < c1; c++) {
            uint32_t pr = M;
            for (uint32_t i = 0; i < M; i++)
                if (!used[i] && AUG[(size_t)i * LD + c]) { pr = i; break; }
            if (pr == M) continue;
            uint16_t inv = ginv(AUG[(size_t)pr * LD + c]);
            if (inv != 1) {
                uint16_t *rp = AUG + (size_t)pr * LD;
                for (uint32_t t = c0; t < c1; t++) rp[t] = gmul(rp[t], inv);
                uint16_t *dp = D + (size_t)pr * BLOCK;
                for (uint32_t t = 0; t < w; t++) dp[t] = gmul(dp[t], inv);
            }
            D[(size_t)pr * BLOCK + k] ^= (uint16_t)(inv ^ 1u);
            const uint16_t *rp = AUG + (size_t)pr * LD;
            const uint16_t *dp = D + (size_t)pr * BLOCK;
#pragma omp parallel for schedule(static)
            for (uint32_t i = 0; i < M; i++) {
                uint16_t fct = AUG[(size_t)i * LD + c];
                if (i == pr || !fct) continue;
                uint16_t *ri = AUG + (size_t)i * LD;
                uint16_t *di = D + (size_t)i * BLOCK;
                uint32_t lf = LOGT[fct];
                for (uint32_t t = c0; t < c1; t++) {
                    uint16_t v = rp[t];
                    ri[t] ^= (uint16_t)(EXPT[lf + LOGT[v]] & -(uint16_t)(v != 0));
                }
                for (uint32_t t = 0; t < w; t++) {
                    uint16_t v = dp[t];
                    di[t] ^= (uint16_t)(EXPT[lf + LOGT[v]] & -(uint16_t)(v != 0));
                }
                di[k] ^= fct;
            }
            used[pr] = 1;
            prow[k] = pr;
            piv_col[npiv] = c;
            piv_row[npiv] = pr;
            npiv++; k++;
        }
        if (k && c1 < LD) {
            uint32_t ntr = LD - c1;
            for (uint32_t j = 0; j < k; j++)
                memcpy(buf + (size_t)j * ntr, AUG + (size_t)prow[j] * LD + c1, ntr * 2);
#pragma omp parallel
            {
                /* Multiplication by a FIXED scalar is F_2-linear, so it splits
                 * over the two bytes of the operand:  f*v = f*v_lo ^ f*(v_hi<<8).
                 * Two 256-entry L1-resident tables per (row, pivot) pair replace
                 * the two dependent gathers into the 390 KB log/antilog tables;
                 * the 512 table entries are amortised over ntr ~ 10^4 columns. */
                uint16_t T0[256], T1[256];
                const uint32_t NLO = (N >= 8) ? 256u : (1u << N);
#pragma omp for schedule(static)
                for (uint32_t i = 0; i < M; i++) {
                    uint16_t *ri = AUG + (size_t)i * LD + c1;
                    const uint16_t *di = D + (size_t)i * BLOCK;
                    for (uint32_t j = 0; j < k; j++) {
                        uint16_t fct = di[j];
                        if (!fct) continue;
                        const uint16_t *rj = buf + (size_t)j * ntr;
                        uint32_t lf = LOGT[fct];
                        /* the field has 2^N elements, so only the first
                         * min(256, 2^N) low-byte entries are ever read, and the
                         * high-byte table is identically zero when N <= 8 */
                        memset(T0, 0, sizeof(T0));
                        memset(T1, 0, sizeof(T1));
                        for (uint32_t b = 1; b < NLO; b++)
                            T0[b] = EXPT[lf + LOGT[b]];
                        if (N > 8)
                            for (uint32_t b = 1; b < 256; b++)
                                T1[b] = EXPT[lf + LOGT[b << 8]];
                        for (uint32_t t = 0; t < ntr; t++) {
                            uint16_t v = rj[t];
                            ri[t] ^= (uint16_t)(T0[v & 0xFF] ^ T1[v >> 8]);
                        }
                    }
                }
            }
        }
        fprintf(stderr, "  rref(2^n): %u pivots after column %u/%u\n", npiv, c1, NCOLS);
    }

    /* a pivot column is DETERMINED iff its row has no free column left */
    unsigned char *isfree = malloc(NCOLS);
    memset(isfree, 1, NCOLS);
    for (uint32_t i = 0; i < npiv; i++) isfree[piv_col[i]] = 0;
    uint32_t nfree = 0;
    for (uint32_t c = 0; c < NCOLS; c++) nfree += isfree[c];
    printf("rank %u nfree %u\n", npiv, nfree);
    for (uint32_t i = 0; i < npiv; i++) {
        const uint16_t *r = AUG + (size_t)piv_row[i] * LD;
        int det = 1;
        for (uint32_t c = 0; c < NCOLS; c++)
            if (isfree[c] && r[c]) { det = 0; break; }
        if (det) printf("D %u %u\n", piv_col[i], (unsigned)r[NCOLS]);
    }
    return 0;
}
