/* mom2n.c -- the F_{2^n} moment matrices of the r_KR = 2 assembly, in C.
 *
 * `assemble_fast_2n.moment_matrix_2n` computes, per unordered pair of unknown
 * inner blocks,
 *
 *     Mom[i][j] = XOR_{p < N} ( PW[A][i][p] * PW[B][j][p] )        (nP ~ 50)
 *
 * i.e. nP^2 * N field multiplications; at the DuX(2^16) scale (N = 2^16 points
 * per structure, 10 unordered pairs) that is 1.6e9 per structure, which numpy
 * does at ~1.5 ns/element -- about 25 core-seconds per structure, and the
 * assembly of ~1 400 structures then dominates the whole attack.
 *
 * The caller passes the DISCRETE LOGARITHMS of the two operand matrices as
 * int32, with the field element 0 mapped to the sentinel 2*(2^n-1), together
 * with an antilog table padded with zeros past 2*(2^n-1).  Then a product is a
 * single add + gather, and a zero operand automatically yields 0:
 *
 *     Mom[i][j] = XOR_p expx[ la[i][p] + lb[j][p] ]
 *
 * The point axis is walked in chunks so that the two operand panels stay in L2
 * while all nP^2 pairs of rows are accumulated, and the (i, j) loops are
 * 2x2-TILED: four independent XOR chains, and each panel element is loaded
 * once per two products instead of once per product.
 *
 * S13 measured both knobs on the machine the Yu2X runs use (Xeon Gold 6230R,
 * 1 MB L2 per core, two hyperthreads per core), single-threaded, on the shapes
 * the 8-round Yu2X-8 and 11-round Yu2X-16 assemblies actually call with
 * (nP = 63; N = 2^17 and 2^16).  Every variant returns bit-identical output --
 * XOR is associative, so the chunking and the tiling only reorder the sum:
 *
 *     variant            F_{2^8}, N = 2^17     F_{2^16}, N = 2^16
 *     plain, CHUNK 4096     457 M gather/s        230 M gather/s
 *     plain, CHUNK 2048     477                   394
 *     tiled, CHUNK 4096     545                   175
 *     tiled, CHUNK 2048     548                   415        <- chosen
 *     tiled, CHUNK 1024     485                   347
 *
 * CHUNK 4096 puts 2 x 63 x 4096 x 4 B = 2 MB of panel in flight, which spills
 * a 1 MB L2 that two hyperthreads already share; 2048 halves that.
 *
 * Build:  make mom2n         (a shared library, loaded through ctypes)
 */
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef CHUNK
#define CHUNK 2048
#endif

void mom2n(const int32_t *la, const int32_t *lb, int32_t nPa, int32_t nPb,
           int64_t N, const int32_t *expx, int32_t *out)
{
    memset(out, 0, (size_t)nPa * nPb * sizeof(int32_t));
#pragma omp parallel
    {
        int32_t *acc = calloc((size_t)nPa * nPb, sizeof(int32_t));
#pragma omp for schedule(static)
        for (int64_t c0 = 0; c0 < N; c0 += CHUNK) {
            int64_t c1 = c0 + CHUNK < N ? c0 + CHUNK : N;
            int32_t i = 0;
            for (; i + 1 < nPa; i += 2) {          /* the 2x2 tile */
                const int32_t *a0 = la + (size_t)i * N;
                const int32_t *a1 = la + (size_t)(i + 1) * N;
                int32_t j = 0;
                for (; j + 1 < nPb; j += 2) {
                    const int32_t *b0 = lb + (size_t)j * N;
                    const int32_t *b1 = lb + (size_t)(j + 1) * N;
                    int32_t s00 = 0, s01 = 0, s10 = 0, s11 = 0;
                    for (int64_t p = c0; p < c1; p++) {
                        int32_t x0 = a0[p], x1 = a1[p];
                        int32_t y0 = b0[p], y1 = b1[p];
                        s00 ^= expx[x0 + y0];
                        s01 ^= expx[x0 + y1];
                        s10 ^= expx[x1 + y0];
                        s11 ^= expx[x1 + y1];
                    }
                    acc[(size_t)i * nPb + j] ^= s00;
                    acc[(size_t)i * nPb + j + 1] ^= s01;
                    acc[(size_t)(i + 1) * nPb + j] ^= s10;
                    acc[(size_t)(i + 1) * nPb + j + 1] ^= s11;
                }
                for (; j < nPb; j++) {             /* odd nPb */
                    const int32_t *b0 = lb + (size_t)j * N;
                    int32_t s0 = 0, s1 = 0;
                    for (int64_t p = c0; p < c1; p++) {
                        int32_t y0 = b0[p];
                        s0 ^= expx[a0[p] + y0];
                        s1 ^= expx[a1[p] + y0];
                    }
                    acc[(size_t)i * nPb + j] ^= s0;
                    acc[(size_t)(i + 1) * nPb + j] ^= s1;
                }
            }
            for (; i < nPa; i++) {                 /* odd nPa */
                const int32_t *a0 = la + (size_t)i * N;
                for (int32_t j = 0; j < nPb; j++) {
                    const int32_t *b0 = lb + (size_t)j * N;
                    int32_t s = 0;
                    for (int64_t p = c0; p < c1; p++)
                        s ^= expx[a0[p] + b0[p]];
                    acc[(size_t)i * nPb + j] ^= s;
                }
            }
        }
#pragma omp critical
        for (size_t t = 0; t < (size_t)nPa * nPb; t++) out[t] ^= acc[t];
        free(acc);
    }
}

/* v[i] = XOR_p PW[i][p] * y[p]  (the vector form, same encoding) */
void momvec2n(const int32_t *la, int32_t nPa, int64_t N, const int32_t *ly,
              const int32_t *expx, int32_t *out)
{
#pragma omp parallel for schedule(static)
    for (int32_t i = 0; i < nPa; i++) {
        const int32_t *ai = la + (size_t)i * N;
        int32_t s = 0;
        for (int64_t p = 0; p < N; p++) s ^= expx[ai[p] + ly[p]];
        out[i] = s;
    }
}
