# Y02 . Ni et al.'s degree tables vs the max-plus criterion (O7)

Source (cross-check literature, not a specification): Ni, Wang, Li, *Tracking Algebraic Degree with Exponent Sets*, DCC 2026, 94:123 -- Table 3 and Table 4 (p. 13), Table 6 (pp. 20-21), Table 7 (p. 22). Every printed number is transcribed verbatim into `results/Y02_degree_bounds/ni_tables.json`.

**Conventions.** Their round *r* is our layer *r* (their round 0 is the input); their output index *i* is block 0, position *i*; our formal degree `D` is the max-plus bound on the SUM of the exponents, taken on block 0 exactly (no max over blocks). The Boolean degree it implies is the knapsack

```
hw(D, s, n) = max { sum_i HW(a_i) : sum_i a_i <= D, a_i <= 2^n - 1 }
```

which for `s = 1` is Ni et al.'s own Lemma 1, `floor(log2(D+1))`. Because a Hamming weight `w` costs at least `2^w - 1` and that cost is convex, `hw(D, s, n) = s n` exactly when `D >= T = s (2^n - 1)`: **the Boolean criterion `deg < s n` and Theorem O7's `D < T` are literally the same condition**, so whatever slack the intermediate cells show, the two machines can only disagree about a balanced layer when one of the two bounds is not tight.

The conversion is exact only when the `s` variables reach the output word symmetrically -- in particular when they all sit in one 4-word block. For active words spread over several blocks the total degree cannot see that the early layers mix only the same-block variable, so `hw` overshoots; that is where Ni's per-variable exponent sets are genuinely sharper.

## 1. Cell counts

| table | cells compared | agree | max-plus looser (Ni sharper) | max-plus tighter (O7 sharper) | not listed |
|---|---|---|---|---|---|
| table3 | 116 | 94 | 0 | 22 | 44 |
| table4 | 64 | 62 | 2 | 0 | 16 |
| table6 | 380 | 135 | 209 | 36 | 12 |
| table7 | 212 | 62 | 150 | 0 | 8 |
| **all** | **772** | **353** | **361** | **58** | **80** |

Split by whether the active words share one block (where the conversion is exact):

| active words | cells | agree | max-plus looser | max-plus tighter |
|---|---|---|---|---|
| same block | 180 | 156 | 2 | 22 |
| several blocks | 592 | 197 | 359 | 36 |

## 2. Where the two machines disagree about a BALANCED word

These are the only cells that change a distinguisher. `O7 balanced` means `hw < s n` (equivalently `D < T`); `Ni balanced` means their printed degree is `< s n`.

| table | cipher | active words | out | layer | Ni | s n | D | hw | O7 | Ni | D = 2^k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| table3 | Yu2X-8 | [3] | 0 | 5 | 8 | 8 | 243 | 7 | balanced | - |  |
| table3 | Yu2X-8 | [3] | 1 | 5 | 8 | 8 | 250 | 7 | balanced | - |  |
| table3 | Yu2X-16 | [3] | 0 | 9 | 16 | 16 | 64047 | 15 | balanced | - |  |
| table3 | Yu2X-16 | [3] | 1 | 9 | 16 | 16 | 64091 | 15 | balanced | - |  |
| table6 | Yu2X-8 | [0, 4, 8, 12] | 0 | 6 | 30 | 32 | 1024 | 32 | - | balanced | yes |
| table6 | Yu2X-8 | [0, 4, 8, 12] | 1 | 6 | 30 | 32 | 1024 | 32 | - | balanced | yes |
| table6 | Yu2X-8 | [0, 1, 3, 4, 8, 12] | 2 | 6 | 46 | 48 | 1536 | 48 | - | balanced |  |
| table6 | Yu2X-8 | [0, 1, 3, 4, 8, 12] | 3 | 6 | 47 | 48 | 2048 | 48 | - | balanced | yes |
| table6 | Yu2X-8 | [0, 1, 3, 4, 6, 8, 12] | 3 | 6 | 54 | 56 | 2048 | 56 | - | balanced | yes |
| table6 | Yu2X-8 | [0, 1, 4, 5, 8, 9, 12, 13] | 2 | 6 | 64 | 64 | 1536 | 60 | balanced | - |  |
| table6 | Yu2X-8 | [0, 1, 3, 4, 5, 8, 9, 12, 13] | 2 | 6 | 72 | 72 | 1536 | 66 | balanced | - |  |
| table6 | Yu2X-8 | [0, 1, 3, 4, 5, 8, 9, 12, 13] | 3 | 6 | 72 | 72 | 2048 | 70 | balanced | - | yes |
| table6 | Yu2X-8 | [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15] | 3 | 6 | 100 | 104 | 3644 | 104 | - | balanced |  |
| table6 | Yu2X-8 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15] | 3 | 6 | 108 | 112 | 3882 | 112 | - | balanced |  |
| table6 | Yu2X-8 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15] | 3 | 6 | 116 | 120 | 4003 | 120 | - | balanced |  |
| table6 | Yu2X-8 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] | 3 | 6 | 124 | 128 | 4096 | 128 | - | balanced | yes |
| table7 | Yu2X-16 | [0, 4, 8, 12] | 0 | 10 | 62 | 64 | 262144 | 64 | - | balanced | yes |
| table7 | Yu2X-16 | [0, 4, 8, 12] | 1 | 10 | 62 | 64 | 262144 | 64 | - | balanced | yes |
| table7 | Yu2X-16 | [0, 1, 3, 4, 8, 12] | 2 | 10 | 94 | 96 | 393216 | 96 | - | balanced |  |
| table7 | Yu2X-16 | [0, 1, 3, 4, 8, 12] | 3 | 10 | 95 | 96 | 524288 | 96 | - | balanced | yes |
| table7 | Yu2X-16 | [0, 1, 3, 4, 6, 8, 12] | 3 | 10 | 110 | 112 | 524288 | 112 | - | balanced | yes |

21 cells out of 772 compared.

## 3. Saturation cells with D exactly a power of two (Frobenius boundary)

A formal degree that is exactly `2^k` has a Frobenius power as its leading monomial, whose coefficient is often forced to vanish; O7 cannot use that and reports `not balanced` with margin 0 or -4.

| table | cipher | active words | out | layer | Ni | s n | D | O7 | Ni |
|---|---|---|---|---|---|---|---|---|---|
| table6 | Yu2X-8 | [0, 4, 8, 12] | 0 | 6 | 30 | 32 | 1024 | - | balanced |
| table6 | Yu2X-8 | [0, 4, 8, 12] | 0 | 7 | 32 | 32 | 4096 | - | - |
| table6 | Yu2X-8 | [0, 4, 8, 12] | 1 | 6 | 30 | 32 | 1024 | - | balanced |
| table6 | Yu2X-8 | [0, 4, 8, 12] | 1 | 7 | 32 | 32 | 4096 | - | - |
| table6 | Yu2X-8 | [0, 4, 8, 12] | 3 | 6 | 32 | 32 | 2048 | - | - |
| table6 | Yu2X-8 | [0, 1, 4, 8, 12] | 0 | 7 | 40 | 40 | 4096 | - | - |
| table6 | Yu2X-8 | [0, 1, 4, 8, 12] | 1 | 7 | 40 | 40 | 4096 | - | - |
| table6 | Yu2X-8 | [0, 1, 4, 8, 12] | 3 | 6 | 40 | 40 | 2048 | - | - |
| table6 | Yu2X-8 | [0, 1, 3, 4, 8, 12] | 0 | 7 | 48 | 48 | 4096 | - | - |
| table6 | Yu2X-8 | [0, 1, 3, 4, 8, 12] | 1 | 7 | 48 | 48 | 4096 | - | - |
| table6 | Yu2X-8 | [0, 1, 3, 4, 8, 12] | 3 | 6 | 47 | 48 | 2048 | - | balanced |
| table6 | Yu2X-8 | [0, 1, 3, 4, 8, 12] | 3 | 7 | 48 | 48 | 8192 | - | - |
| table6 | Yu2X-8 | [0, 1, 3, 4, 6, 8, 12] | 0 | 7 | 56 | 56 | 4096 | - | - |
| table6 | Yu2X-8 | [0, 1, 3, 4, 6, 8, 12] | 1 | 7 | 56 | 56 | 4096 | - | - |
| table6 | Yu2X-8 | [0, 1, 3, 4, 6, 8, 12] | 3 | 6 | 54 | 56 | 2048 | - | balanced |
| table6 | Yu2X-8 | [0, 1, 3, 4, 6, 8, 12] | 3 | 7 | 56 | 56 | 8192 | - | - |
| table6 | Yu2X-8 | [0, 1, 4, 5, 8, 9, 12, 13] | 0 | 7 | 64 | 64 | 4096 | - | - |
| table6 | Yu2X-8 | [0, 1, 4, 5, 8, 9, 12, 13] | 1 | 7 | 64 | 64 | 4096 | - | - |
| table6 | Yu2X-8 | [0, 1, 4, 5, 8, 9, 12, 13] | 3 | 6 | 64 | 64 | 2048 | - | - |
| table6 | Yu2X-8 | [0, 1, 3, 4, 5, 8, 9, 12, 13] | 0 | 7 | 72 | 72 | 4096 | - | - |
| table6 | Yu2X-8 | [0, 1, 3, 4, 5, 8, 9, 12, 13] | 1 | 7 | 72 | 72 | 4096 | - | - |
| table6 | Yu2X-8 | [0, 1, 3, 4, 5, 8, 9, 12, 13, 15] | 0 | 7 | 80 | 80 | 4096 | - | - |
| table6 | Yu2X-8 | [0, 1, 3, 4, 5, 8, 9, 12, 13, 15] | 1 | 7 | 80 | 80 | 4096 | - | - |
| table6 | Yu2X-8 | [0, 1, 3, 4, 5, 8, 9, 12, 13, 15] | 3 | 7 | 80 | 80 | 8192 | - | - |
| table6 | Yu2X-8 | [0, 1, 3, 4, 5, 7, 8, 9, 12, 13, 15] | 0 | 7 | 88 | 88 | 4096 | - | - |
| table6 | Yu2X-8 | [0, 1, 3, 4, 5, 7, 8, 9, 12, 13, 15] | 1 | 7 | 88 | 88 | 4096 | - | - |
| table6 | Yu2X-8 | [0, 1, 3, 4, 5, 7, 8, 9, 12, 13, 15] | 3 | 7 | 88 | 88 | 8192 | - | - |
| table6 | Yu2X-8 | [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15] | 0 | 7 | 96 | 96 | 4096 | - | - |
| table6 | Yu2X-8 | [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15] | 1 | 7 | 96 | 96 | 4096 | - | - |
| table6 | Yu2X-8 | [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15] | 3 | 7 | 96 | 96 | 8192 | - | - |
| table6 | Yu2X-8 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] | 0 | 7 | 128 | 128 | 8192 | - | - |
| table6 | Yu2X-8 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] | 1 | 7 | 128 | 128 | 8192 | - | - |
| table6 | Yu2X-8 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] | 3 | 6 | 124 | 128 | 4096 | - | balanced |
| table6 | Yu2X-8 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] | 3 | 7 | 128 | 128 | 16384 | - | - |
| table7 | Yu2X-16 | [0, 4, 8, 12] | 0 | 10 | 62 | 64 | 262144 | - | balanced |
| table7 | Yu2X-16 | [0, 4, 8, 12] | 0 | 11 | 64 | 64 | 1048576 | - | - |
| table7 | Yu2X-16 | [0, 4, 8, 12] | 1 | 10 | 62 | 64 | 262144 | - | balanced |
| table7 | Yu2X-16 | [0, 4, 8, 12] | 1 | 11 | 64 | 64 | 1048576 | - | - |
| table7 | Yu2X-16 | [0, 4, 8, 12] | 3 | 10 | 64 | 64 | 524288 | - | - |
| table7 | Yu2X-16 | [0, 1, 4, 8, 12] | 0 | 11 | 80 | 80 | 1048576 | - | - |
| table7 | Yu2X-16 | [0, 1, 4, 8, 12] | 1 | 11 | 80 | 80 | 1048576 | - | - |
| table7 | Yu2X-16 | [0, 1, 4, 8, 12] | 3 | 10 | 80 | 80 | 524288 | - | - |
| table7 | Yu2X-16 | [0, 1, 3, 4, 8, 12] | 0 | 11 | 96 | 96 | 1048576 | - | - |
| table7 | Yu2X-16 | [0, 1, 3, 4, 8, 12] | 1 | 11 | 96 | 96 | 1048576 | - | - |
| table7 | Yu2X-16 | [0, 1, 3, 4, 8, 12] | 3 | 10 | 95 | 96 | 524288 | - | balanced |
| table7 | Yu2X-16 | [0, 1, 3, 4, 8, 12] | 3 | 11 | 96 | 96 | 2097152 | - | - |
| table7 | Yu2X-16 | [0, 1, 3, 4, 6, 8, 12] | 0 | 11 | 112 | 112 | 1048576 | - | - |
| table7 | Yu2X-16 | [0, 1, 3, 4, 6, 8, 12] | 1 | 11 | 112 | 112 | 1048576 | - | - |
| table7 | Yu2X-16 | [0, 1, 3, 4, 6, 8, 12] | 3 | 10 | 110 | 112 | 524288 | - | balanced |
| table7 | Yu2X-16 | [0, 1, 3, 4, 6, 8, 12] | 3 | 11 | 112 | 112 | 2097152 | - | - |

## 4. Same-block cells that disagree

| table | cipher | active words | out | layer | Ni | D | hw | verdict |
|---|---|---|---|---|---|---|---|---|
| table3 | Yu2X-8 | [3] | 0 | 4 | 6 | 62 | 5 | maxplus_tighter |
| table3 | Yu2X-8 | [3] | 0 | 5 | 8 | 243 | 7 | maxplus_tighter |
| table3 | Yu2X-8 | [3] | 1 | 4 | 6 | 62 | 5 | maxplus_tighter |
| table3 | Yu2X-8 | [3] | 1 | 5 | 8 | 250 | 7 | maxplus_tighter |
| table3 | Yu2X-8 | [3] | 3 | 4 | 7 | 124 | 6 | maxplus_tighter |
| table3 | Yu2X-16 | [3] | 0 | 4 | 6 | 62 | 5 | maxplus_tighter |
| table3 | Yu2X-16 | [3] | 0 | 5 | 8 | 243 | 7 | maxplus_tighter |
| table3 | Yu2X-16 | [3] | 0 | 6 | 10 | 1006 | 9 | maxplus_tighter |
| table3 | Yu2X-16 | [3] | 0 | 7 | 12 | 3969 | 11 | maxplus_tighter |
| table3 | Yu2X-16 | [3] | 0 | 8 | 14 | 16044 | 13 | maxplus_tighter |
| table3 | Yu2X-16 | [3] | 0 | 9 | 16 | 64047 | 15 | maxplus_tighter |
| table3 | Yu2X-16 | [3] | 1 | 4 | 6 | 62 | 5 | maxplus_tighter |
| table3 | Yu2X-16 | [3] | 1 | 5 | 8 | 250 | 7 | maxplus_tighter |
| table3 | Yu2X-16 | [3] | 1 | 6 | 10 | 1006 | 9 | maxplus_tighter |
| table3 | Yu2X-16 | [3] | 1 | 7 | 12 | 3993 | 11 | maxplus_tighter |
| table3 | Yu2X-16 | [3] | 1 | 8 | 14 | 16082 | 13 | maxplus_tighter |
| table3 | Yu2X-16 | [3] | 1 | 9 | 16 | 64091 | 15 | maxplus_tighter |
| table3 | Yu2X-16 | [3] | 3 | 4 | 7 | 124 | 6 | maxplus_tighter |
| table3 | Yu2X-16 | [3] | 3 | 5 | 9 | 493 | 8 | maxplus_tighter |
| table3 | Yu2X-16 | [3] | 3 | 6 | 11 | 2012 | 10 | maxplus_tighter |
| table3 | Yu2X-16 | [3] | 3 | 7 | 13 | 7962 | 12 | maxplus_tighter |
| table3 | Yu2X-16 | [3] | 3 | 8 | 15 | 32126 | 14 | maxplus_tighter |
| table4 | Yu2X-8 | [0, 1] | 1 | 3 | 5 | 14 | 6 | maxplus_looser |
| table4 | Yu2X-16 | [0, 1] | 1 | 3 | 5 | 14 | 6 | maxplus_looser |

## 5. Rows printed for several input indices: 0 inconsistencies

Every row Ni prints for a set of input indices ("0/1/2", "(0,1)/(0,2)") really does carry the same max-plus bound for each member of the set.

