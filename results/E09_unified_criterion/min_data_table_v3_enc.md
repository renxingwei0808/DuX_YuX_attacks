# E09 · minimal-data table v3 inside the O7 framework, **encryption (CPA) direction**

The decryption-direction v3 (`min_data_table_v3.md`) is kept **unchanged**; this file is the output of the same script,
`min_data_v3.py --direction enc`: all 15 balanced-position classes, the dim K of each class
(`tools/cheap_rows.py --direction enc`) and its usability. The search space is that of v2:
0 or 1 free block (O11) x every subset of the remaining words x every layer, active sets canonicalised modulo block rotation;
characteristic 2 allows mixed subspace dimensions.

## 1. Cell-by-cell comparison with the encryption rows of Table 12 (eprint v1, Appendix F, `tab:opt`)

Table 12 has two encryption rows, the two DuX instances at layer 7 with a full block `1110` @2^64 and the two YuX instances at layer 7 with a full block
`0111` @2^64, and two footnote sentences: "DuX(2^8) and Yu2X-8 stop at layer 4 in the encryption direction" and
"every other next layer is unreachable for any structure in the search space". "Usable classes" are those of Appendix F: `1111`, `xxx1` (= `0001`) for DuX,
`1110` for both families and `0111` for YuX.

| Instance | Table 12 | v3: deepest usable layer | Cheapest class / data reaching it | Witness structure | Deepest layer of any key-recovery class | Deepest layer of any class | Next layer | Agrees |
|---|---|---|---|---|---|---|---|---|
| DuX(65537) | layer 7 `1110` full block 2^64 | 7 | `1110` @2^64.0 | free block [0], words [] (4 words, degree 163840) | 7 | 7 | layer 8 is **unreachable** for any structure | ✅ |
| DuX(2^16) | layer 7 `1110` full block 2^64 | 7 | `1110` @2^64 | free block [0], words [] (4 words, degree 163840) | 7 | 7 | layer 8 is **unreachable** for any structure | ✅ |
| YupX-65537 | layer 7 `0111` full block 2^64 | 7 | `0111` @2^64.0 | free block [0], words [] (4 words, degree 163840) | 7 | 7 | layer 8 is **unreachable** for any structure | ✅ |
| Yu2X-16 | layer 7 `0111` full block 2^64 | 7 | `0111` @2^64 | free block [0], words [] (4 words, degree 163840) | 7 | 7 | layer 8 is **unreachable** for any structure | ✅ |
| DuX(2^8) | stops at layer 4 | 4 | `1110` @2^24 | no free block, words [1, 5, 9], dims [8, 8, 8] (3 words, degree 640) | 4 | 5 | layer 5 has only `0010` @2^100 (dim K = 1, char-2 unusable (dim K = 1 combined row collapses in characteristic 2)) | ✅ |
| Yu2X-8 | stops at layer 4 | 4 | `0111` / `1110` / `1111` @2^32 | no free block, words [1, 2, 3, 5], dims [8, 8, 8, 8] (4 words, degree 960) | 4 | 5 | layer 5 has only `0001` @2^100 (dim K = 0, distinguisher only) | ✅ |

"Usable" follows Table 12 (the classes listed in Appendix F); "any key-recovery class" means the classes among the 15 with dim K >= 2 or with plain rows;
"any class" includes the distinguisher-only classes and the unusable dim K = 1 classes in characteristic 2. The two footnote sentences of Table 12 are read over the usable classes:
if the next layer is reachable only by unusable classes, "stops at layer l" still holds, and the reachable cells are listed in the "next layer" column.

**The YuX layer-7 full-block cell**: the criterion gives `0111` (the margin at position 0 is exactly 0: D = 8^6 = 262 144 = T = 4 x 65 536).
By the rank-3 argument of `experiments/Y09_topform_constants/run.py --part rank` (the top forms of YuX in the encryption direction lie in
F_p[y0, y1, sigma], with fibres of size p), **the sum at that boundary position is always 0**, so the paper writes this cell as `1111`
(O14-CPA); this table keeps the criterion's own `0111` and is not edited by hand. The corresponding eight rows of DuX have rank 4, so there is no such shortcut;
the constant c_b(65537) at position 3 of layer 7 of DuX(65537) was not computed, and `1110` stands.

## 2. Cell-by-cell comparison with the encryption rows of `min_data_table_v2.md`

v2 has 216 encryption cells (6 instances x 12 layers x 3 classes; v2's `xxx1` corresponds to v3's `0001`):
**216 identical, 0 cheaper, 0 more expensive**.

### Identical cells

| Instance | Class (v2 / v3) | Min. data at layers 1-12 (log2; `—` = unreachable) | dim K (v2 / v3) |
|---|---|---|---|
| DuX(65537) | `1111` / `1111` | 16, 16, 16, 16, 16, 32, 208, —, —, —, —, — | 8 / 8 |
| DuX(65537) | `xxx1` / `0001` | 16, 16, 16, 16, 16, 32, 208, —, —, —, —, — | 4 / 4 |
| DuX(65537) | `1110` / `1110` | 16, 16, 16, 16, 16, 16, 64, —, —, —, —, — | 4 / 4 |
| DuX(2^16) | `1111` / `1111` | 2, 5, 8, 11, 14, 18, 196, —, —, —, —, — | 8 / 8 |
| DuX(2^16) | `xxx1` / `0001` | 2, 5, 8, 11, 14, 18, 196, —, —, —, —, — | 4 / 4 |
| DuX(2^16) | `1110` / `1110` | 2, 4, 7, 10, 13, 16, 64, —, —, —, —, — | 4 / 4 |
| DuX(2^8) | `1111` / `1111` | 2, 5, 8, 32, —, —, —, —, —, —, —, — | 8 / 8 |
| DuX(2^8) | `xxx1` / `0001` | 2, 5, 8, 32, —, —, —, —, —, —, —, — | 4 / 4 |
| DuX(2^8) | `1110` / `1110` | 2, 4, 7, 24, —, —, —, —, —, —, —, — | 4 / 4 |
| YupX-65537 | `1111` / `1111` | 16, 16, 16, 16, 16, 32, 208, —, —, —, —, — | 8 / 8 |
| YupX-65537 | `0111` / `0111` | 16, 16, 16, 16, 16, 16, 64, —, —, —, —, — | 4 / 4 |
| YupX-65537 | `1110` / `1110` | 16, 16, 16, 16, 16, 32, 208, —, —, —, —, — | 4 / 4 |
| Yu2X-16 | `1111` / `1111` | 3, 5, 8, 11, 14, 32, 196, —, —, —, —, — | 8 / 8 |
| Yu2X-16 | `0111` / `0111` | 2, 5, 7, 10, 13, 16, 64, —, —, —, —, — | 4 / 4 |
| Yu2X-16 | `1110` / `1110` | 3, 5, 8, 11, 14, 32, 196, —, —, —, —, — | 4 / 4 |
| Yu2X-8 | `1111` / `1111` | 3, 5, 8, 32, —, —, —, —, —, —, —, — | 8 / 8 |
| Yu2X-8 | `0111` / `0111` | 2, 5, 7, 32, —, —, —, —, —, —, —, — | 4 / 4 |
| Yu2X-8 | `1110` / `1110` | 3, 5, 8, 32, —, —, —, —, —, —, —, — | 4 / 4 |

## 3. Minimal data per instance, layer and class

Only reachable cells are listed; a layer marked `—` is unreachable inside the O7 framework.

### DuX(65537) · encryption (CPA)

| Layer | Class | dim K | Usability | Min. data | Free block | Extra words | Dims | Degree needed |
|---|---|---|---|---|---|---|---|---|
| 1 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 2 |
| 1 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1 |
| 1 | `0010` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 1 |
| 1 | `0001` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 3 |
| 1 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 2 |
| 1 | `1010` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 2 |
| 1 | `1001` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 3 |
| 1 | `0110` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 1 |
| 1 | `0101` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 3 |
| 1 | `0011` | 5 | dim K = 5 | 2^16.0 | — | 1 | None | 3 |
| 1 | `1110` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 2 |
| 1 | `1101` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 3 |
| 1 | `1011` | 5 | dim K = 5 | 2^16.0 | — | 1 | None | 3 |
| 1 | `0111` | 5 | dim K = 5 | 2^16.0 | — | 1 | None | 3 |
| 1 | `1111` | 8 | dim K = 8 + plain rows | 2^16.0 | — | 1 | None | 3 |
| 2 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 15 |
| 2 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 9 |
| 2 | `0010` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 6 |
| 2 | `0001` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 24 |
| 2 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 15 |
| 2 | `1010` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 15 |
| 2 | `1001` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 24 |
| 2 | `0110` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 9 |
| 2 | `0101` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 24 |
| 2 | `0011` | 5 | dim K = 5 | 2^16.0 | — | 1 | None | 24 |
| 2 | `1110` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 15 |
| 2 | `1101` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 24 |
| 2 | `1011` | 5 | dim K = 5 | 2^16.0 | — | 1 | None | 24 |
| 2 | `0111` | 5 | dim K = 5 | 2^16.0 | — | 1 | None | 24 |
| 2 | `1111` | 8 | dim K = 8 + plain rows | 2^16.0 | — | 1 | None | 24 |
| 3 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 120 |
| 3 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 72 |
| 3 | `0010` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 48 |
| 3 | `0001` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 192 |
| 3 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 120 |
| 3 | `1010` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 120 |
| 3 | `1001` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 192 |
| 3 | `0110` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 72 |
| 3 | `0101` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 192 |
| 3 | `0011` | 5 | dim K = 5 | 2^16.0 | — | 1 | None | 192 |
| 3 | `1110` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 120 |
| 3 | `1101` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 192 |
| 3 | `1011` | 5 | dim K = 5 | 2^16.0 | — | 1 | None | 192 |
| 3 | `0111` | 5 | dim K = 5 | 2^16.0 | — | 1 | None | 192 |
| 3 | `1111` | 8 | dim K = 8 + plain rows | 2^16.0 | — | 1 | None | 192 |
| 4 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 960 |
| 4 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 576 |
| 4 | `0010` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 384 |
| 4 | `0001` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 1536 |
| 4 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 960 |
| 4 | `1010` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 960 |
| 4 | `1001` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 1536 |
| 4 | `0110` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 576 |
| 4 | `0101` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 1536 |
| 4 | `0011` | 5 | dim K = 5 | 2^16.0 | — | 1 | None | 1536 |
| 4 | `1110` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 960 |
| 4 | `1101` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 1536 |
| 4 | `1011` | 5 | dim K = 5 | 2^16.0 | — | 1 | None | 1536 |
| 4 | `0111` | 5 | dim K = 5 | 2^16.0 | — | 1 | None | 1536 |
| 4 | `1111` | 8 | dim K = 8 + plain rows | 2^16.0 | — | 1 | None | 1536 |
| 5 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 7680 |
| 5 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 4608 |
| 5 | `0010` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 3072 |
| 5 | `0001` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 12288 |
| 5 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 7680 |
| 5 | `1010` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 7680 |
| 5 | `1001` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 12288 |
| 5 | `0110` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 4608 |
| 5 | `0101` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 12288 |
| 5 | `0011` | 5 | dim K = 5 | 2^16.0 | — | 1 | None | 12288 |
| 5 | `1110` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 7680 |
| 5 | `1101` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 12288 |
| 5 | `1011` | 5 | dim K = 5 | 2^16.0 | — | 1 | None | 12288 |
| 5 | `0111` | 5 | dim K = 5 | 2^16.0 | — | 1 | None | 12288 |
| 5 | `1111` | 8 | dim K = 8 + plain rows | 2^16.0 | — | 1 | None | 12288 |
| 6 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 61440 |
| 6 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 36864 |
| 6 | `0010` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 24576 |
| 6 | `0001` | 4 | dim K = 4 | 2^32.0 | — | 2 | None | 98304 |
| 6 | `1100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 61440 |
| 6 | `1010` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 61440 |
| 6 | `1001` | 4 | dim K = 4 | 2^32.0 | — | 2 | None | 98304 |
| 6 | `0110` | 1 | dim K = 1 | 2^16.0 | — | 1 | None | 36864 |
| 6 | `0101` | 4 | dim K = 4 | 2^32.0 | — | 2 | None | 98304 |
| 6 | `0011` | 5 | dim K = 5 | 2^32.0 | — | 2 | None | 98304 |
| 6 | `1110` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 61440 |
| 6 | `1101` | 4 | dim K = 4 | 2^32.0 | — | 2 | None | 98304 |
| 6 | `1011` | 5 | dim K = 5 | 2^32.0 | — | 2 | None | 98304 |
| 6 | `0111` | 5 | dim K = 5 | 2^32.0 | — | 2 | None | 98304 |
| 6 | `1111` | 8 | dim K = 8 + plain rows | 2^32.0 | — | 2 | None | 98304 |
| 7 | `1000` | 0 | distinguisher only | 2^64.0 | [0] | 0 | None | 163840 |
| 7 | `0100` | 0 | distinguisher only | 2^64.0 | — | 4 | None | 196608 |
| 7 | `0010` | 1 | dim K = 1 | 2^48.0 | — | 3 | None | 131072 |
| 7 | `0001` | 4 | dim K = 4 | 2^208.0 | [0] | 9 | None | 786432 |
| 7 | `1100` | 0 | distinguisher only | 2^64.0 | [0] | 0 | None | 163840 |
| 7 | `1010` | 1 | dim K = 1 | 2^64.0 | [0] | 0 | None | 163840 |
| 7 | `1001` | 4 | dim K = 4 | 2^208.0 | [0] | 9 | None | 786432 |
| 7 | `0110` | 1 | dim K = 1 | 2^64.0 | — | 4 | None | 196608 |
| 7 | `0101` | 4 | dim K = 4 | 2^208.0 | [0] | 9 | None | 786432 |
| 7 | `0011` | 5 | dim K = 5 | 2^208.0 | [0] | 9 | None | 786432 |
| 7 | `1110` | 4 | dim K = 4 | 2^64.0 | [0] | 0 | None | 163840 |
| 7 | `1101` | 4 | dim K = 4 | 2^208.0 | [0] | 9 | None | 786432 |
| 7 | `1011` | 5 | dim K = 5 | 2^208.0 | [0] | 9 | None | 786432 |
| 7 | `0111` | 5 | dim K = 5 | 2^208.0 | [0] | 9 | None | 786432 |
| 7 | `1111` | 8 | dim K = 8 + plain rows | 2^208.0 | [0] | 9 | None | 786432 |

### DuX(2^16) · encryption (CPA)

| Layer | Class | dim K | Usability | Min. data | Free block | Extra words | Dims | Degree needed |
|---|---|---|---|---|---|---|---|---|
| 1 | `1000` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `0100` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 1 |
| 1 | `0010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^1 | — | 1 | [1] | 0 |
| 1 | `0001` | 4 | dim K = 4 | 2^2 | — | 1 | [2] | 2 |
| 1 | `1100` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `1010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^2 | — | 1 | [2] | 2 |
| 1 | `1001` | 4 | dim K = 4 | 2^2 | — | 1 | [2] | 2 |
| 1 | `0110` | 2 | dim K = 2 | 2^2 | — | 1 | [2] | 1 |
| 1 | `0101` | 4 | dim K = 4 | 2^2 | — | 1 | [2] | 2 |
| 1 | `0011` | 5 | dim K = 5 | 2^2 | — | 1 | [2] | 2 |
| 1 | `1110` | 4 | dim K = 4 | 2^2 | — | 1 | [2] | 2 |
| 1 | `1101` | 4 | dim K = 4 | 2^2 | — | 1 | [2] | 2 |
| 1 | `1011` | 5 | dim K = 5 | 2^2 | — | 1 | [2] | 2 |
| 1 | `0111` | 6 | dim K = 6 | 2^2 | — | 1 | [2] | 2 |
| 1 | `1111` | 8 | dim K = 8 + plain rows | 2^2 | — | 1 | [2] | 2 |
| 2 | `1000` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 10 |
| 2 | `0100` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 6 |
| 2 | `0010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^3 | — | 1 | [3] | 6 |
| 2 | `0001` | 4 | dim K = 4 | 2^5 | — | 1 | [5] | 24 |
| 2 | `1100` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 10 |
| 2 | `1010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^4 | — | 1 | [4] | 10 |
| 2 | `1001` | 4 | dim K = 4 | 2^5 | — | 1 | [5] | 24 |
| 2 | `0110` | 2 | dim K = 2 | 2^3 | — | 1 | [3] | 6 |
| 2 | `0101` | 4 | dim K = 4 | 2^5 | — | 1 | [5] | 24 |
| 2 | `0011` | 5 | dim K = 5 | 2^5 | — | 1 | [5] | 24 |
| 2 | `1110` | 4 | dim K = 4 | 2^4 | — | 1 | [4] | 10 |
| 2 | `1101` | 4 | dim K = 4 | 2^5 | — | 1 | [5] | 24 |
| 2 | `1011` | 5 | dim K = 5 | 2^5 | — | 1 | [5] | 24 |
| 2 | `0111` | 6 | dim K = 6 | 2^5 | — | 1 | [5] | 24 |
| 2 | `1111` | 8 | dim K = 8 + plain rows | 2^5 | — | 1 | [5] | 24 |
| 3 | `1000` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 120 |
| 3 | `0100` | 0 | distinguisher only | 2^6 | — | 1 | [6] | 48 |
| 3 | `0010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^6 | — | 1 | [6] | 48 |
| 3 | `0001` | 4 | dim K = 4 | 2^8 | — | 1 | [8] | 192 |
| 3 | `1100` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 120 |
| 3 | `1010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^7 | — | 1 | [7] | 120 |
| 3 | `1001` | 4 | dim K = 4 | 2^8 | — | 1 | [8] | 192 |
| 3 | `0110` | 2 | dim K = 2 | 2^6 | — | 1 | [6] | 48 |
| 3 | `0101` | 4 | dim K = 4 | 2^8 | — | 1 | [8] | 192 |
| 3 | `0011` | 5 | dim K = 5 | 2^8 | — | 1 | [8] | 192 |
| 3 | `1110` | 4 | dim K = 4 | 2^7 | — | 1 | [7] | 120 |
| 3 | `1101` | 4 | dim K = 4 | 2^8 | — | 1 | [8] | 192 |
| 3 | `1011` | 5 | dim K = 5 | 2^8 | — | 1 | [8] | 192 |
| 3 | `0111` | 6 | dim K = 6 | 2^8 | — | 1 | [8] | 192 |
| 3 | `1111` | 8 | dim K = 8 + plain rows | 2^8 | — | 1 | [8] | 192 |
| 4 | `1000` | 0 | distinguisher only | 2^10 | — | 1 | [10] | 960 |
| 4 | `0100` | 0 | distinguisher only | 2^9 | — | 1 | [9] | 384 |
| 4 | `0010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^9 | — | 1 | [9] | 384 |
| 4 | `0001` | 4 | dim K = 4 | 2^11 | — | 1 | [11] | 1536 |
| 4 | `1100` | 0 | distinguisher only | 2^10 | — | 1 | [10] | 960 |
| 4 | `1010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^10 | — | 1 | [10] | 960 |
| 4 | `1001` | 4 | dim K = 4 | 2^11 | — | 1 | [11] | 1536 |
| 4 | `0110` | 2 | dim K = 2 | 2^9 | — | 1 | [9] | 384 |
| 4 | `0101` | 4 | dim K = 4 | 2^11 | — | 1 | [11] | 1536 |
| 4 | `0011` | 5 | dim K = 5 | 2^11 | — | 1 | [11] | 1536 |
| 4 | `1110` | 4 | dim K = 4 | 2^10 | — | 1 | [10] | 960 |
| 4 | `1101` | 4 | dim K = 4 | 2^11 | — | 1 | [11] | 1536 |
| 4 | `1011` | 5 | dim K = 5 | 2^11 | — | 1 | [11] | 1536 |
| 4 | `0111` | 6 | dim K = 6 | 2^11 | — | 1 | [11] | 1536 |
| 4 | `1111` | 8 | dim K = 8 + plain rows | 2^11 | — | 1 | [11] | 1536 |
| 5 | `1000` | 0 | distinguisher only | 2^13 | — | 1 | [13] | 7680 |
| 5 | `0100` | 0 | distinguisher only | 2^12 | — | 1 | [12] | 3072 |
| 5 | `0010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^12 | — | 1 | [12] | 3072 |
| 5 | `0001` | 4 | dim K = 4 | 2^14 | — | 1 | [14] | 12288 |
| 5 | `1100` | 0 | distinguisher only | 2^13 | — | 1 | [13] | 7680 |
| 5 | `1010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^13 | — | 1 | [13] | 7680 |
| 5 | `1001` | 4 | dim K = 4 | 2^14 | — | 1 | [14] | 12288 |
| 5 | `0110` | 2 | dim K = 2 | 2^12 | — | 1 | [12] | 3072 |
| 5 | `0101` | 4 | dim K = 4 | 2^14 | — | 1 | [14] | 12288 |
| 5 | `0011` | 5 | dim K = 5 | 2^14 | — | 1 | [14] | 12288 |
| 5 | `1110` | 4 | dim K = 4 | 2^13 | — | 1 | [13] | 7680 |
| 5 | `1101` | 4 | dim K = 4 | 2^14 | — | 1 | [14] | 12288 |
| 5 | `1011` | 5 | dim K = 5 | 2^14 | — | 1 | [14] | 12288 |
| 5 | `0111` | 6 | dim K = 6 | 2^14 | — | 1 | [14] | 12288 |
| 5 | `1111` | 8 | dim K = 8 + plain rows | 2^14 | — | 1 | [14] | 12288 |
| 6 | `1000` | 0 | distinguisher only | 2^16 | — | 1 | [16] | 61440 |
| 6 | `0100` | 0 | distinguisher only | 2^15 | — | 1 | [15] | 24576 |
| 6 | `0010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^15 | — | 1 | [15] | 24576 |
| 6 | `0001` | 4 | dim K = 4 | 2^18 | — | 2 | [16, 2] | 65536 |
| 6 | `1100` | 0 | distinguisher only | 2^16 | — | 1 | [16] | 61440 |
| 6 | `1010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^16 | — | 1 | [16] | 61440 |
| 6 | `1001` | 4 | dim K = 4 | 2^18 | — | 2 | [16, 2] | 65536 |
| 6 | `0110` | 2 | dim K = 2 | 2^15 | — | 1 | [15] | 24576 |
| 6 | `0101` | 4 | dim K = 4 | 2^18 | — | 2 | [16, 2] | 65536 |
| 6 | `0011` | 5 | dim K = 5 | 2^18 | — | 2 | [16, 2] | 65536 |
| 6 | `1110` | 4 | dim K = 4 | 2^16 | — | 1 | [16] | 61440 |
| 6 | `1101` | 4 | dim K = 4 | 2^18 | — | 2 | [16, 2] | 65536 |
| 6 | `1011` | 5 | dim K = 5 | 2^18 | — | 2 | [16, 2] | 65536 |
| 6 | `0111` | 6 | dim K = 6 | 2^18 | — | 2 | [16, 2] | 65536 |
| 6 | `1111` | 8 | dim K = 8 + plain rows | 2^18 | — | 2 | [16, 2] | 65536 |
| 7 | `1000` | 0 | distinguisher only | 2^64 | [0] | 0 | [] | 163840 |
| 7 | `0100` | 0 | distinguisher only | 2^51 | — | 4 | [16, 16, 16, 3] | 196608 |
| 7 | `0010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^34 | — | 3 | [16, 16, 2] | 131072 |
| 7 | `0001` | 4 | dim K = 4 | 2^196 | [0] | 9 | [16, 16, 16, 16, 16, 16, 16, 16, 4] | 786432 |
| 7 | `1100` | 0 | distinguisher only | 2^64 | [0] | 0 | [] | 163840 |
| 7 | `1010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^64 | [0] | 0 | [] | 163840 |
| 7 | `1001` | 4 | dim K = 4 | 2^196 | [0] | 9 | [16, 16, 16, 16, 16, 16, 16, 16, 4] | 786432 |
| 7 | `0110` | 2 | dim K = 2 | 2^51 | — | 4 | [16, 16, 16, 3] | 196608 |
| 7 | `0101` | 4 | dim K = 4 | 2^196 | [0] | 9 | [16, 16, 16, 16, 16, 16, 16, 16, 4] | 786432 |
| 7 | `0011` | 5 | dim K = 5 | 2^196 | [0] | 9 | [16, 16, 16, 16, 16, 16, 16, 16, 4] | 786432 |
| 7 | `1110` | 4 | dim K = 4 | 2^64 | [0] | 0 | [] | 163840 |
| 7 | `1101` | 4 | dim K = 4 | 2^196 | [0] | 9 | [16, 16, 16, 16, 16, 16, 16, 16, 4] | 786432 |
| 7 | `1011` | 5 | dim K = 5 | 2^196 | [0] | 9 | [16, 16, 16, 16, 16, 16, 16, 16, 4] | 786432 |
| 7 | `0111` | 6 | dim K = 6 | 2^196 | [0] | 9 | [16, 16, 16, 16, 16, 16, 16, 16, 4] | 786432 |
| 7 | `1111` | 8 | dim K = 8 + plain rows | 2^196 | [0] | 9 | [16, 16, 16, 16, 16, 16, 16, 16, 4] | 786432 |

### DuX(2^8) · encryption (CPA)

| Layer | Class | dim K | Usability | Min. data | Free block | Extra words | Dims | Degree needed |
|---|---|---|---|---|---|---|---|---|
| 1 | `1000` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `0100` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 1 |
| 1 | `0010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^1 | — | 1 | [1] | 0 |
| 1 | `0001` | 4 | dim K = 4 | 2^2 | — | 1 | [2] | 2 |
| 1 | `1100` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `1010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^2 | — | 1 | [2] | 2 |
| 1 | `1001` | 4 | dim K = 4 | 2^2 | — | 1 | [2] | 2 |
| 1 | `0110` | 2 | dim K = 2 | 2^2 | — | 1 | [2] | 1 |
| 1 | `0101` | 4 | dim K = 4 | 2^2 | — | 1 | [2] | 2 |
| 1 | `0011` | 5 | dim K = 5 | 2^2 | — | 1 | [2] | 2 |
| 1 | `1110` | 4 | dim K = 4 | 2^2 | — | 1 | [2] | 2 |
| 1 | `1101` | 4 | dim K = 4 | 2^2 | — | 1 | [2] | 2 |
| 1 | `1011` | 5 | dim K = 5 | 2^2 | — | 1 | [2] | 2 |
| 1 | `0111` | 6 | dim K = 6 | 2^2 | — | 1 | [2] | 2 |
| 1 | `1111` | 8 | dim K = 8 + plain rows | 2^2 | — | 1 | [2] | 2 |
| 2 | `1000` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 10 |
| 2 | `0100` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 6 |
| 2 | `0010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^3 | — | 1 | [3] | 6 |
| 2 | `0001` | 4 | dim K = 4 | 2^5 | — | 1 | [5] | 24 |
| 2 | `1100` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 10 |
| 2 | `1010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^4 | — | 1 | [4] | 10 |
| 2 | `1001` | 4 | dim K = 4 | 2^5 | — | 1 | [5] | 24 |
| 2 | `0110` | 2 | dim K = 2 | 2^3 | — | 1 | [3] | 6 |
| 2 | `0101` | 4 | dim K = 4 | 2^5 | — | 1 | [5] | 24 |
| 2 | `0011` | 5 | dim K = 5 | 2^5 | — | 1 | [5] | 24 |
| 2 | `1110` | 4 | dim K = 4 | 2^4 | — | 1 | [4] | 10 |
| 2 | `1101` | 4 | dim K = 4 | 2^5 | — | 1 | [5] | 24 |
| 2 | `1011` | 5 | dim K = 5 | 2^5 | — | 1 | [5] | 24 |
| 2 | `0111` | 6 | dim K = 6 | 2^5 | — | 1 | [5] | 24 |
| 2 | `1111` | 8 | dim K = 8 + plain rows | 2^5 | — | 1 | [5] | 24 |
| 3 | `1000` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 120 |
| 3 | `0100` | 0 | distinguisher only | 2^6 | — | 1 | [6] | 48 |
| 3 | `0010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^6 | — | 1 | [6] | 48 |
| 3 | `0001` | 4 | dim K = 4 | 2^8 | — | 1 | [8] | 192 |
| 3 | `1100` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 120 |
| 3 | `1010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^7 | — | 1 | [7] | 120 |
| 3 | `1001` | 4 | dim K = 4 | 2^8 | — | 1 | [8] | 192 |
| 3 | `0110` | 2 | dim K = 2 | 2^6 | — | 1 | [6] | 48 |
| 3 | `0101` | 4 | dim K = 4 | 2^8 | — | 1 | [8] | 192 |
| 3 | `0011` | 5 | dim K = 5 | 2^8 | — | 1 | [8] | 192 |
| 3 | `1110` | 4 | dim K = 4 | 2^7 | — | 1 | [7] | 120 |
| 3 | `1101` | 4 | dim K = 4 | 2^8 | — | 1 | [8] | 192 |
| 3 | `1011` | 5 | dim K = 5 | 2^8 | — | 1 | [8] | 192 |
| 3 | `0111` | 6 | dim K = 6 | 2^8 | — | 1 | [8] | 192 |
| 3 | `1111` | 8 | dim K = 8 + plain rows | 2^8 | — | 1 | [8] | 192 |
| 4 | `1000` | 0 | distinguisher only | 2^24 | — | 3 | [8, 8, 8] | 640 |
| 4 | `0100` | 0 | distinguisher only | 2^16 | — | 2 | [8, 8] | 384 |
| 4 | `0010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^10 | — | 2 | [8, 2] | 256 |
| 4 | `0001` | 4 | dim K = 4 | 2^32 | [0] | 0 | [] | 512 |
| 4 | `1100` | 0 | distinguisher only | 2^24 | — | 3 | [8, 8, 8] | 640 |
| 4 | `1010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^24 | — | 3 | [8, 8, 8] | 640 |
| 4 | `1001` | 4 | dim K = 4 | 2^32 | [0] | 0 | [] | 512 |
| 4 | `0110` | 2 | dim K = 2 | 2^16 | — | 2 | [8, 8] | 384 |
| 4 | `0101` | 4 | dim K = 4 | 2^32 | [0] | 0 | [] | 512 |
| 4 | `0011` | 5 | dim K = 5 | 2^32 | [0] | 0 | [] | 512 |
| 4 | `1110` | 4 | dim K = 4 | 2^24 | — | 3 | [8, 8, 8] | 640 |
| 4 | `1101` | 4 | dim K = 4 | 2^32 | [0] | 0 | [] | 512 |
| 4 | `1011` | 5 | dim K = 5 | 2^32 | [0] | 0 | [] | 512 |
| 4 | `0111` | 6 | dim K = 6 | 2^32 | [0] | 0 | [] | 512 |
| 4 | `1111` | 8 | dim K = 8 + plain rows | 2^32 | [0] | 0 | [] | 512 |
| 5 | `0010` | 1 | char-2 unusable (dim K = 1 combined row collapses in characteristic 2) | 2^100 | [0] | 9 | [8, 8, 8, 8, 8, 8, 8, 8, 4] | 3072 |

### YupX-65537 · encryption (CPA)

| Layer | Class | dim K | Usability | Min. data | Free block | Extra words | Dims | Degree needed |
|---|---|---|---|---|---|---|---|---|
| 1 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 5 |
| 1 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 3 |
| 1 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 2 |
| 1 | `0001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1 |
| 1 | `1100` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 5 |
| 1 | `1010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 5 |
| 1 | `1001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 5 |
| 1 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 3 |
| 1 | `0101` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 3 |
| 1 | `0011` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 2 |
| 1 | `1110` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 5 |
| 1 | `1101` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 5 |
| 1 | `1011` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 5 |
| 1 | `0111` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 3 |
| 1 | `1111` | 8 | dim K = 8 + plain rows | 2^16.0 | — | 1 | None | 5 |
| 2 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 40 |
| 2 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 25 |
| 2 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 15 |
| 2 | `0001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 10 |
| 2 | `1100` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 40 |
| 2 | `1010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 40 |
| 2 | `1001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 40 |
| 2 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 25 |
| 2 | `0101` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 25 |
| 2 | `0011` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 15 |
| 2 | `1110` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 40 |
| 2 | `1101` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 40 |
| 2 | `1011` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 40 |
| 2 | `0111` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 25 |
| 2 | `1111` | 8 | dim K = 8 + plain rows | 2^16.0 | — | 1 | None | 40 |
| 3 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 320 |
| 3 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 200 |
| 3 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 120 |
| 3 | `0001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 80 |
| 3 | `1100` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 320 |
| 3 | `1010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 320 |
| 3 | `1001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 320 |
| 3 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 200 |
| 3 | `0101` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 200 |
| 3 | `0011` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 120 |
| 3 | `1110` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 320 |
| 3 | `1101` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 320 |
| 3 | `1011` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 320 |
| 3 | `0111` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 200 |
| 3 | `1111` | 8 | dim K = 8 + plain rows | 2^16.0 | — | 1 | None | 320 |
| 4 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 2560 |
| 4 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1600 |
| 4 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 960 |
| 4 | `0001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 640 |
| 4 | `1100` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 2560 |
| 4 | `1010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 2560 |
| 4 | `1001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 2560 |
| 4 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1600 |
| 4 | `0101` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 1600 |
| 4 | `0011` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 960 |
| 4 | `1110` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 2560 |
| 4 | `1101` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 2560 |
| 4 | `1011` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 2560 |
| 4 | `0111` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 1600 |
| 4 | `1111` | 8 | dim K = 8 + plain rows | 2^16.0 | — | 1 | None | 2560 |
| 5 | `1000` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 20480 |
| 5 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 12800 |
| 5 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 7680 |
| 5 | `0001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 5120 |
| 5 | `1100` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 20480 |
| 5 | `1010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 20480 |
| 5 | `1001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 20480 |
| 5 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 12800 |
| 5 | `0101` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 12800 |
| 5 | `0011` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 7680 |
| 5 | `1110` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 20480 |
| 5 | `1101` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 20480 |
| 5 | `1011` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 20480 |
| 5 | `0111` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 12800 |
| 5 | `1111` | 8 | dim K = 8 + plain rows | 2^16.0 | — | 1 | None | 20480 |
| 6 | `1000` | 0 | distinguisher only | 2^32.0 | — | 2 | None | 98304 |
| 6 | `0100` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 61440 |
| 6 | `0010` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 61440 |
| 6 | `0001` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 40960 |
| 6 | `1100` | 4 | dim K = 4 | 2^32.0 | — | 2 | None | 98304 |
| 6 | `1010` | 0 | distinguisher only | 2^32.0 | — | 2 | None | 98304 |
| 6 | `1001` | 0 | distinguisher only | 2^32.0 | — | 2 | None | 98304 |
| 6 | `0110` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 61440 |
| 6 | `0101` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 61440 |
| 6 | `0011` | 0 | distinguisher only | 2^16.0 | — | 1 | None | 61440 |
| 6 | `1110` | 4 | dim K = 4 | 2^32.0 | — | 2 | None | 98304 |
| 6 | `1101` | 4 | dim K = 4 | 2^32.0 | — | 2 | None | 98304 |
| 6 | `1011` | 4 | dim K = 4 | 2^32.0 | — | 2 | None | 98304 |
| 6 | `0111` | 4 | dim K = 4 | 2^16.0 | — | 1 | None | 61440 |
| 6 | `1111` | 8 | dim K = 8 + plain rows | 2^32.0 | — | 2 | None | 98304 |
| 7 | `1000` | 0 | distinguisher only | 2^208.0 | [0] | 9 | None | 786432 |
| 7 | `0100` | 0 | distinguisher only | 2^64.0 | [0] | 0 | None | 163840 |
| 7 | `0010` | 0 | distinguisher only | 2^64.0 | [0] | 0 | None | 98304 |
| 7 | `0001` | 0 | distinguisher only | 2^64.0 | — | 4 | None | 196608 |
| 7 | `1100` | 4 | dim K = 4 | 2^208.0 | [0] | 9 | None | 786432 |
| 7 | `1010` | 0 | distinguisher only | 2^208.0 | [0] | 9 | None | 786432 |
| 7 | `1001` | 0 | distinguisher only | 2^208.0 | [0] | 9 | None | 786432 |
| 7 | `0110` | 0 | distinguisher only | 2^64.0 | [0] | 0 | None | 163840 |
| 7 | `0101` | 0 | distinguisher only | 2^64.0 | [0] | 0 | None | 163840 |
| 7 | `0011` | 0 | distinguisher only | 2^64.0 | [0] | 0 | None | 98304 |
| 7 | `1110` | 4 | dim K = 4 | 2^208.0 | [0] | 9 | None | 786432 |
| 7 | `1101` | 4 | dim K = 4 | 2^208.0 | [0] | 9 | None | 786432 |
| 7 | `1011` | 4 | dim K = 4 | 2^208.0 | [0] | 9 | None | 786432 |
| 7 | `0111` | 4 | dim K = 4 | 2^64.0 | [0] | 0 | None | 163840 |
| 7 | `1111` | 8 | dim K = 8 + plain rows | 2^208.0 | [0] | 9 | None | 786432 |

### Yu2X-16 · encryption (CPA)

| Layer | Class | dim K | Usability | Min. data | Free block | Extra words | Dims | Degree needed |
|---|---|---|---|---|---|---|---|---|
| 1 | `1000` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 5 |
| 1 | `0100` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `0010` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `0001` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 1 |
| 1 | `1100` | 4 | dim K = 4 | 2^3 | — | 1 | [3] | 5 |
| 1 | `1010` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 5 |
| 1 | `1001` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 5 |
| 1 | `0110` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `0101` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `0011` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `1110` | 4 | dim K = 4 | 2^3 | — | 1 | [3] | 5 |
| 1 | `1101` | 4 | dim K = 4 | 2^3 | — | 1 | [3] | 5 |
| 1 | `1011` | 4 | dim K = 4 | 2^3 | — | 1 | [3] | 5 |
| 1 | `0111` | 4 | dim K = 4 | 2^2 | — | 1 | [2] | 2 |
| 1 | `1111` | 8 | dim K = 8 + plain rows | 2^3 | — | 1 | [3] | 5 |
| 2 | `1000` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 24 |
| 2 | `0100` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 25 |
| 2 | `0010` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 9 |
| 2 | `0001` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 6 |
| 2 | `1100` | 4 | dim K = 4 | 2^5 | — | 1 | [5] | 24 |
| 2 | `1010` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 24 |
| 2 | `1001` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 24 |
| 2 | `0110` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 25 |
| 2 | `0101` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 25 |
| 2 | `0011` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 9 |
| 2 | `1110` | 4 | dim K = 4 | 2^5 | — | 1 | [5] | 24 |
| 2 | `1101` | 4 | dim K = 4 | 2^5 | — | 1 | [5] | 24 |
| 2 | `1011` | 4 | dim K = 4 | 2^5 | — | 1 | [5] | 24 |
| 2 | `0111` | 4 | dim K = 4 | 2^5 | — | 1 | [5] | 25 |
| 2 | `1111` | 8 | dim K = 8 + plain rows | 2^5 | — | 1 | [5] | 24 |
| 3 | `1000` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 192 |
| 3 | `0100` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 120 |
| 3 | `0010` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 120 |
| 3 | `0001` | 0 | distinguisher only | 2^6 | — | 1 | [6] | 48 |
| 3 | `1100` | 4 | dim K = 4 | 2^8 | — | 1 | [8] | 192 |
| 3 | `1010` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 192 |
| 3 | `1001` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 192 |
| 3 | `0110` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 120 |
| 3 | `0101` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 120 |
| 3 | `0011` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 120 |
| 3 | `1110` | 4 | dim K = 4 | 2^8 | — | 1 | [8] | 192 |
| 3 | `1101` | 4 | dim K = 4 | 2^8 | — | 1 | [8] | 192 |
| 3 | `1011` | 4 | dim K = 4 | 2^8 | — | 1 | [8] | 192 |
| 3 | `0111` | 4 | dim K = 4 | 2^7 | — | 1 | [7] | 120 |
| 3 | `1111` | 8 | dim K = 8 + plain rows | 2^8 | — | 1 | [8] | 192 |
| 4 | `1000` | 0 | distinguisher only | 2^11 | — | 1 | [11] | 1536 |
| 4 | `0100` | 0 | distinguisher only | 2^10 | — | 1 | [10] | 960 |
| 4 | `0010` | 0 | distinguisher only | 2^10 | — | 1 | [10] | 960 |
| 4 | `0001` | 0 | distinguisher only | 2^9 | — | 1 | [9] | 384 |
| 4 | `1100` | 4 | dim K = 4 | 2^11 | — | 1 | [11] | 1536 |
| 4 | `1010` | 0 | distinguisher only | 2^11 | — | 1 | [11] | 1536 |
| 4 | `1001` | 0 | distinguisher only | 2^11 | — | 1 | [11] | 1536 |
| 4 | `0110` | 0 | distinguisher only | 2^10 | — | 1 | [10] | 960 |
| 4 | `0101` | 0 | distinguisher only | 2^10 | — | 1 | [10] | 960 |
| 4 | `0011` | 0 | distinguisher only | 2^10 | — | 1 | [10] | 960 |
| 4 | `1110` | 4 | dim K = 4 | 2^11 | — | 1 | [11] | 1536 |
| 4 | `1101` | 4 | dim K = 4 | 2^11 | — | 1 | [11] | 1536 |
| 4 | `1011` | 4 | dim K = 4 | 2^11 | — | 1 | [11] | 1536 |
| 4 | `0111` | 4 | dim K = 4 | 2^10 | — | 1 | [10] | 960 |
| 4 | `1111` | 8 | dim K = 8 + plain rows | 2^11 | — | 1 | [11] | 1536 |
| 5 | `1000` | 0 | distinguisher only | 2^14 | — | 1 | [14] | 12288 |
| 5 | `0100` | 0 | distinguisher only | 2^13 | — | 1 | [13] | 7680 |
| 5 | `0010` | 0 | distinguisher only | 2^13 | — | 1 | [13] | 7680 |
| 5 | `0001` | 0 | distinguisher only | 2^12 | — | 1 | [12] | 3072 |
| 5 | `1100` | 4 | dim K = 4 | 2^14 | — | 1 | [14] | 12288 |
| 5 | `1010` | 0 | distinguisher only | 2^14 | — | 1 | [14] | 12288 |
| 5 | `1001` | 0 | distinguisher only | 2^14 | — | 1 | [14] | 12288 |
| 5 | `0110` | 0 | distinguisher only | 2^13 | — | 1 | [13] | 7680 |
| 5 | `0101` | 0 | distinguisher only | 2^13 | — | 1 | [13] | 7680 |
| 5 | `0011` | 0 | distinguisher only | 2^13 | — | 1 | [13] | 7680 |
| 5 | `1110` | 4 | dim K = 4 | 2^14 | — | 1 | [14] | 12288 |
| 5 | `1101` | 4 | dim K = 4 | 2^14 | — | 1 | [14] | 12288 |
| 5 | `1011` | 4 | dim K = 4 | 2^14 | — | 1 | [14] | 12288 |
| 5 | `0111` | 4 | dim K = 4 | 2^13 | — | 1 | [13] | 7680 |
| 5 | `1111` | 8 | dim K = 8 + plain rows | 2^14 | — | 1 | [14] | 12288 |
| 6 | `1000` | 0 | distinguisher only | 2^32 | — | 2 | [16, 16] | 98304 |
| 6 | `0100` | 0 | distinguisher only | 2^16 | — | 1 | [16] | 61440 |
| 6 | `0010` | 0 | distinguisher only | 2^16 | — | 1 | [16] | 61440 |
| 6 | `0001` | 0 | distinguisher only | 2^15 | — | 1 | [15] | 24576 |
| 6 | `1100` | 4 | dim K = 4 | 2^32 | — | 2 | [16, 16] | 98304 |
| 6 | `1010` | 0 | distinguisher only | 2^32 | — | 2 | [16, 16] | 98304 |
| 6 | `1001` | 0 | distinguisher only | 2^32 | — | 2 | [16, 16] | 98304 |
| 6 | `0110` | 0 | distinguisher only | 2^16 | — | 1 | [16] | 61440 |
| 6 | `0101` | 0 | distinguisher only | 2^16 | — | 1 | [16] | 61440 |
| 6 | `0011` | 0 | distinguisher only | 2^16 | — | 1 | [16] | 61440 |
| 6 | `1110` | 4 | dim K = 4 | 2^32 | — | 2 | [16, 16] | 98304 |
| 6 | `1101` | 4 | dim K = 4 | 2^32 | — | 2 | [16, 16] | 98304 |
| 6 | `1011` | 4 | dim K = 4 | 2^32 | — | 2 | [16, 16] | 98304 |
| 6 | `0111` | 4 | dim K = 4 | 2^16 | — | 1 | [16] | 61440 |
| 6 | `1111` | 8 | dim K = 8 + plain rows | 2^32 | — | 2 | [16, 16] | 98304 |
| 7 | `1000` | 0 | distinguisher only | 2^196 | [0] | 9 | [16, 16, 16, 16, 16, 16, 16, 16, 4] | 786432 |
| 7 | `0100` | 0 | distinguisher only | 2^64 | [0] | 0 | [] | 163840 |
| 7 | `0010` | 0 | distinguisher only | 2^64 | [0] | 0 | [] | 98304 |
| 7 | `0001` | 0 | distinguisher only | 2^51 | — | 4 | [16, 16, 16, 3] | 196608 |
| 7 | `1100` | 4 | dim K = 4 | 2^196 | [0] | 9 | [16, 16, 16, 16, 16, 16, 16, 16, 4] | 786432 |
| 7 | `1010` | 0 | distinguisher only | 2^196 | [0] | 9 | [16, 16, 16, 16, 16, 16, 16, 16, 4] | 786432 |
| 7 | `1001` | 0 | distinguisher only | 2^196 | [0] | 9 | [16, 16, 16, 16, 16, 16, 16, 16, 4] | 786432 |
| 7 | `0110` | 0 | distinguisher only | 2^64 | [0] | 0 | [] | 163840 |
| 7 | `0101` | 0 | distinguisher only | 2^64 | [0] | 0 | [] | 163840 |
| 7 | `0011` | 0 | distinguisher only | 2^64 | [0] | 0 | [] | 98304 |
| 7 | `1110` | 4 | dim K = 4 | 2^196 | [0] | 9 | [16, 16, 16, 16, 16, 16, 16, 16, 4] | 786432 |
| 7 | `1101` | 4 | dim K = 4 | 2^196 | [0] | 9 | [16, 16, 16, 16, 16, 16, 16, 16, 4] | 786432 |
| 7 | `1011` | 4 | dim K = 4 | 2^196 | [0] | 9 | [16, 16, 16, 16, 16, 16, 16, 16, 4] | 786432 |
| 7 | `0111` | 4 | dim K = 4 | 2^64 | [0] | 0 | [] | 163840 |
| 7 | `1111` | 8 | dim K = 8 + plain rows | 2^196 | [0] | 9 | [16, 16, 16, 16, 16, 16, 16, 16, 4] | 786432 |

### Yu2X-8 · encryption (CPA)

| Layer | Class | dim K | Usability | Min. data | Free block | Extra words | Dims | Degree needed |
|---|---|---|---|---|---|---|---|---|
| 1 | `1000` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 5 |
| 1 | `0100` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `0010` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `0001` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 1 |
| 1 | `1100` | 4 | dim K = 4 | 2^3 | — | 1 | [3] | 5 |
| 1 | `1010` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 5 |
| 1 | `1001` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 5 |
| 1 | `0110` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `0101` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `0011` | 0 | distinguisher only | 2^2 | — | 1 | [2] | 2 |
| 1 | `1110` | 4 | dim K = 4 | 2^3 | — | 1 | [3] | 5 |
| 1 | `1101` | 4 | dim K = 4 | 2^3 | — | 1 | [3] | 5 |
| 1 | `1011` | 4 | dim K = 4 | 2^3 | — | 1 | [3] | 5 |
| 1 | `0111` | 4 | dim K = 4 | 2^2 | — | 1 | [2] | 2 |
| 1 | `1111` | 8 | dim K = 8 + plain rows | 2^3 | — | 1 | [3] | 5 |
| 2 | `1000` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 24 |
| 2 | `0100` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 25 |
| 2 | `0010` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 9 |
| 2 | `0001` | 0 | distinguisher only | 2^3 | — | 1 | [3] | 6 |
| 2 | `1100` | 4 | dim K = 4 | 2^5 | — | 1 | [5] | 24 |
| 2 | `1010` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 24 |
| 2 | `1001` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 24 |
| 2 | `0110` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 25 |
| 2 | `0101` | 0 | distinguisher only | 2^5 | — | 1 | [5] | 25 |
| 2 | `0011` | 0 | distinguisher only | 2^4 | — | 1 | [4] | 9 |
| 2 | `1110` | 4 | dim K = 4 | 2^5 | — | 1 | [5] | 24 |
| 2 | `1101` | 4 | dim K = 4 | 2^5 | — | 1 | [5] | 24 |
| 2 | `1011` | 4 | dim K = 4 | 2^5 | — | 1 | [5] | 24 |
| 2 | `0111` | 4 | dim K = 4 | 2^5 | — | 1 | [5] | 25 |
| 2 | `1111` | 8 | dim K = 8 + plain rows | 2^5 | — | 1 | [5] | 24 |
| 3 | `1000` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 192 |
| 3 | `0100` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 120 |
| 3 | `0010` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 120 |
| 3 | `0001` | 0 | distinguisher only | 2^6 | — | 1 | [6] | 48 |
| 3 | `1100` | 4 | dim K = 4 | 2^8 | — | 1 | [8] | 192 |
| 3 | `1010` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 192 |
| 3 | `1001` | 0 | distinguisher only | 2^8 | — | 1 | [8] | 192 |
| 3 | `0110` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 120 |
| 3 | `0101` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 120 |
| 3 | `0011` | 0 | distinguisher only | 2^7 | — | 1 | [7] | 120 |
| 3 | `1110` | 4 | dim K = 4 | 2^8 | — | 1 | [8] | 192 |
| 3 | `1101` | 4 | dim K = 4 | 2^8 | — | 1 | [8] | 192 |
| 3 | `1011` | 4 | dim K = 4 | 2^8 | — | 1 | [8] | 192 |
| 3 | `0111` | 4 | dim K = 4 | 2^7 | — | 1 | [7] | 120 |
| 3 | `1111` | 8 | dim K = 8 + plain rows | 2^8 | — | 1 | [8] | 192 |
| 4 | `1000` | 0 | distinguisher only | 2^32 | [0] | 0 | [] | 512 |
| 4 | `0100` | 0 | distinguisher only | 2^32 | — | 4 | [8, 8, 8, 8] | 960 |
| 4 | `0010` | 0 | distinguisher only | 2^23 | — | 3 | [8, 8, 7] | 576 |
| 4 | `0001` | 0 | distinguisher only | 2^16 | — | 2 | [8, 8] | 384 |
| 4 | `1100` | 4 | dim K = 4 | 2^32 | [0] | 0 | [] | 512 |
| 4 | `1010` | 0 | distinguisher only | 2^32 | [0] | 0 | [] | 512 |
| 4 | `1001` | 0 | distinguisher only | 2^32 | [0] | 0 | [] | 512 |
| 4 | `0110` | 0 | distinguisher only | 2^32 | — | 4 | [8, 8, 8, 8] | 960 |
| 4 | `0101` | 0 | distinguisher only | 2^32 | — | 4 | [8, 8, 8, 8] | 960 |
| 4 | `0011` | 0 | distinguisher only | 2^23 | — | 3 | [8, 8, 7] | 576 |
| 4 | `1110` | 4 | dim K = 4 | 2^32 | [0] | 0 | [] | 512 |
| 4 | `1101` | 4 | dim K = 4 | 2^32 | [0] | 0 | [] | 512 |
| 4 | `1011` | 4 | dim K = 4 | 2^32 | [0] | 0 | [] | 512 |
| 4 | `0111` | 4 | dim K = 4 | 2^32 | — | 4 | [8, 8, 8, 8] | 960 |
| 4 | `1111` | 8 | dim K = 8 + plain rows | 2^32 | [0] | 0 | [] | 512 |
| 5 | `0001` | 0 | distinguisher only | 2^100 | [0] | 9 | [8, 8, 8, 8, 8, 8, 8, 8, 4] | 3072 |

