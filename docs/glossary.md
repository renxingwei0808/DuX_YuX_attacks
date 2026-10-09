# Glossary of labels used in the code and the records

The code, the test names and the result files use short internal labels.  This
page maps them to the statements of the paper and its supplement.

## Observations and techniques

| Label | Meaning | In the paper |
|---|---|---|
| O2 | The key-dependent linear layer is transparent: `L1 = Rot_{-4} o L0`, so DuX is equivalent to a cipher with a fixed L0, re-labelled round keys and a block rotation of the output | Proposition "transparency of the key-dependent linear layer" |
| O3 | DuX's decryption layer `L^-1` mixes only positions {p, p+1} of a block, so position 3 lags; growth 2 + sqrt(3) per layer (YuX: 4 per layer) | Proposition "positional lag in DuX" |
| O5 | Summing over a whole structure removes the pure-key terms, so one extension round is linear algebra | Lemma "vanishing of the pure-key terms" |
| O5' | In characteristic 2 the DuX key schedule has `2 x0 = 0`, so `y1 + y2 = x0` | `docs/dux_specification.md` |
| O6 | Boolean degree of the two-round extension: 7, not 6 (experiment E08) | Supplement, "Boolean degree of the two-round extension" |
| O7 | The unified zero-sum criterion: a word whose formal degree D is below the threshold T of the structure sums to zero | Theorem "unified zero-sum criterion" |
| O8, O8(c) | Which key words a one-round equation determines (O8), and the template bound `rank_max <= rank(Phi on V_0)` measured on pseudo-structures (O8(c)) | Proposition "key dependence", Proposition "template bound" |
| O8' | Empirical rule: the minimal number of structures is the rank saturation point | Section IV, "From equations to the master key" |
| O9 | Encryption direction: the next-to-leading coefficient vanishes, so cells with D = q are still balanced | Theorem "vanishing next-to-leading coefficient" (supplement) |
| O10 | Cheap coordinate elimination: the left kernel K of the expensive columns gives dim K rows that contain only the cheap coordinate | Lemma "cheap coordinates" |
| O10, char-2 collapse | In characteristic 2 a combined row with dim K = 1 collapses to orbit sums and cannot determine the key | Lemma "block equivariance", Theorem "characteristic-2 collapse" |
| O11 | Full-block structures: the first `S^-1` layer is free when a whole block runs through F_q^4 | Lemma "full block" |
| O12 | Weighted moments: `sum_x x^a Z(x) = 0` for `|a| < T - D` | Theorem "weighted moments" |
| O13 | In characteristic 2 the exact degree of YuX is below the max-plus bound, independently of the key | Lemma "layer-2 collapse of YuX in characteristic 2" (supplement) |
| O14, O14-CPA | Prime-field full blocks at the boundary D = T sum to a key-independent constant (decryption), and to zero for YuX in the encryption direction | Propositions "key-independent constants at the boundary" and "boundary cells in the encryption direction" (supplement) |
| O15 | Point sets with the divided-difference mask: the criterion holds with `T' = sum(|U_i| - 1)` | Proposition "point sets" |
| O16 | In characteristic 2 the cubic coordinate b is a second cheap coordinate (cheap set {1, 2}) | Lemma "cubic coordinate" |
| O17 | Mixed structures (full-field words times multiplicative cosets): the thresholds add up, for weighted sums | Proposition "mixed structures" |
| O18 | Multi-index weights `a in N^s`, joint Vandermonde over several axes | Theorem "weighted moments" (multi-index form) |
| T1 | Technique: the cheap set {1, 2} in characteristic 2 (O16); used for DuX(2^16) 12/12 and DuX(2^8) 8/12 | Section V |
| T2 | Technique: a multiplicative coset on one word and a full-field word on another (O17); used for DuX(65537) 11/12 and 12/12 | Section V |
| T3 | Technique: multi-index weights (O18) instead of several structures | Sections V and VI |
| T4 | Ten rounds of DuX from one structure with a few weights | Section V |
| C1 | Weighted moments applied to the reduced configuration of Liu and Sun | Supplement, comparison with Liu and Sun |
| r_KR | Number of extension rounds of a key recovery (1 or 2) | Section IV |
| N_w | Number of weights (exponents a) per structure | Section IV |
| `1101`, `0001`, ... | Balance pattern of the four positions of a block (bit p set = position p balanced in all four blocks) | Section II |
| rank, pinned | Rank of the assembled system; number of linearisation monomials it determines | Table "rank against determined monomials" |
| inner_correct | Number of correctly recovered words of rk^0 = master key (16/16 = success) | -- |

## Batch labels

Directory names, run tags and docstrings also carry the batch labels under
which we organised the work: `E01`-`E15` and `Y01`-`Y10` are experiment
series on DuX and YuX, `W1`-`W29` and `S1`-`S22` are work packages (the `S`
packages ran on the 52-core server), and `R2`-`R9` are revision rounds of the
manuscript.  They identify records only; nothing in the paper depends on
them.  The most important ones for the paper:

| Label | Content |
|---|---|
| S18 | The measurement protocol and the reported runs of every executed attack (`results/S18_protocol/`, `docs/measurement_protocol.md`) |
| S19, S20, S21 | Full runs of techniques T1, T2 and T3 on the real instances (`results/R9_server/`) |
| S16 / R8 | Point-set runs (O15) and the O14 check at p = 257 (`results/R8_server/`) |
| R6D*, R7* | Third and further keys of the eleven-round YuX and other rows (`results/R6_third_keys/`, `results/R7_keys/`) |
| Y06, Y08 | Two-round key recovery on YuX (`results/Y06_kr2/`) |

Comments that cite "the memo", "the ledger" or a section number of an internal
report refer to our internal working notes, which are not part of this
repository.  `results/README.md` takes the place of the results ledger; its
sections A, A', B, B', C, C' and D are named as in those comments.
