# E07: key recovery with two extension rounds (r_KR = 2)

This directory holds the main attack code of the paper: the structured
linearisation of two extension rounds (Section IV), its assembly over F_p and
F_{2^n}, the weighted moments (O12), the combined rows of cheap coordinate
elimination (O10, O16), and the streamed driver that runs the full-round
attacks.

## Setting

With r = l + 2 rounds and a zero sum at layer l, the unknowns are rk^0
(16 words) and rk^1 (16 words; by O2 the fixed L0 is used and no round-key bit
is read).  The state W = SL(LM(SL(P + rk^0)) + rk^1) gives, for every outer
block j, the equation of the cheap coordinate of the outer S-box

```
sum_P u0 u3 + k'3 sum_P u0 + k'0 sum_P u3 + sum_P u2 = 0,   u = block j of L0(SL(P + rk^0))
```

whose unknowns are products of rk^0 monomials of two inner blocks and rk^1
words times rk^0 monomials.  The coefficients are moments of the structure,
so a structure contributes `n_eq` linear equations per weight.

## Files

| File | Role |
|---|---|
| `attack_12round.py` | the streamed driver used for every reported r_KR = 2 run: walks a structure slice by slice (2^32 and more points), weighted moments by one dgemm per chunk, combined rows (`--combine`), cheap set {1,2} (`--cheap-set 1,2`), point sets (`--point-set`), mixed coset structures (`--mixed`), multi-index weights (`--weight-dims`, `--vandermonde-axes`), full blocks (`--full-block`), checkpoints, `--procs 1` serial path |
| `attack_2round_fast.py` | in-memory driver for smaller structures and the toys |
| `attack_2round_toy.py` | symbolic expansion (`joint_expansion`, `monomial_set`), structure generation, the prototype attack |
| `assemble_fast.py` | assembly over F_p as matrix products; `Moments` / `rows_from_moments` stream the points; `weighted_moment_stream` is the slice decomposition of O12 |
| `assemble_fast_2n.py` | assembly over F_{2^n} (log/antilog tables and XOR, no BLAS) |
| `weighted.py` | `plan`: the admissible weights for a structure and the rows it uses (`usable_weights`) |
| `modp_solve.py` | blocked Gauss-Jordan over F_p (BLAS) and a numpy version over F_{2^n} |
| `fast/gf2nsolve.c`, `fast/gf2n_solve.py` | blocked Gauss-Jordan over F_{2^n} in C with OpenMP, and its Python wrapper |
| `fast/mom2n.c` | moment kernel for characteristic 2 |
| `nmin_scan.py` | assemble once, solve on prefixes: the minimal number of structures and the rank saturation point |
| `rank_phi.py` | the template bound rank(Phi on V_0) on pseudo-structures, without any encryption (Proposition "template bound") |
| `throughput.py` | per-point assembly cost, split into decryption, powers, moments and rows |

Build the C kernels first:

```bash
make -C experiments/E07_key_recovery_2round/fast
```

## Examples

```bash
# smoke test of the 12-round algebra on the toy field (seconds)
python experiments/E07_key_recovery_2round/attack_12round.py --instance toy-257 \
    --layers 6 --active 3,7 --weights 150 --combine 0001 --normalise --structures 1 \
    --procs 1 --seed 11 --out /tmp/e07 --tag smoke_toy257

# template bound on pseudo-structures
python experiments/E07_key_recovery_2round/rank_phi.py --instance yuxtoy-257 --rounds 6 \
    --unknown-blocks 0,1 --structures 900 --checkpoints 300,500,700,800,900 \
    --moments realizable --points 256 --seed 2026
```

The eight-round toy analog over F_257 of the 12-round attack (paper,
Section VII, "Checks and Controls") reaches the same rank 8188 and the same
2592 determined monomials as the real attack:
`results/E07_key_recovery_2round/w16/fast_T2norm_toy257_r8_1d.json`.
The commands of every reported run are in `REPRODUCE.md` at the top of the
repository.

## Two pitfalls

1. `--weights N` is the NUMBER of weights (a = 0..N-1), not an upper bound.
   A weight beyond the margin T - D makes the equations false at the true key
   and the solver then silently returns a wrong answer
   (`tests/test_weighted_rows.py` tests exactly this), so let `weighted.plan`
   compute the margin.
2. `--limit-points` is for timing only; truncated structures violate the
   identities that `monomial_set` relies on.
