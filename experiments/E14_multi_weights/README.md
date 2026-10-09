# E14: multi-index weights (T3), ten rounds from one structure (T4), and the configuration of Liu and Sun (C1)

## T3: multi-index weights

The weighted-moments theorem holds for every multi-index `a in N^s` with
`|a| < T - D`.  A multi-structure attack can therefore use one structure with
weights on several axes instead.

* Counting: `weighted.plan(dims=s)`; for full blocks the cost of a power of
  word i is `deg_i`, the degree of the corresponding S-box coordinate after
  the substitution of O11 (DuX positions 2/1/0/3 -> 2/3/5/8, YuX positions
  3/2/1/0 -> 2/3/5/8).  `tests/test_multi_index_weights.py` pins the counts.
* Assembly: `attack_2round_fast.py --weight-dims k` (in memory) and
  `attack_12round.py --weight-dims 2` (streamed, both characteristics);
  `--vandermonde-axes m` joins m axes into one pass over the structure.
* `rank_curves.py`: on the toys, one structure with multi-index weights
  reaches the same ceiling as several structures with one-dimensional weights.
* `data_plan.py`: the data of every multi-structure row once the weights are
  multi-index.

## T4: ten rounds from one structure

`t4_10round.py` runs `attack_1round.py --structures 1 --auto-weight` on 50
keys for both large DuX instances: data 2^16 instead of two structures
(2^17).  Records: `results/E14_multi_weights/t4_10round.{md,json}`.

## C1: the reduced configuration of Liu and Sun

Their reduced experiment on DuX(2^8) (7 rounds, s = 3, 2^24 per subspace,
three stages that need 10, 4 and 4 subspaces).  `c1_ls_reduced.py`
implements their three-stage system and measures the rank of each stage
against the number of weights on one subspace, on three random master keys:
weighted moments alone bring every stage down to one subspace.  Record:
`results/E14_multi_weights/C1_ls_reduced.{md,json}`.
