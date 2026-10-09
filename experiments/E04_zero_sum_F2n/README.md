# E04: zero sums on DuX(2^n)

```bash
python experiments/E04_zero_sum_F2n/run.py --instance dux-2^8 --active 3,7,11 \
    --layers 10 --keys 2 --chunk 20 --out results/E04_zero_sum_F2n/<name>
```

* `--active` lists the active words; `--dim m` lets each active word run over
  an m-dimensional F_2-affine subspace (default: the whole field); data
  2^{m s}.
* `sweep.py` runs the systematic sweep of structures up to 2^26.
* `fast/` holds the C kernel (`make -C experiments/E04_zero_sum_F2n/fast`)
  and its driver `run_fast.py`; `fast/s1_matrix.sh` is the matrix of large
  runs up to 2^32 (`results/E04_zero_sum_F2n/large/`).

These measurements are the characteristic-2 rows of Table "zero-sum
distinguishers" of the paper.
