# E02: degree bounds

* `python tools/maxplus.py --q 65537 --pos 3 --layers 11`: the max-plus bound
  on the formal degree of every word after every layer (the class recursion of
  Section III of the paper).
* `python tools/exponent_sets.py --n 16 --pos 3 --layers 11`: exact
  propagation of univariate exponent sets (Ni et al., Prop. 1), giving Boolean
  degree bounds per word.
* `symbolic_layers.py`: exact symbolic expansion of the first layers over
  F_p (with coefficients, so cancellations are visible), compared with the
  bounds; `--subgroup-scan` checks the tightness of the 9-layer
  distinguisher.

```bash
python experiments/E02_degree_bounds/symbolic_layers.py --instance dux-65537 \
    --layers 10 --keys 1 --positions 3 --subgroup-scan
```

Records: `results/E02_degree_bounds/`.
