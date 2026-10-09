# Y10: the Boolean hybrid route and the degree d_3

1. `rank_toy.py`: on the toy yuxtoy-2^4, do weighted rows add rank inside a
   Boolean linearised system (our full-block zero sum fed into the Boolean
   linearisation of Ni et al.)?  They do (97.6 % of the rows, two keys).
2. `d3.py`: the Boolean degree d_3 of three extension rounds of Yu2X-16 in the
   key bits.  The hybrid route to 13 rounds needs d_3 <= 10; exact Moebius
   transforms on 12, 16 and 20 symbolic key bits of the real Yu2X-16 all
   saturate, so d_3 >= 20 and the route is closed.  `d3.py` also recomputes
   the three numbers of Ni et al. (C(128, <= 12) = 2^54.55, data 2^95.55,
   time 2^115.38) from their formulas.

```bash
python experiments/Y10_boolean_hybrid/d3.py --moebius-bits 12 16 20 --structures 4 \
    --out results/Y10_boolean_hybrid
python experiments/Y10_boolean_hybrid/rank_toy.py --structures 6 --weights 12 --keys 2 \
    --checkpoint 2 --out results/Y10_boolean_hybrid
```

Records: `results/Y10_boolean_hybrid/`.
