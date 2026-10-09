# E08: Boolean degree of the two-round extension

Ni et al. (DCC 2026, 94:123, Sect. 4.3) use Boolean degree d = 6 for two
extension rounds.  `run.py` measures it on the toy F_{2^4} instance, where the
full ANF in the key bits is computable by the Moebius transform (`moebius.py`,
`keypoly.py`): the degree is 7, the same for DuX and Yu2X, which changes the
estimate 2^120.4 of their 12-round Yu2X-16 attack (supplement, "Boolean
degree of the two-round extension").

```bash
python experiments/E08_boolean_degree_extension/run.py --out results/E08_boolean_degree_extension
```

Record: `results/E08_boolean_degree_extension/boolean_degrees.json`.
