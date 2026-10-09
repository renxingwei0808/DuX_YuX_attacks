# E03: zero sums on DuX(65537)

Direct summation of the decryption-direction state over a structure, layer by
layer, on several random keys.

```bash
python experiments/E03_zero_sum_Fp/run.py --instance dux-65537 --layers 12 \
    --keys 2 --positions all --out results/E03_zero_sum_Fp/<name>
```

`--subgroup k` uses cosets of the multiplicative subgroup of order 2^k (data
2^k).  The multi-word prime-field structures (two words at 2^32, signed
coset differences) were measured with the C kernel of E09
(`results/E03_zero_sum_Fp/multiword_full/`).  These measurements are the
prime-field rows of Table "zero-sum distinguishers" of the paper.
