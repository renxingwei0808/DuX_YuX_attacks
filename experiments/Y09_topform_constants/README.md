# Y09: key-independent constants at the boundary cells (O14)

For a prime-field full-block structure (O11), the sum of a word whose formal
degree equals the threshold, D = T = 4(p - 1), is a constant that does not
depend on the key (supplement, "Boundary Cells over Prime Fields").  In the
decryption direction of YuX the constants are nonzero; in the encryption
direction they are zero for YuX for every p, which turns the layer-7
full-block pattern of YupX-65537 from `0111` into `1111`.

| Direction | Top position | Growth per layer | Boundary | Primes |
|---|---|---|---|---|
| YuX decryption (S^-1 = Pf^4) | 3 | 4 | 4^{l-1} = 4(p-1) | 5 / l3, 17 / l4, 257 / l6, 65537 / l10 |
| YuX encryption (S = Pf^-4) | 0 | 8 | 8^{l-1} = 4(p-1) | 3 / l2, 17 / l3, 65537 / l7 |
| DuX encryption | 3 | 8 | same | same |

```bash
python experiments/Y09_topform_constants/run.py --part all --out results/Y09_topform_constants
#   --part dec   p = 5 layer 3 and p = 17 layer 4: 3 real master keys x 2 sets of
#                inactive constants against the top-form recursion
#   --part enc   YuX at p = 17 layer 3, DuX at p = 3 layer 2
#   --part rank  the mechanism: rank of the rows that the top forms use

# p = 257, layer 6: recursion (seconds) and the direct sum over 2^32 points (C)
python tools/topform_constants.py --p 257 --layers 6 --export-linv /tmp/linv_257.txt
gcc -O3 -march=native -fopenmp -o tools/topform_kernels tools/topform_kernels.c
./tools/topform_kernels projective 257 6 tools/linv_257.txt
./tools/topform_kernels direct     257 6 <seed> tools/linv_257.txt
```

The toy instances `yuxtoy-5`, `yuxtoy-17` and `toy-3` exist only for this
check (the circulants are invertible there).  Records:
`results/Y09_topform_constants/y09_all.json`, and for p = 257
`results/R8_server/R8_O14_yuxtoy257_fb0_l6_3keys.json` and
`results/R8_server/direct_p257_l6_server.txt`.
