# E11: cheap coordinate elimination (O10) and the characteristic-2 collapse

`tools/cheap_rows.py` computes the left kernel K of the expensive columns of a
balanced pattern (Lemma "cheap coordinates"): every y in K gives a row that
contains only the cheap coordinate of the outer S-box.

| Direction | Outer map | Matrix | DuX cheap column | YuX cheap column |
|---|---|---|---|---|
| `dec` (chosen ciphertext) | encryption S | L | position 2 (`a = x2 - x0 x3 - alpha`) | position 3 (`z0 = x3 - x0 x1 - x2 - alpha`) |
| `enc` (chosen plaintext) | S^-1 | L^-1 | positions {0,3} (`g`, `f`) | positions {0,1} (`y0`, `y1 - y0`) |

```bash
python tools/cheap_rows.py --cipher dux --field 65537 --direction dec --balanced 0001
python tools/cheap_rows.py --cipher yux --field 2^16  --direction dec --balanced 1110
python experiments/E11_cheap_rows/run.py --out results/E11_cheap_rows
```

`char2_collapse.py` checks the characteristic-2 collapse theorem: with dim K =
1 the combined row only sees orbit sums of the key columns in characteristic 2,
while over F_p it reaches all four character components.  It measures the rank
of pseudo-structure systems on the toys:

```bash
python experiments/E11_cheap_rows/char2_collapse.py --instance toy-2^4 --rounds 6 \
    --combine 0001 --structures 6000 --points 256 --out results/E11_cheap_rows/collapse_toy-2^4.json
python experiments/E11_cheap_rows/char2_collapse.py --instance toy-257 --rounds 7 \
    --combine 0001 --structures 8700 --points 256 --out results/E11_cheap_rows/collapse_toy-257.json
```

Records: `results/E11_cheap_rows/` (the full kernel table is
`cheap_rows_table.md`); tests: `tests/test_cheap_rows.py`.
