# Y01: YuX reference implementation and specification audit

`yux/` has the same interface as `dux/`; `docs/yux_specification.md` maps
every definitional statement of the YuX paper (Liu et al., IEEE TIT 70(5),
2024) to code and tests.

```bash
python -m pytest -q tests/test_yux.py
python experiments/Y01_spec_audit/audit.py --out results/Y01_spec_audit
python scripts/gen_vectors.py --cipher yux      # regenerates tests/vectors/yux_*.json
```

`audit.py` checks three things mechanically:

1. every clause of the table in `docs/yux_specification.md` names a test that
   exists in `tests/test_yux.py`;
2. the two explicit objects the paper prints (the Yu2X encryption rotation set
   and the YupX row v_p) are recomputed by inverting the decryption circulant
   and compared with the paper;
3. erratum E1: the quadratic system printed in Sect. VI-D
   (`yux/sbox.py::paper_VI_D_system`) is compared point by point with
   S = Pf^-4 on six fields.

Record: `results/Y01_spec_audit/audit_table.json`.
