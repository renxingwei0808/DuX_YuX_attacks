# E05: general monomial prediction and multivariate exponent sets

Two complementary upper bounds in the decryption direction (with the fixed L0
of O2), used to compare our criterion with the monomial-prediction approach.

1. `gmp_dux.py`: general monomial prediction over the field (Cui, Hu, Wang
   and Wei, ASIACRYPT 2022), modelled with the rules of Ni et al., Appendix A,
   and solved with z3.  `--mode top` asks only whether the top monomial
   `prod_j X_j^{2^n - 1}` is reachable; UNSAT proves the zero sum.

   ```bash
   python experiments/E05_monomial_prediction/gmp_dux.py --n 8 --active 3 --layers 5 --words 0,1,2,3
   python experiments/E05_monomial_prediction/gmp_dux.py --n 4 --active 3,7,11,15 --layers 5 --mode top --timeout 3600
   python experiments/E05_monomial_prediction/run_tables.py --spec decide4 --mode top --procs 12 \
       --timeout 14400 --out results/E05_monomial_prediction/gmp_top_n4_decide.jsonl
   ```

2. `tools/exponent_sets_multi.py`: multivariate exponent sets,
   `deg_m(word, layer) <= sum_j min(d(layer, word; active_j), m)`, a refinement
   of Prop. 3 of Ni et al. that runs in seconds.

Order of the bounds: experiment <= monomial prediction <= exponent sets <= n s.
Records: `results/E05_monomial_prediction/`.
