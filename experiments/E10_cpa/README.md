# E10: the chosen-plaintext model (the designers' model)

The design paper of DuX claims resistance to higher-order differential attacks
from seven rounds on, in the chosen-plaintext model.  This directory repeats
the criterion, the measurements and the key recovery in the encryption
direction (Appendix A of the paper, supplement "Chosen-Plaintext Results").

Differences to the decryption direction: S has coordinate degrees (5,3,2,8);
the forward L0 is dense, so all 16 words share one bound from layer 2 on and
the degree grows by 8 per layer; the best active position is 1.

```bash
python tools/zero_sum_criterion.py --q 65537 --active 1 --direction enc --layers 8
python experiments/E10_cpa/run_zero_sum_cp.py --instance dux-65537 --active 1 --layers 8 \
    --keys 3 --out results/E10_cpa
# last-round key recovery, seven rounds
python experiments/E10_cpa/attack_cp_lastround.py --instance dux-65537 --rounds 7 --active 1 --keys 2 --seed 2026
python experiments/E10_cpa/attack_cp_lastround.py --instance dux-2^16 --rounds 7 --active 1 \
    --coords 0,1,2,3 --keys 2 --seed 2026
# two extension rounds, eight rounds, one structure of 2^16 plaintexts and 400 weights
python experiments/E10_cpa/kr2_cpa.py --instance dux-2^16 --layers 6 --active 1 --combine 1110 \
    --weights 400 --structures 1 --seed 2026 --out results/E10_cpa
# the base case of the theorem on the next-to-leading coefficient (O9)
python experiments/E10_cpa/check_thmS1_base.py --instance dux-2^16 --layers 6 --keys 200 \
    --seed 1 --positions 0,1,2,3 --out results/E10_cpa/thmS1_base_dux-2^16.json
```

`--coords` lists the balanced positions of the distinguisher, not S-box
coordinates.  `fast/` holds the C kernel `mom_cp.c` (the encryption-direction
mirror of `experiments/E06_key_recovery_1round/fast/mom.c`) for structures of 2^32 plaintexts
(`make -C experiments/E10_cpa/fast`).  `best_word_sets.py` finds the best word
set for a given number of active words.

Records: `results/E10_cpa/`; reported runs in `results/S18_protocol/`.
