# YuX: specification as implemented

This note records how the reference model in `yux/` implements the YuX family
(Liu et al., IEEE TIT 70(5), 2024).  The specification source is the
definitional part of that paper only: Section III (Construction 1 and Fig. 1),
Section IV-A (S/S^-1, linear layer, round-key addition), Section IV-B (the
instances Yu2X-n and YupX-p), Algorithm 1 (key schedule), Algorithm 2 (round
constants), Algorithm 3 (encryption), Table II (parameters) and footnote 2
(rotation convention).  Section 2.3 of Ni, Wang and Li (DCC 2026, 94:123) was
used as a cross-check only; the public code of the designers was not used.

`experiments/Y01_spec_audit/audit.py` parses the clause table below and checks
that every test it names exists in `tests/test_yux.py`.

## 1. Clause table

Each clause: position in the paper, implementation, and the test that locks it.
All tests are in `tests/test_yux.py` unless stated otherwise.

| # | Clause | Paper | Implementation | Test |
|---|---|---|---|---|
| S1 | The state is 16 words over F_q in four blocks of four words; each block goes through one S-box | Sect. IV-A, Algorithm 3 line 4 | `yux/params.py::WORDS/BLOCKS/BLOCK_WORDS`, `YuX.SL` | `test_encrypt_decrypt_roundtrip_scalar_and_vector` |
| S2 | Pf(x0,x1,x2,x3) = (x1, x2, x3, x0 + x1 x2 + x3 + alpha) | Sect. III-A, Construction 1 | `yux/sbox.py::Pf` | `test_Pf_inv_is_the_inverse_of_Pf` |
| S3 | Pf^-1(x0,x1,x2,x3) = (x3 - x0 x1 - x2 - alpha, x0, x1, x2) | Sect. III-A ("As for the compositional inverse") | `yux/sbox.py::Pf_inv` | same as above |
| S4 | S^-1 = Pf^4 (decryption direction) | Sect. IV-A.2 | `yux/sbox.py::S_inv` | `test_S_and_S_inv_are_mutually_inverse` |
| S5 | S = Pf^-4 (encryption direction) | Sect. IV-A.2 | `yux/sbox.py::S` | same as above |
| S6 | Closed form of S^-1: y0 = x0+x1x2+x3+alpha; y1 = x1+x2x3+y0+alpha; y2 = x2+x3y0+y1+alpha; y3 = x3+y0y1+y2+alpha; degrees (2,2,3,4) | derived from S4 (iterating Pf) | `yux/sbox.py::S_inv_closed` | `test_S_inv_closed_form_equals_Pf4` |
| S7 | Closed form of S: z0 = x3-x0x1-x2-alpha; z1 = x2-z0x0-x1-alpha; z2 = x1-z1z0-x0-alpha; z3 = x0-z2z1-z0-alpha; S = (z3,z2,z1,z0); degrees (8,5,3,2) | derived from S5 | `yux/sbox.py::S_closed` | `test_S_closed_form_equals_Pf_minus_4` |
| S8 | The vectorised S/S^-1 agree with the scalar ones point by point | implementation requirement | `yux/sbox.py::vS/vS_inv` | `test_vectorised_sbox_matches_scalar` |
| L1 | Rotation convention (x0,...,x15) <<< 1 = (x1,...,x15,x0), i.e. (X <<< j)_i = X_{(i+j) mod 16} | footnote 2 | `dux/linear.py::rotl` (shared) | `test_paper_rotation_sets_and_v_p_are_locked` |
| L2 | M = sum_{i in {0,3,4,8,9,12,14}} Rot_i, LP_inv = M (decryption direction, both fields) | Sect. IV-A.3 | `yux/params.py::ROT_INV`, `LinearLayer.L_inv` | same as above |
| L3 | LP = M^-1 (encryption direction) | Sect. IV-A.3 | `LinearLayer.L`, obtained by inverting `forward_row` | `test_linear_layers_are_mutually_inverse` |
| L4 | For Yu2X, LP = XOR_{i in {1,2,3,5,6,7,8,12,13,14,15}} Rot_i | Sect. IV-B.1 | computed, locked by the test | `test_paper_rotation_sets_and_v_p_are_locked` |
| L5 | For YupX, LP is circulant with first row v_p = (4/7, 5/21, 5/21, 5/21, 4/7, 5/21, 5/21, 5/21, -3/7, -16/21, -16/21, -16/21, -3/7, 5/21, 5/21, 5/21) | Sect. IV-B.2 | `yux/params.py::V_P_FRACTIONS`, `yux/linear.py::paper_v_p` | same as above |
| L6 | For p = 65537, v_p = (9363, 53054, 53054, 53054, 9363, 53054, 53054, 53054, 9362, 53053, 53053, 53053, 9362, 53054, 53054, 53054) | Sect. IV-B.2 | `yux/params.py::V_P_65537` | same as above (and M Circ(v_p) = I is checked) |
| L7 | LP(x)_i = sum_j v_p[j] x_{(i+j) mod 16} | Sect. IV-B.2 | the F_p branch of `LinearLayer.L` | `test_linear_layers_are_mutually_inverse` |
| L8 | YuX has NO key-dependent choice of linear layer | Sect. IV-A.3 (one LP and one LP_inv) | `LinearLayer.L(x, t, vec)` accepts and ignores t | `test_linear_layer_ignores_the_t_argument` |
| L9 | The vectorised linear layer agrees with the scalar one | implementation requirement | the `vec=True` path of `LinearLayer.L/L_inv` | `test_vectorised_linear_matches_scalar` |
| A1 | Encryption: x = P + rk^0 | Algorithm 3 line 1 | `YuX.encrypt` | `test_encrypt_decrypt_roundtrip_scalar_and_vector` |
| A2 | Encryption round i (i = 1..r-1): SL, then LP, then + rk^i | Algorithm 3 lines 4-8 | same as above | same as above |
| A3 | Last round: SL, then + rk^r (no LP) | Algorithm 3 lines 10-11 | same as above | same as above |
| A4 | Decryption is the step-by-step inverse (subtract round key, LP_inv, S^-1) | inverse of Algorithm 3 | `YuX.decrypt` | same as above |
| A5 | `decrypt_layers(C, rks, l)` is the state after the first l S^-1 layers of decryption | analysis object, same definition as in `dux` | `YuX.decrypt_layers` | `test_decrypt_layers_last_layer_matches_decrypt` |
| A6 | `encrypt_layers` is its chosen-plaintext mirror | same as above | `YuX.encrypt_layers` | same as above |
| K1 | rk^0 = Key | Algorithm 1 line 1 | `yux/keyschedule.py::key_expand` | `test_key_schedule_is_invertible` |
| K2 | X4 = X0 + S((X1+X2+X3) <<< 3) + (rc^{i-1}_{4j},...,rc^{i-1}_{4j+3}) | Algorithm 1 line 6 | `ks_round` | same as above |
| K3 | Each step j = 0..3 outputs one block and shifts X0 <- X1, X1 <- X2, X2 <- X3, X3 <- X4 | Algorithm 1 lines 7-8 | `ks_round` | same as above |
| K4 | rk^i = X4 X5 X6 X7 (the four blocks output in that round) | Algorithm 1 line 7 | return value of `ks_round` | same as above |
| K5 | "<<< 3" is the rotation of the four words of a block, (X <<< 3)_i = X_{(i+3) mod 4} | convention of footnote 2 applied to F_q^4 | `keyschedule.rot3` | `test_key_schedule_block_rotation_and_constants` |
| K6 | The key schedule is invertible (without inverting S) | follows from K2-K4 | `ks_round_inverse`, `YuX.key_schedule_inverse` | `test_key_schedule_is_invertible` |
| C1 | Round constants: i = 0..r-1, j = 0..3, id = 16 i + 4 j | Algorithm 2 lines 1-4 | `keyschedule.round_constants` | `test_key_schedule_block_rotation_and_constants` |
| C2 | (c0,c1,c2,c3) = S(id+1, id+2, id+3, id+4), with S the ENCRYPTION S-box | Algorithm 2 line 5 | same as above | same as above |
| C3 | rc^i is used to produce rk^{i+1} | index in Algorithm 1 line 6 | `key_expand` | `test_key_schedule_is_invertible` |
| T1 | Yu2X-8: F_{2^8}, x^8+x^4+x^3+x+1, alpha = 205, 12 rounds, 128 bits | Table II | `INSTANCES["yu2x-8"]` | `test_table_II_parameters` |
| T2 | Yu2X-16: F_{2^16}, x^16+x^12+x^3+x+1, alpha = 205, 12* / 14 rounds | Table II | `INSTANCES["yu2x-16"]` (`rounds` = 12, `rounds_128` = 14) | same as above |
| T3 | YupX-65537: F_65537, alpha = 205, 9* (118 bits) / 14 (128 bits) rounds | Table II | `INSTANCES["yupx-65537"]` (`rounds` = 14, `rounds_fhe` = 9) | same as above |
| T4 | The asterisk means "the recommended number of rounds in fully homomorphic application scenarios" | Table II footnote | field `rounds_note` | same as above |
| T5 | Yu2X uses "not less than 12" rounds (128 bits); YupX "not less than 9" (118 bits) | Sect. IV-B.1 / IV-B.2 | same as above | same as above |
| R1 | Three toy instances `yuxtoy-2^4/193/257` (NOT in the paper, only used to validate pipelines cheaply) | -- | `INSTANCES` | `test_encrypt_decrypt_roundtrip_scalar_and_vector` |
| R2 | Regression vectors (generated by us, freezing the semantics) | -- | `tests/vectors/yux_*.json`, from `scripts/gen_vectors.py --cipher yux` | `test_yux_regression_vectors` |
| R3 | The instance-name prefix selects the cipher: `yu2x-`/`yupx-`/`yuxtoy-` -> YuX, `dux-`/`toy-` -> DuX | -- | `dux/registry.py` | `test_registry_dispatches_by_prefix` |

40 clauses in total.

## 2. Ambiguities

For each: what the paper says, what we implement, and the effect on our results.

| # | Ambiguity | Paper | Implementation | Effect on the results |
|---|---|---|---|---|
| A1 | X_i in line 2 of Algorithm 1 | `X_i = (key_i, key_{i+1}, key_{i+2}, key_{i+3}), i in {0,...,3}`, literally four SLIDING windows using only key_0..key_6; Sect. VI-D writes `RK_{0,i} = [k_{4i}, k_{4i+1}, k_{4i+2}, k_{4i+3}]`, i.e. NON-OVERLAPPING blocks | non-overlapping blocks (`ks_literal=False`, the default); the literal reading is available as `ks_literal=True` | none: every attack recovers rk^0 = Key, which does not depend on the reading.  Under the literal reading rk^1..rk^r carry only 7 words of entropy (`test_ks_literal_reading_uses_only_seven_key_words`); running the two-round extension under both readings gives identical systems (`results/R6_third_keys/s10_R6D2_yuxtoy257_r6_{default,ksliteral}.json`) |
| A2 | Integer -> field element (round constants) | Algorithm 2 feeds the integers id+1..id+4 to S without fixing the encoding | `F.from_int` (identity on [0,q) for F_p; coefficient vector for F_{2^n}, consistent with the encoding of alpha = 205) | none (round constants only change affine constants of the key schedule) |
| A3 | Is alpha shared by the instances? | the Constant column of Table II reads 205 for all three | 205 for all three; the toys use 205 mod p (`yuxtoy-2^4` uses 5) | none (alpha is a known constant everywhere) |
| A4 | Meaning of "<<< 3" in the key schedule | line 6 of Algorithm 1 only writes <<< 3; footnote 2 is stated for the 16-word state | rotation of the four words of a block with the same convention, supported by `LK(x0,x1,x2) = [x0+x1+x2] <<< 3, x_i in F_q^4` in Sect. VI-D | none (as A2) |
| A5 | Two round numbers for Yu2X-16 and YupX | Table II has two rows per instance, the starred one is the FHE recommendation | both are recorded: `rounds` = 12 for Yu2X-16 (FHE, also the target of Ni et al.) and 14 for YupX (128 bits); the other one in `rounds_128` / `rounds_fhe` | results always state which one is meant; "YupX 9 rounds" is the recommended FHE instance `rounds_fhe = 9` |
| A6 | Range of n for Yu2X-n | Sect. IV-B.1 says n >= 8 | only the Table II instances n = 8, 16; the toy n = 4 is marked as not a paper instance | none |
| A7 | Is LP MDS? | Sect. V-D claims LP is an MDS map over (F_q^4)^4 | not a clause (design rationale, not definition); the implementation follows L2/L3 | none (no attack uses the MDS property) |

## 3. Errata

| # | Location | Printed | Difference to the definition | Handling |
|---|---|---|---|---|
| E1 | Sect. VI-D, the four quadratic S-box equations | y3 = x0 x1 + x2 + alpha - x3; y2 = y3 x0 + x2 + alpha - x1; y1 = y2 y3 + x0 + alpha - x1; y0 = y1 y2 + y3 + alpha - x0 | In characteristic 2 this is exactly S = Pf^-4 of Construction 1 (with y_i = z_{3-i}); over F_p it is not: the first line is -z0 instead of z0, and the signs and alpha terms of the later lines are off accordingly | implemented per Construction 1; `yux/sbox.py::paper_VI_D_system` copies the printed system and `test_paper_VI_D_system_is_a_sign_typo_over_Fp` checks "identical in characteristic 2, different at 100/100 random points over F_p, first line off by exactly a sign" |
| E2 | Table 6, last row (Ni et al., not the design paper) | input word set `(0,1,2,3,4,5,6,7,8,9,10,11,12,14,13,15)` | 14 and 13 swapped; the set is all 16 words | `experiments/Y02_degree_bounds/ni_tables.py` treats it as a set |

## 4. Differences to DuX

| Item | YuX | DuX | Consequence |
|---|---|---|---|
| S-box | Pf^4 / Pf^-4, alpha = 205 | R^4 / (R^-1)^4, alpha = 179 | decryption degrees (2,2,3,4) vs (2,3,4,2); encryption (8,5,3,2) vs (5,3,2,8) |
| Cheap coordinate (encryption S) | position 3, z0 = x3 - x0x1 - x2 - alpha | position 2, a = x2 - x0x3 - alpha | different combined rows: `assemble_fast.OUTER_EQUATIONS` |
| Cheap coordinates (S^-1, CPA) | y0 and y1 - y0 at position 0, cheap columns {0,1} | f (position 3) and g (position 0), cheap columns {0,3} | the two tables of E11 |
| Decryption linear layer | sum_{j in {0,3,4,8,9,12,14}} Rot_j, 7 terms, all residues mod 4 | sum_{j in {1,4,8,9,13}}, 5 terms, residues {0,1} only | YuX degree grows by 4 per layer, DuX by 2+sqrt(3); every W' word of YuX depends on all four positions of Z |
| Encryption linear layer | F_2: 11 terms {1,2,3,5,6,7,8,12,13,14,15}; F_p: circulant v_p | F_2: 11 terms {0,1,3,4,7,8,9,10,12,14,15}; F_p: M0 | -- |
| Key-dependent layer | none | L0/L1 selected by two low bits of rk^i_15 (only a block rotation) | `equivalent_fixedL0_keys` is trivial for YuX |
| Key schedule | four-branch NLFSR, X4 = X0 + S((X1+X2+X3) <<< 3) + rc | Rf^4 with the 2 x0 term | both invertible: any round key gives the master key |

## 5. Validity of the implementation

* `pytest -q tests/test_yux.py` covers every clause of Section 1.
* Encryption and decryption are mutually inverse on the three paper instances
  and the toys, and the scalar and vectorised paths agree word for word.
* The two explicit objects the paper prints, the Yu2X encryption rotation set
  and the YupX row v_p (as fractions and as integers for p = 65537), are NOT
  copied into the implementation: they are recomputed by inverting the LP_inv
  circulant and then compared with the paper (`audit.py`).
* The key schedule is invertible under both readings of A1.
* The regression vectors `tests/vectors/yux_{yu2x-8,yu2x-16,yupx-65537}.json`
  (three per instance, with rk^1, all round keys, the encryption states and the
  decryption layer states) freeze the semantics.
