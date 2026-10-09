# DuX: specification as implemented

This note records how the reference model in `dux/` implements DuX
(Wu et al., DCC 2026, 94:140) and why the attacks in this repository carry no
specification risk.

## 0. Summary

| Item | Reading |
|---|---|
| Specification source | the formal definition sections, the algorithms and the figures of the design paper; the expansions in its security-analysis section (Sect. 5) are not normative.  This rule and the readings below were confirmed with the design team. |
| Parameters | the same for all three instances: alpha = 179, r = 12 rounds, 128-bit security target; the block and the master key are 16 words of F_q (256 bits for DuX(2^16)) |
| Key schedule | the third branch of `Rf` is `S(x1) + x3 + 2 x0 + rc`, the formula of Sect. 4.3 and Fig. 3; the four-fold expansion printed in Sect. 5.5.1 drops every `2(.)` term and is right only in characteristic 2 |
| Implementation | `dux/` follows every clause below; no line of `dux/` had to change when the readings were confirmed |
| Security model | left to the analyst by the designers; our main line is the chosen-ciphertext model, with chosen-plaintext results for comparison |

The public implementation of the designers differs from the paper in several
points (Section 4) and is not used anywhere in this repository.

## 1. Parameters (Table 2 of the design paper)

| Instance | Field | Defining polynomial | alpha | Rounds | Security target | Master key |
|---|---|---|---|---|---|---|
| DuX(2^8) | F_{2^8} | x^8+x^4+x^3+x+1 (0x11B) | 179 | 12 | 128 bits | 16 words = 128 bits |
| DuX(2^16) | F_{2^16} | x^16+x^12+x^3+x+1 (0x1100B) | 179 | 12 | 128 bits | 16 words = 256 bits |
| DuX(65537) | F_65537 | -- | 179 | 12 | 128 bits | 16 words |

All complexities are compared with the 128-bit security target, not with the
key space.

## 2. Components

Notation: `+`, `-`, `*` are the field operations (both `+` and `-` are XOR in
characteristic 2); the state is `X = (x0, ..., x15)`, indices from 0.

**S-box (Sect. 3.1.1, Fig. 1).**

```
R    (x0,x1,x2,x3) = (x1, x2, x0*x1 + x3 + alpha, x0)
R^-1 (x0,x1,x2,x3) = (x3, x0, x1, x2 - x0*x3 - alpha)
S^-1 = R^4        decryption direction, coordinate degrees (2,3,4,2)
S    = (R^-1)^4   encryption direction, coordinate degrees (5,3,2,8)
```

Closed forms derived from the definitions (`dux/sbox.py`; their equality with
the iterated forms is tested):

```
S^-1:  f = x0*x1 + x3 + alpha,  g = x1*x2 + x0 + alpha
       (y0,y1,y2,y3) = ( g,  x2*f + x1 + alpha,  f*g + x2 + alpha,  f )

S   :  a = x2 - x0*x3 - alpha            (deg 2)
       b = x1 - x3*a  - alpha            (deg 3)
       c = x0 - a*b   - alpha            (deg 5)
       (y0,y1,y2,y3) = ( c, b, a, x3 - b*c - alpha )     (y3 of degree 8)
```

**SL layer (Sect. 4.2, Fig. 2).** Four S-boxes on the blocks 0-3, 4-7, 8-11,
12-15; `S` in encryption, `S^-1` in decryption.

**Linear layer (Sect. 3.2, Fig. 2).** Rotation and circulant convention, fixed
uniquely by the integer row printed in Sect. 3.2.2:

```
(X <<< j)_i = X_{(i+j) mod 16},   (Circ(m) X)_i = sum_j m_j X_{(i+j) mod 16}
L0^-1(X) = (X<<<1) + (X<<<4) + (X<<<8) + (X<<<9)  + (X<<<13)
L1^-1(X) = (X<<<1) + (X<<<5) + (X<<<8) + (X<<<12) + (X<<<13)
L0 over F_{2^n}: rotation set {0,1,3,4,7,8,9,10,12,14,15}
L1 over F_{2^n}: rotation set {0,3,4,5,6,8,10,11,12,13,15}
L0 over F_p:     M0 = Circ(-359, -209, -174, -129, -229, 116, 86, 521,
                           291, 311, 21, -194, 161, -14, -239, 261) / 1105,
                 for p = 65537: Circ(23249, 3677, 25325, 6346, 19394, 26808,
                 17615, 52608, 3974, 53794, 52311, 41042, 7829, 30663, 60021, 60318)
L1 over F_p:     M1 = the first row of M0 shifted by 4
t_xor = (rk^i_15 & 1) xor ((rk^i_15 >> 1) & 1)
LM(X) = L0(X) if t_xor = 0 else L1(X)
```

`t_xor` of round i is taken from the same round key `rk^i` that is added after
the linear layer.  Since `L1 = Rot_{-4} o L0`, the key-dependent choice is only
a block rotation (observation O2 in `docs/glossary.md`).

**Key schedule (Sect. 4.3, Fig. 3, Algorithms 1 and 2).**

```
Rf(X, rc) = ( S(x0) + x2 + rc,
              S(x1) + x0 + x3 + rc,
              S(x1) + x3 + 2*x0 + rc,
              x1 )                                 x_i, rc in F_q^4

round constants:  for i = 0..r-1, j = 0..3:  t = 16 i + 4 j,
                  (c0,c1,c2,c3) = S(t+1, t+2, t+3, t+4),  rc^i = rc^i || (c0,c1,c2,c3)
expansion:        rk^0 = K
                  rk^i = Rf applied four times to rk^{i-1}, the k-th time
                         with the k-th 4-word sub-block of rc^{i-1}
```

`Rf` is invertible, so any round key gives the master key.  In
characteristic 2, `2*x0 = 0` and `y1 + y2 = x0` (observation O5').

**Encryption and decryption (Sect. 4.4, Algorithm 3).**

```
Enc:  x = P + rk^0
      for i = 1..r-1:  x = SL(x);  x = LM(x, t_xor(rk^i));  x = x + rk^i
      x = SL(x);  C = x + rk^r                      (no linear layer in the last round)
Dec:  the step-by-step inverse
```

`decrypt_layers(C, rks, l)` returns the state after the first l `S^-1`
layers of decryption; every zero-sum experiment in this repository is a
statement about this state.

**Conventions the paper leaves implicit.** alpha = 179 is read as the
polynomial x^7+x^5+x^4+x+1 in F_{2^n} and as the integer 179 in F_p; the
integers t+k of Algorithm 1 are embedded the same way (all are below 2^8);
"the two lowest bits" of an F_p word are those of its representative in
[0, q).  None of these affects a zero sum or a key recovery result.

## 3. Traceability

| Clause | Source | Implementation | Tests |
|---|---|---|---|
| Parameters | Table 2 | `dux/params.py` | `test_vector_file_matches_model` |
| R, R^-1, S^-1 = R^4, S = (R^-1)^4, closed forms | Sect. 3.1.1, Fig. 1 | `dux/sbox.py` | `tests/test_dux.py` |
| SL | Sect. 4.2, Fig. 2 | `dux/cipher.py` | `test_encrypt_decrypt_roundtrip` |
| Rotation convention, L^-1, forward layers | Sect. 3.2.1-3.2.3 | `dux/linear.py`, `dux/params.py` | `test_paper_M0_row_p65537`, `test_paper_Fp_matrix_fractions_reduce_to_printed_row` |
| t_xor from word 15 | Algorithm 3 line 5 | `dux/params.py:TXOR_WORD`, `dux/linear.py:t_xor` | `tests/test_dux.py` |
| Rf, Rf^-1, round constants, expansion | Sect. 4.3, Fig. 3, Algorithms 1-2 | `dux/keyschedule.py` | `tests/test_dux.py`, `tests/test_vectors.py` |
| Encryption, decryption | Algorithm 3, Sect. 4.4 | `dux/cipher.py` | `tests/test_vectors.py` |
| Vectorised path equals scalar path | implementation | `dux/field.py`, `dux/sbox.py` | `test_vectors_match_vectorised_path` |

The regression vectors `tests/vectors/dux_{2^8,2^16,65537}.json` hold five
(K, P, C) triples per instance with all round keys, the encryption states and
the decryption layer states (`scripts/gen_vectors.py --seed 2026 --count 5`).
We generated them ourselves, because the public implementation of DuX does
not agree with the paper.

## 4. Differences of the designers' public implementation

The public implementation (github.com/Wyu-d11/DuXEncHomo, commit `796374b`)
agrees with the paper on the S-box directions, the rotation sets, the
circulant convention, `t_xor` and the round structure, and differs as follows.
In all cases we follow the paper.

| # | Item | Paper | Public code |
|---|---|---|---|
| C1 | constant alpha | 179 | 0xCD = 205 (all instances) |
| C2 | third branch of Rf | S(x1) + x3 + 2 x0 + rc | S(x1) + x3 |
| C3 | round constants | four different 4-word sub-blocks per round | the same `RC[i]` four times |
| C4 | key schedule of DuX(2^16) | Algorithm 2 | a placeholder bitwise OR |
| C5 | L^-1 over F_p | rotations and additions mod p | XOR, then reduction |
| C6 | t_xor in decryption | rk^i_15 | word 15 of an already transformed round key |
| C7 | round number | 12 | 12 in the tests, 9 in one parameter file |

## 5. Errata in the design paper

All formulas below were checked symbolically against the definitions by
`scripts/check_paper_formulas.py` (output: `results/W1_spec_audit/paper_formula_check.txt`).

| # | Location | Difference to the definition |
|---|---|---|
| T1 | Sect. 3.1.1, expansion of R^-2 | the 4th component lacks a -alpha term (confirmed by the designers) |
| T2 | Sect. 5.3, y1 of S | lacks a -alpha term |
| T3 | Sect. 5.3, y0 of S | uses b + alpha instead of b in the inner term |
| T4 | Sect. 5.3, y3 of S | lacks a -alpha term even with the correct y0, y1 |
| T5 | Sect. 5.5.1, the quadratic S-box system | y2 has the wrong sign and y1 does not match under either sign convention; the chain structure, and hence the equation count, is right |
| T6 | Algorithm 3 line 2 | sub- and superscript swapped in the last term |
| T7 | Sect. 5.3, list of integral distinguishers | one output list has 15 words |
| T8 | Sect. 5.5.1, four-fold key-schedule expansion | drops every 2(.) term, so it holds only in characteristic 2 |

**The six-round integral examples of Sect. 5.3.** The four examples
(active word -> balanced word) 3 -> 0, 7 -> 7, 11 -> 4, 15 -> 12 contradict
each other, because DuX is equivariant under the rotation by one block.  Our
direct measurement on two keys and all 16 active positions (encryption
direction) gives the pattern `1111` at layer 5 on both large instances; at
layer 6, DuX(2^16) has `1111` for active positions = 1 mod 4, `1110` for 0, 2
and `0110` for 3, and DuX(65537) has `1110` for 0, 1, 2 and `0110` for 3; at
layer 7 no word is balanced (`results/E10_cpa/cp_zerosum_*.json`).
