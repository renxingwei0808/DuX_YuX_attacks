"""E10 / W11 -- chosen-PLAINTEXT last-round key recovery (designer CPA model).

Setting.  r rounds, a CPA zero-sum that balances the state Y after S layer
r - 1 on (at least) some block positions.  From `encrypt`,

    C = S( L_t(Y) + rk^{r-1} ) + rk^r,      t = t_xor(rk^{r-1}),

so  S^{-1}(C - rk^r) = L_t(Y) + rk^{r-1}  and, summing over a structure whose
size is 0 in F_q, the unknown rk^{r-1} DROPS OUT:

    sum_P S^{-1}(C - rk^r) = L_t( sum_P Y )
    =>  [ L_t^{-1}( sum_P S^{-1}(C - rk^r) ) ]_{4b+p} = 0   for every balanced p.

t is unknown, but by O2 (L_1^{-1} = Rot_4 o L_0^{-1}) choosing L_0 only permutes
which block each equation belongs to, and we write the equations of all four
blocks, so the SET of equations is the same -- the script always uses L_0.

Linearisation.  L^{-1} = sum of the rotations {1,4,8,9,13}, whose offsets mod 4
are {1,0,0,1,1}, so the equation at output word 4b+p is a SUM (not a product)
of five S^{-1} coordinates, at block positions p and p+1, taken from four
different blocks.  Expanding S^{-1}(C + kappa) once in both kappa = -rk^r and C
(experiments/E08.../keypoly.py) and replacing each ciphertext monomial by its
structure moment turns every (structure, block, position) into ONE F_q-linear
equation in the monomials of kappa.  Only 14 (F_{2^n}) / 15 (F_p) kappa
monomials per block survive the sum, so the whole system has ~53-57 unknowns
and 12-16 equations per structure: a handful of structures suffice.

Finally rk^r is inverted through the key schedule (Rf_inv, which is invertible
in both characteristics) to the master key, and the result is checked against a
fresh plaintext/ciphertext pair.

Only the S^{-1} coordinates that L^{-1} actually reads at the balanced
positions enter the system (`used_sinv_coords`): with a single balanced
position p the five rotations hit coordinates p and p+1 only, so a partial
pattern such as the layer-7 `0010` of DuX(2^16) gives a SMALLER unknown set
(13 kappa monomials per block over F_{2^n} instead of 14) than the full
`1111` pattern.  For the patterns used by W11 (`1111`, `1110`) the set is
unchanged, so those numbers are not affected.

    python attack_cp_lastround.py --instance dux-65537 --rounds 7 --active 1 --coords 0,1,2
    python attack_cp_lastround.py --instance dux-2^16  --rounds 7 --active 1 --coords 0,1,2,3
    # S8: 8-round DuX(2^16) from the layer-7 `0010` zero-sum (2 words, 2^32
    # chosen plaintexts each) -- run it through fast/attack_cp_lastround_fast.py
    python attack_cp_lastround.py --instance dux-2^16 --rounds 8 --active 1,5 --coords 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "E06_key_recovery_1round"))
sys.path.insert(0, os.path.join(HERE, "..", "E08_boolean_degree_extension"))

import itertools                                        # noqa: E402

from dux import DuX                                     # noqa: E402
from dux.params import ROT_INV                          # noqa: E402
from dux.keyschedule import Rf_inv, round_constants     # noqa: E402
from keypoly import KeyPoly                             # noqa: E402
from attack_1round_partial import (determined_values,           # noqa: E402
                                   brute_force_free_words)


def sinv_joint(F, alpha):
    """S^{-1}(C + kappa): per coordinate, [(kappa-exponents, C-exponents, coeff)].
    Variables 0..3 are kappa, 4..7 are the ciphertext words of the block."""
    x = [KeyPoly.var_plus_const(F, i, 0, 8) + KeyPoly.var_plus_const(F, 4 + i, 0, 8)
         for i in range(4)]
    f = (x[0] * x[1] + x[3]).add_const(alpha)
    g = (x[1] * x[2] + x[0]).add_const(alpha)
    y1 = (x[2] * f + x[1]).add_const(alpha)
    y2 = (f * g + x[2]).add_const(alpha)
    out = []
    for t in (g, y1, y2, f):                 # coordinates 0, 1, 2, 3
        out.append([(e[:4], e[4:], co) for e, co in t.d.items() if any(e[4:])])
    return out


def used_sinv_coords(pos):
    """The S^{-1} coordinates that the equations at balanced block positions
    `pos` actually read.  L^{-1} = sum of the rotations {1,4,8,9,13}, whose
    offsets mod 4 are {1,0,0,1,1}, so position p reads coordinates p and
    p+1 (mod 4) -- and nothing else."""
    return sorted({(p + l) % 4 for p in pos for l in ROT_INV[0]})


def system_spec(F, alpha, pos):
    """Everything about the linearised last-round system that depends only on
    the field and the balanced positions: the joint expansion of S^{-1}(C+kappa),
    the surviving kappa monomials, the ciphertext moments they need, and the
    column layout (one shared constant column plus one column per (block,
    non-constant kappa monomial))."""
    joint = sinv_joint(F, alpha)
    used = used_sinv_coords(pos)
    kmons = sorted({ke for co in used for ke, _, _ in joint[co]})
    needed = sorted({pe for co in used for _, pe, _ in joint[co]})
    const = (0, 0, 0, 0)
    cols = [(None, const)] + [(b, m) for b in range(4) for m in kmons if m != const]
    return {"joint": joint, "used": used, "kmons": kmons, "needed": needed,
            "const": const, "cols": cols,
            "cidx": {m: i for i, m in enumerate(cols)},
            # When Y is balanced at ALL four block positions, L^{-1} is
            # invertible, so [L^{-1}u]_i = 0 for every i is equivalent to
            # u_j = 0 for every j -- and u_j involves only block j//4.  Using
            # the decoupled form then gives four independent per-block systems,
            # which is what makes the leftover key word enumerable; with a
            # partial pattern the equations genuinely mix blocks and the global
            # system is used.
            "decoupled": len(pos) == 4,
            "n_eq": len(pos) * 4}


def structure_rows(spec, F, pos, mom, rows, rhs, blkrows):
    """Append the equations of ONE structure (given its per-block ciphertext
    moments `mom`) to `rows`/`rhs` and, in the decoupled case, to `blkrows`."""
    joint, kmons, cidx, const = (spec["joint"], spec["kmons"], spec["cidx"],
                                 spec["const"])
    if spec["decoupled"]:
        for b in range(4):
            for co in spec["used"]:
                row = [0] * len(kmons)
                for ke, pe, coeff in joint[co]:
                    v = F.mul(coeff, mom[b][pe])
                    if v:
                        row[kmons.index(ke)] = F.add(row[kmons.index(ke)], v)
                blkrows[b].append(row)
    for b in range(4):
        for p in pos:
            row = [0] * len(spec["cols"])
            for l in ROT_INV[0]:
                j = (4 * b + p + l) % 16
                bj, co = j // 4, j % 4
                for ke, pe, coeff in joint[co]:
                    v = F.mul(coeff, mom[bj][pe])
                    if not v:
                        continue
                    i = cidx[(None, const)] if ke == const else cidx[(bj, ke)]
                    row[i] = F.add(row[i], v)
            rows.append(row)
            rhs.append(0)


def solve_system(spec, F, rows, blkrows):
    """Solve for kappa = -rk^r.  Returns (kappa, rank, unknowns, free_words,
    blk_cand); undetermined words are None in `kappa` and, in the decoupled
    case, carry a candidate list in `blk_cand`."""
    kmons, cols, cidx, const = (spec["kmons"], spec["cols"], spec["cidx"],
                                spec["const"])
    kappa = [None] * 16
    free_words, blk_cand = [], {}
    if spec["decoupled"]:
        rank = 0
        cl = [m for m in kmons if m != const]
        for b in range(4):
            R = [[v for t, v in enumerate(r) if kmons[t] != const]
                 for r in blkrows[b]]
            B = [F.sub(0, r[kmons.index(const)]) for r in blkrows[b]]
            det, rk = determined_values(F, R, B, cl)
            rank += rk
            kb = [det.get(tuple(1 if t == i else 0 for t in range(4)))
                  for i in range(4)]
            fr = [i for i in range(4) if kb[i] is None]
            if fr:
                cand = brute_force_free_words(
                    F, blkrows[b], kmons,
                    {i: kb[i] for i in range(4) if kb[i] is not None}, fr,
                    n_rows=len(blkrows[b]), max_out=64)
                free_words.append({"block": b, "free": fr,
                                   "candidates": None if cand is None else len(cand)})
                if cand and len(cand) == 1:
                    for slot, i in enumerate(fr):
                        kb[i] = cand[0][slot]
                elif cand:
                    # keep the whole candidate list; the surviving ambiguity is
                    # resolved below against a known pair
                    blk_cand[b] = (fr, cand)
            kappa[4 * b:4 * b + 4] = kb
        cl = [(b, m) for b in range(4) for m in cl]
    else:
        ci = cidx[(None, const)]
        R = [[v for t, v in enumerate(r) if t != ci] for r in rows]
        B = [F.sub(0, r[ci]) for r in rows]
        cl = [m for t, m in enumerate(cols) if t != ci]
        det, rank = determined_values(F, R, B, cl)
        for b in range(4):
            for i in range(4):
                e = tuple(1 if t == i else 0 for t in range(4))
                if (b, e) in det:
                    kappa[4 * b + i] = det[(b, e)]
        stage2 = None
        if any(v is None for v in kappa):
            det2, rank2 = solve_free_words_stage2(spec, F, rows, kappa)
            if det2:
                for (b, i), v in det2.items():
                    kappa[4 * b + i] = v
                stage2 = {"unknowns": len(det2) + sum(v is None for v in kappa),
                          "rank": rank2, "pinned": len(det2)}
        for b in range(4):
            fr = [i for i in range(4) if kappa[4 * b + i] is None]
            if fr:
                free_words.append({"block": b, "free": fr, "candidates": None})
        if stage2 is not None:
            free_words.append({"stage2": stage2})
    return kappa, rank, len(cl), free_words, blk_cand


def solve_free_words_stage2(spec, F, rows, kappa):
    """Second stage for a PARTIAL balanced pattern (Lemma O8 in the encryption
    direction).

    With a single balanced position the equations read only S^{-1} coordinates
    2 and 3.  kappa_3 enters coordinate 3 (f = x0 x1 + x3 + alpha) linearly and
    with no ciphertext factor, so its term contributes sum_P kappa_3 = 0 (O5);
    it survives only inside coordinate 2 (y2 = f g + x2 + alpha), where it is
    still LINEAR and always multiplied by a monomial in kappa_0..kappa_2.  The
    linearisation therefore spends three columns (kappa_3, kappa_2 kappa_3,
    kappa_1 kappa_3) on one unknown and cannot separate them -- exactly the
    mechanism of Lemma O8 on the decryption side.

    Once kappa_0..kappa_2 are known, each of those columns collapses to
    kappa_3^{(b)} times a known scalar and the SAME equations become a linear
    system in the four remaining words.  Returns {block: value} and its rank.
    """
    cols = spec["cols"]
    free = sorted({i for b in range(4) for i in range(4)
                   if kappa[4 * b + i] is None})
    if not free:
        return {}, 0
    for m in spec["kmons"]:                 # linearity in the free words
        if sum(m[i] for i in free) > 1:
            return None, 0
    lab = [(b, i) for b in range(4) for i in free]
    li = {t: k for k, t in enumerate(lab)}
    A, B = [], []
    for row in rows:
        coef = [0] * len(lab)
        const = 0
        for t, (b, m) in enumerate(cols):
            v = row[t]
            if not v:
                continue
            if b is None:                   # the shared constant column
                const = F.add(const, v)
                continue
            kb = kappa[4 * b:4 * b + 4]
            fi = [i for i in free if m[i]]
            if any(kb[i] is None for i in range(4) if m[i] and i not in free):
                return None, 0
            val = v
            for i in range(4):
                if i in free:
                    continue
                for _ in range(m[i]):
                    val = F.mul(val, kb[i])
            if fi:
                k = li[(b, fi[0])]
                coef[k] = F.add(coef[k], val)
            else:
                const = F.add(const, val)
        A.append(coef)
        B.append(F.sub(0, const))
    det, rank = determined_values(F, A, B, lab)
    return det, rank


def master_from_last_round_key(c, rounds, rcs, rk_last):
    """Invert the key schedule: rk^r -> master key (Rf_inv is invertible in
    both characteristics)."""
    stx = list(rk_last)
    for i in range(rounds, 0, -1):
        rc = rcs[i - 1]
        for j in range(3, -1, -1):
            stx = list(Rf_inv(c.F, c.alpha, stx, rc[4 * j:4 * j + 4]))
    return tuple(stx)


def structure_moments(c, rks, rounds, active, seed, needed, chunk_log2=22):
    """Encrypt one chosen-plaintext structure and return, per block, the
    ciphertext moments {exponent vector -> sum over the structure}."""
    F = c.F
    rng = np.random.default_rng(seed)
    consts = [int(v) for v in rng.integers(0, F.q, size=16)]
    total = F.q ** len(active)
    csize = min(total, 1 << chunk_log2)
    maxe = max(max(pe) for pe in needed)
    mom = [{pe: 0 for pe in needed} for _ in range(4)]
    for start in range(0, total, csize):
        idx = np.arange(start, start + min(csize, total - start), dtype=np.int64)
        P = [np.full(len(idx), consts[i], dtype=np.int64) for i in range(16)]
        for t, w in enumerate(active):
            P[w] = (idx // (F.q ** t)) % F.q
        C = c.encrypt(tuple(P), rks, rounds=rounds, vec=True)
        for b in range(4):
            pw = []
            for i in range(4):
                col = [np.ones(len(idx), dtype=np.int64)]
                for _ in range(maxe):
                    col.append(F.vmul(col[-1], C[4 * b + i]))
                pw.append(col)
            for pe in needed:
                acc = None
                for i in range(4):
                    if pe[i]:
                        acc = pw[i][pe[i]] if acc is None else F.vmul(acc, pw[i][pe[i]])
                if acc is not None:
                    mom[b][pe] = F.add(mom[b][pe], F.vsum(acc))
    return mom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="dux-65537")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--active", default="1", help="active PLAINTEXT words")
    ap.add_argument("--coords", default="0,1,2",
                    help="balanced block positions of Y (the CPA distinguisher)")
    ap.add_argument("--structures", type=int, default=None)
    ap.add_argument("--keys", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--chunk", type=int, default=22)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    active = [int(v) for v in a.active.split(",")]
    pos = [int(v) for v in a.coords.split(",")]
    c = DuX(a.instance, rounds=a.rounds)
    F = c.F
    spec = system_spec(F, c.alpha, pos)
    cols, kmons, needed, n_eq = (spec["cols"], spec["kmons"], spec["needed"],
                                 spec["n_eq"])
    n_struct = a.structures or (len(cols) // n_eq + 3)
    npts = F.q ** len(active)
    print(f"{a.instance} CPA: {a.rounds} rounds, active plaintext words {active} "
          f"({npts} = 2^{np.log2(npts):.1f} per structure), balanced positions "
          f"{pos}")
    print(f"  {len(cols)} unknowns ({len(kmons)} kappa monomials per block), "
          f"{len(needed)} ciphertext moments, {n_eq} equations/structure, "
          f"{n_struct} structures = 2^{np.log2(n_struct * npts):.1f} chosen plaintexts")

    rcs = round_constants(F, c.alpha, a.rounds)
    results = []
    for ki in range(a.keys):
        t0 = time.time()
        rng = np.random.default_rng(a.seed + ki)
        K = c.random_key(rng)
        rks = c.key_schedule(K)
        rows, rhs = [], []
        blkrows = {b: [] for b in range(4)}
        for st in range(n_struct):
            mom = structure_moments(c, rks, a.rounds, active,
                                    a.seed + 5000 + 100 * ki + st, needed, a.chunk)
            structure_rows(spec, F, pos, mom, rows, rhs, blkrows)
        kappa, rank, n_unknown, free_words, blk_cand = solve_system(
            spec, F, rows, blkrows)
        # One extra chosen-plaintext query resolves any residual ambiguity:
        # every combination of the per-block candidate lists gives a candidate
        # rk^r, hence (the key schedule being invertible) a candidate master
        # key, which is tested on one known plaintext/ciphertext pair.
        Pchk = tuple(int(v) for v in np.random.default_rng(a.seed + 77 + ki)
                     .integers(0, F.q, size=16))
        Cchk = tuple(c.encrypt(Pchk, rks, rounds=a.rounds))
        trials = 1
        if blk_cand:
            lists = [blk_cand[b][1] if b in blk_cand else [None] for b in range(4)]
            total = 1
            for L in lists:
                total *= len(L)
            trials = total
            found = False
            for combo in itertools.islice(itertools.product(*lists), 1 << 20):
                kk = list(kappa)
                for b in range(4):
                    if b in blk_cand:
                        fr = blk_cand[b][0]
                        for slot, i in enumerate(fr):
                            kk[4 * b + i] = combo[b][slot]
                if any(v is None for v in kk):
                    continue
                cand_rk = [F.sub(0, v) for v in kk]
                if tuple(c.encrypt(Pchk, c.key_schedule(
                        master_from_last_round_key(c, a.rounds, rcs, cand_rk)),
                        rounds=a.rounds)) == Cchk:
                    kappa = kk
                    found = True
                    break
            if not found:
                print("    no candidate combination reproduces the known pair")
        rk_last = [None if v is None else F.sub(0, v) for v in kappa]
        if free_words:
            print(f"    enumerated words: {free_words}; "
                  f"{trials} combinations tested against one known pair")
        got = tuple(rk_last) == tuple(rks[a.rounds])
        master = (master_from_last_round_key(c, a.rounds, rcs, rks[a.rounds])
                  if got else None)
        # independent check on a fresh plaintext/ciphertext pair
        Pv = tuple(int(v) for v in rng.integers(0, F.q, size=16))
        ok = (master is not None
              and tuple(c.encrypt(Pv, c.key_schedule(master), rounds=a.rounds))
              == tuple(c.encrypt(Pv, rks, rounds=a.rounds)))
        el = time.time() - t0
        print(f"  key {ki}: rank {rank}/{n_unknown}; rk^{a.rounds} recovered: {got}; "
              f"master key == truth: {master == tuple(K)}; "
              f"known-pair check: {ok}  ({el:.1f}s)")
        results.append({"key_index": ki, "rank": rank, "unknowns": n_unknown,
                        "decoupled": bool(spec["decoupled"]),
                        "free_words": free_words,
                        "last_round_key_ok": bool(got),
                        "master_key_ok": bool(master == tuple(K)),
                        "known_pair_ok": bool(ok), "elapsed_s": round(el, 1)})

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        fn = os.path.join(a.out, f"cp_attack_{a.instance}_r{a.rounds}_"
                                 f"{'_'.join(map(str, active))}.json")
        json.dump({"instance": a.instance, "rounds": a.rounds, "active": active,
                   "balanced_positions": pos, "unknowns": len(cols),
                   "kappa_monomials_per_block": len(kmons),
                   "moments": len(needed), "equations_per_structure": n_eq,
                   "structures": n_struct,
                   "data_log2": round(float(np.log2(n_struct * npts)), 2),
                   "keys": a.keys, "seed": a.seed, "results": results},
                  open(fn, "w"), indent=1)
        print("saved", fn)


if __name__ == "__main__":
    main()
