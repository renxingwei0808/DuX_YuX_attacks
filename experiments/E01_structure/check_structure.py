"""E01 -- structural facts about DuX that the attack plan relies on.

Prints a short report; every claim is also covered by tests/test_dux.py.
  O2  L1 = Rot_{-4} o L0, L1^{-1} = Rot_{+4} o L0^{-1}; key-dependent LM is
      equivalent to fixed L0 + rotated round keys + output block rotation.
  O3  L^{-1} maps block position p only to positions {p, p+1}  (YuX: all).
  O5  In characteristic 2 the key-schedule term 2*x0 vanishes: y1 - y2 = x0.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dux import DuX, INSTANCES  # noqa: E402
from dux.params import ROT_INV, ROT_FWD_BIN  # noqa: E402
from dux.linear import rotl  # noqa: E402
from dux.keyschedule import Rf  # noqa: E402


def main():
    print("O3: within-block position mixing (positions mod 4 reached by each rotation set)")
    for name, rots in [("DuX L0^-1", ROT_INV[0]), ("DuX L1^-1", ROT_INV[1]),
                       ("DuX L0 (fwd, F_2^n)", ROT_FWD_BIN[0]),
                       ("YuX L^-1", (0, 3, 4, 8, 9, 12, 14))]:
        print(f"  {name:22s} rotations {rots} -> positions {sorted(set(j % 4 for j in rots))}")

    print("\nO2: fixed-L0 equivalence on random keys/plaintexts")
    for inst in INSTANCES:
        c = DuX(inst)
        rng = np.random.default_rng(11)
        ok = True
        Ts = set()
        for _ in range(20):
            K = c.random_key(rng)
            rks = c.key_schedule(K)
            P = tuple(int(v) for v in rng.integers(0, c.F.q, size=16))
            rks2, T = c.equivalent_fixedL0_keys(rks)
            Ts.add(T % 4)
            ok &= c.encrypt(P, rks) == rotl(c.encrypt_fixedL0(P, rks2), -4 * T)
        print(f"  {inst:10s}: equivalence holds = {ok}; output rotations seen (T mod 4) = {sorted(Ts)}")

    print("\nO5: key schedule Rf in characteristic 2: y1 - y2 == x0 ?")
    for inst in ("dux-2^8", "dux-2^16", "dux-65537"):
        c = DuX(inst)
        rng = np.random.default_rng(12)
        X = tuple(int(v) for v in rng.integers(0, c.F.q, size=16))
        rc = tuple(int(v) for v in rng.integers(0, c.F.q, size=4))
        Y = Rf(c.F, c.alpha, X, rc)
        diff = tuple(c.F.sub(Y[4 + i], Y[8 + i]) for i in range(4))
        print(f"  {inst:10s}: y1 - y2 = {diff}, x0 = {X[:4]}  ->  equal: {diff == X[:4]}")


if __name__ == "__main__":
    main()
