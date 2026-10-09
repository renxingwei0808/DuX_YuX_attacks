"""Regression vectors for the paper instances of DuX and YuX.

The vectors are produced BY THIS REPOSITORY's reference models; they are not
external test vectors (none exist that agree with either paper).  Their only
purpose is to freeze the current semantics so that later experiments cannot
change a cipher by accident.  If a vector file has to be regenerated, the commit
must say why and which definitional statement of the paper forced it.

Usage:  python scripts/gen_vectors.py [--cipher dux|yux|both] [--seed 2026]
                                      [--count 5]
The cipher class is taken from `dux.registry.get_cipher`, so the same code
path serves both families.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dux.registry import get_cipher, instances  # noqa: E402
from dux.params import WORDS  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "vectors")


def official(family):
    return [k for k, v in instances(family).items() if v["official"]]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:                                    # pragma: no cover
        return "unknown"


def round_states(c, P, rks):
    """Encryption intermediate states: index 0 = after whitening (P + rk^0),
    index i = after round i (SL, LM, ARK) for 1 <= i <= r-1, index r = C."""
    states = []
    x = c.ARK(tuple(P), rks[0])
    states.append(x)
    for i in range(1, c.r):
        x = c.SL(x)
        x = c.lin.LM(x, rks[i])
        x = c.ARK(x, rks[i])
        states.append(x)
    x = c.SL(x)
    states.append(c.ARK(x, rks[c.r]))
    return states


def build(instance, seed, count):
    c = get_cipher(instance)
    rng = np.random.default_rng(seed)
    spec = instances()[instance]
    vectors = []
    for _ in range(count):
        K = c.random_key(rng)
        rks = c.key_schedule(K)
        P = tuple(int(v) for v in rng.integers(0, c.F.q, size=WORDS))
        states = round_states(c, P, rks)
        C = states[-1]
        assert c.encrypt(P, rks) == C
        assert c.decrypt(C, rks) == P
        dec = c.decrypt_layers(C, rks, c.r, yield_all=True)
        vectors.append({
            "K": list(K),
            "P": list(P),
            "C": list(C),
            "rk1": list(rks[1]),
            "round_keys": [list(rk) for rk in rks],
            "enc_states": [list(s) for s in states],
            "dec_layer_states": [list(s) for s in dec],
        })
    return {
        "instance": instance,
        "family": spec["family"],
        "field": spec["field"],
        "alpha": spec["alpha"],
        "rounds": spec["rounds"],
        "txor_word": 15 if spec["family"] == "dux" else None,
        "seed": seed,
        "generated_by": f"scripts/gen_vectors.py @ {git_commit()}",
        "note": ("self-generated regression vectors; "
                 "see docs/dux_specification.md") if spec["family"] == "dux"
        else ("self-generated regression vectors; "
              "see docs/yux_specification.md"),
        "vectors": vectors,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cipher", choices=("dux", "yux", "both"), default="both")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--count", type=int, default=None,
                    help="vectors per instance (default 5 for dux, 3 for yux)")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    fams = ("dux", "yux") if args.cipher == "both" else (args.cipher,)
    for fam in fams:
        count = args.count if args.count else (5 if fam == "dux" else 3)
        for inst in official(fam):
            data = build(inst, args.seed, count)
            short = inst.split("-", 1)[1] if fam == "dux" else inst
            path = os.path.join(OUT_DIR, f"{fam}_{short}.json")
            with open(path, "w") as fh:
                json.dump(data, fh, indent=1)
            print(f"wrote {os.path.relpath(path)}  ({count} vectors, "
                  f"{os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main()
