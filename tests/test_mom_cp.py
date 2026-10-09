"""S8 -- the single-balanced-position CPA key recovery and its C kernel.

The 8-round DuX(2^16) attack uses a PARTIAL layer-7 pattern (`0010`, S6(c)),
so only S^{-1} coordinates 2 and 3 enter the equations.  Three things are
pinned here:
  * `used_sinv_coords` / `system_spec` restrict the unknown set accordingly,
    and leave the `1111` / `1110` systems of W11 untouched;
  * the second solving stage recovers kappa_3, which the linearisation alone
    cannot see (Lemma O8 in the encryption direction);
  * fast/mom_cp.c reproduces the numpy ciphertext moments bit for bit.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "E10_cpa"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E10_cpa", "fast"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E06_key_recovery_1round"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E08_boolean_degree_extension"))

from dux import DuX  # noqa: E402
import attack_cp_lastround as A  # noqa: E402

MOMCP = os.path.join(ROOT, "experiments", "E10_cpa", "fast", "mom_cp")


def test_used_sinv_coords():
    """L^{-1}'s rotation offsets mod 4 are {1,0,0,1,1}: position p reads
    coordinates p and p+1 only."""
    assert A.used_sinv_coords([2]) == [2, 3]
    assert A.used_sinv_coords([0]) == [0, 1]
    assert A.used_sinv_coords([3]) == [0, 3]
    # every pattern W11 used still sees all four coordinates, so its numbers
    # (53/57 unknowns) are unchanged by the restriction
    assert A.used_sinv_coords([0, 1, 2, 3]) == [0, 1, 2, 3]
    assert A.used_sinv_coords([0, 1, 2]) == [0, 1, 2, 3]


@pytest.mark.parametrize("instance,kmons,unknowns,moments",
                         [("dux-2^16", 13, 49, 17), ("dux-2^8", 13, 49, 17),
                          ("toy-257", 15, 57, 18)])
def test_single_position_system_size(instance, kmons, unknowns, moments):
    c = DuX(instance, rounds=8)
    spec = A.system_spec(c.F, c.alpha, [2])
    assert len(spec["kmons"]) == kmons
    assert len(spec["cols"]) == unknowns
    assert len(spec["needed"]) == moments
    assert spec["n_eq"] == 4 and spec["decoupled"] is False
    # kappa_3 is linear in every surviving monomial: that is what makes the
    # second stage a linear system in the four leftover words
    assert max(m[3] for m in spec["kmons"]) == 1


def test_full_pattern_system_size_unchanged():
    """W11's `1111` system: 14/13 kappa monomials per block, 53/57 unknowns."""
    for instance, kmons in [("dux-2^16", 14), ("dux-65537", 15)]:
        c = DuX(instance, rounds=7)
        spec = A.system_spec(c.F, c.alpha, [0, 1, 2, 3])
        assert len(spec["kmons"]) == kmons
        assert spec["decoupled"] is True and spec["n_eq"] == 16


@pytest.mark.parametrize("instance,rounds,active,nstruct",
                         [("dux-2^8", 5, [1], 11), ("toy-2^4", 4, [1, 5], 11)])
def test_single_position_attack_recovers_master_key(instance, rounds, active,
                                                    nstruct):
    """Characteristic 2, partial pattern `0010`: the whole last round key is
    pinned by stage 1 and the master key comes out."""
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "experiments", "E10_cpa",
                                      "attack_cp_lastround.py"),
         "--instance", instance, "--rounds", str(rounds),
         "--active", ",".join(map(str, active)), "--coords", "2",
         "--structures", str(nstruct), "--keys", "2"],
        capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.count("known-pair check: True") == 2, out.stdout


def test_stage2_recovers_kappa3_on_toy257():
    """F_p, partial pattern `0110` (toy-257 layer 4): the linearisation leaves
    kappa_3 free in all four blocks and the substitution stage pins it."""
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "experiments", "E10_cpa",
                                      "attack_cp_lastround.py"),
         "--instance", "toy-257", "--rounds", "5", "--active", "1,5",
         "--coords", "2", "--structures", "11", "--keys", "1"],
        capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "'stage2'" in out.stdout, out.stdout        # stage 1 did NOT suffice
    assert "'pinned': 4" in out.stdout, out.stdout
    assert "known-pair check: True" in out.stdout, out.stdout


def test_stage2_declines_when_every_position_is_balanced():
    """If Y is balanced at ALL positions, the kappa_3 coefficient is
    [L(sum_P Y)]_{4b} = 0 and the second stage has nothing to solve -- the
    honest outcome is "undetermined", not a wrong key."""
    import numpy as np
    c = DuX("dux-2^16", rounds=7)
    F = c.F
    spec = A.system_spec(F, c.alpha, [2])
    rng = np.random.default_rng(2026)
    K = c.random_key(rng)
    rks = c.key_schedule(K)
    rows, rhs, blk = [], [], {b: [] for b in range(4)}
    for st in range(13):
        mom = A.structure_moments(c, rks, 7, [1], 7000 + st, spec["needed"])
        A.structure_rows(spec, F, [2], mom, rows, rhs, blk)
    kappa, rank, _, free_words, _ = A.solve_system(spec, F, rows, blk)
    assert [i % 4 for i, v in enumerate(kappa) if v is None] == [3, 3, 3, 3]
    for b in range(4):                       # kappa_0..2 ARE recovered
        for i in range(3):
            assert kappa[4 * b + i] == F.sub(0, rks[7][4 * b + i])


@pytest.mark.skipif(not os.path.exists(MOMCP),
                    reason="fast/mom_cp not built (make -C experiments/E10_cpa/fast)")
@pytest.mark.parametrize("instance,rounds,active,dim",
                         [("dux-2^8", 5, [1], 8), ("toy-2^4", 4, [1, 5], 4),
                          ("dux-2^16", 8, [1, 5], 6)])
def test_mom_cp_matches_numpy(instance, rounds, active, dim):
    import attack_cp_lastround_fast as AF
    c = DuX(instance, rounds=rounds)
    spec = A.system_spec(c.F, c.alpha, [2])
    job, K, rks, consts, vals = AF.build_job(c, rounds, active, dim, 2,
                                             spec["needed"], 11)
    path = os.path.join("/tmp", f"duxcpmom_test_{os.getpid()}.bin")
    open(path, "wb").write(job)
    try:
        proc = subprocess.run([MOMCP, path], capture_output=True, text=True,
                              timeout=600)
        assert proc.returncode == 0, proc.stderr[-2000:]
        moms = AF.parse_moments(proc.stdout, 2, spec["needed"])
    finally:
        os.unlink(path)
    for st in range(2):
        ref = AF.python_moments_cp(c, rks, rounds, active, dim, consts[st],
                                   vals[st], spec["needed"])
        for b in range(4):
            assert ref[b] == moms[st][b], f"structure {st} block {b}"
