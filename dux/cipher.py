"""DuX block cipher (paper Sect. 4, Algorithm 3) -- scalar reference model.

Encryption with r rounds and round keys rk^0..rk^r:
    x = P + rk^0
    for i in 1..r-1:  x = SL(x); x = L_{t(rk^i)}(x); x = x + rk^i
    x = SL(x); C = x + rk^r
Decryption is the exact inverse.

`decrypt_layers(C, rks, layers)` returns the intermediate state after the
first `layers` S^{-1} layers of decryption (rk^r first).  This is the
object all higher-order-differential experiments look at: an "l-layer
distinguisher" is a zero-sum on this state.

`encrypt_fixedL0` implements the equivalent cipher of observation O2:
all rounds use L0 and the round keys are block-rotated.
"""
from __future__ import annotations

from .params import INSTANCES, WORDS
from .field import make_field
from .sbox import S, S_inv, vS, vS_inv
from .linear import LinearLayer, rotl, t_xor
from .keyschedule import key_expand


class DuX:
    def __init__(self, instance="dux-65537", alpha=None, rounds=None):
        spec = INSTANCES[instance]
        self.instance = instance
        self.F = make_field(spec["field"])
        self.alpha = spec["alpha"] if alpha is None else alpha
        self.r = spec["rounds"] if rounds is None else rounds
        self.lin = LinearLayer(self.F)

    # --- components ---------------------------------------------------------
    def SL(self, x, vec=False):
        f = vS if vec else S
        out = ()
        for b in range(4):
            out += f(self.F, tuple(x[4 * b:4 * b + 4]), self.alpha)
        return out

    def SL_inv(self, x, vec=False):
        f = vS_inv if vec else S_inv
        out = ()
        for b in range(4):
            out += f(self.F, tuple(x[4 * b:4 * b + 4]), self.alpha)
        return out

    def ARK(self, x, rk, vec=False):
        add = self.F.vadd if vec else self.F.add
        return tuple(add(a, k) for a, k in zip(x, rk))

    def ARK_inv(self, x, rk, vec=False):
        sub = self.F.vsub if vec else self.F.sub
        return tuple(sub(a, k) for a, k in zip(x, rk))

    def key_schedule(self, K):
        return key_expand(self.F, self.alpha, tuple(K), self.r)

    def random_key(self, rng):
        return tuple(int(v) for v in rng.integers(0, self.F.q, size=WORDS))

    # --- full / reduced-round cipher --------------------------------------
    def encrypt(self, P, rks, rounds=None, vec=False):
        r = self.r if rounds is None else rounds
        assert len(rks) >= r + 1
        x = self.ARK(tuple(P), rks[0], vec)
        for i in range(1, r):
            x = self.SL(x, vec)
            x = self.lin.LM(x, rks[i], vec)
            x = self.ARK(x, rks[i], vec)
        x = self.SL(x, vec)
        return self.ARK(x, rks[r], vec)

    def decrypt(self, C, rks, rounds=None, vec=False):
        r = self.r if rounds is None else rounds
        x = self.ARK_inv(tuple(C), rks[r], vec)
        x = self.SL_inv(x, vec)
        for i in range(r - 1, 0, -1):
            x = self.ARK_inv(x, rks[i], vec)
            x = self.lin.LM_inv(x, rks[i], vec)
            x = self.SL_inv(x, vec)
        return self.ARK_inv(x, rks[0], vec)

    def decrypt_layers(self, C, rks, layers, rounds=None, vec=False, yield_all=False):
        """State after `layers` S^{-1} layers of decryption of the
        `rounds`-round cipher (default: full).  If yield_all, return the list
        of states after each layer (1..layers)."""
        r = self.r if rounds is None else rounds
        assert 1 <= layers <= r
        states = []
        x = self.ARK_inv(tuple(C), rks[r], vec)
        x = self.SL_inv(x, vec)
        states.append(x)
        for k in range(1, layers):
            i = r - k
            x = self.ARK_inv(x, rks[i], vec)
            x = self.lin.LM_inv(x, rks[i], vec)
            x = self.SL_inv(x, vec)
            states.append(x)
        return states if yield_all else x

    def encrypt_layers(self, P, rks, layers, rounds=None, vec=False, yield_all=False):
        """State after `layers` S layers of encryption of the `rounds`-round
        cipher (default: full).  Mirror image of decrypt_layers, used by the
        chosen-plaintext (designer CPA model) experiments in experiments/E10_cpa.
        The round structure is the one of `encrypt`:
            ARK(rk^0), then for i = 1..r-1:  S, LM(rk^i), ARK(rk^i), then S, ARK(rk^r),
        so the state after S layer k (1 <= k <= r) is what this returns."""
        r = self.r if rounds is None else rounds
        assert 1 <= layers <= r
        states = []
        x = self.ARK(tuple(P), rks[0], vec)
        x = self.SL(x, vec)
        states.append(x)
        for k in range(1, layers):
            x = self.lin.LM(x, rks[k], vec)
            x = self.ARK(x, rks[k], vec)
            x = self.SL(x, vec)
            states.append(x)
        return states if yield_all else x

    # --- O2: equivalent fixed-L0 cipher --------------------------------------
    def equivalent_fixedL0_keys(self, rks, rounds=None):
        """rk'_i = Rot_{4 T_i}(rk_i), T_i = t_1 + ... + t_i  (T_0 = 0), and
        the final output rotation exponent T = T_{r-1}."""
        r = self.r if rounds is None else rounds
        T = 0
        new = [tuple(rks[0])]
        for i in range(1, r):
            T += t_xor(rks[i])
            new.append(rotl(rks[i], 4 * T))
        new.append(rotl(rks[r], 4 * T))
        return new, T

    def encrypt_fixedL0(self, P, rks, rounds=None, vec=False):
        r = self.r if rounds is None else rounds
        x = self.ARK(tuple(P), rks[0], vec)
        for i in range(1, r):
            x = self.SL(x, vec)
            x = self.lin.L(x, 0, vec)
            x = self.ARK(x, rks[i], vec)
        x = self.SL(x, vec)
        return self.ARK(x, rks[r], vec)
