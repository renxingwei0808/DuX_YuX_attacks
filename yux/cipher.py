"""YuX block cipher (paper Algorithm 3) -- scalar + vectorised reference model.

Encryption with r rounds and round keys rk^0..rk^r (Algorithm 3):
    x = P + rk^0
    for i in 1..r-1:  x = SL(x); x = LP(x); x = x + rk^i
    x = SL(x); C = x + rk^r
Decryption is the exact inverse (subtract the round keys, LP_inv, S^{-1}).

The class mirrors `dux.cipher.DuX` method for method and signature for
signature, so every analysis script in this repository can be pointed at YuX
through `dux.registry.get_cipher(instance)` without a code change.  Two
DuX-specific concepts have no YuX counterpart and are kept as no-ops:

  * the key-dependent linear layer L0/L1 -- `lin.L(x, t, vec)` accepts and
    ignores `t`;
  * observation O2's equivalent fixed-L0 cipher -- `equivalent_fixedL0_keys`
    returns the round keys unchanged and the output rotation exponent T = 0,
    and `encrypt_fixedL0` is `encrypt`.

`decrypt_layers(C, rks, layers)` returns the state after the first `layers`
S^{-1} layers of decryption (rk^r first) -- the object every higher-order
differential experiment sums over.  `encrypt_layers` is its chosen-plaintext
mirror.
"""
from __future__ import annotations

from dux.field import make_field

from .params import INSTANCES, WORDS
from .sbox import S, S_inv, vS, vS_inv
from .linear import LinearLayer
from .keyschedule import key_expand, ks_inverse, round_constants


class YuX:
    def __init__(self, instance="yu2x-16", alpha=None, rounds=None,
                 ks_literal=False):
        spec = INSTANCES[instance]
        self.instance = instance
        self.F = make_field(spec["field"])
        self.alpha = spec["alpha"] if alpha is None else alpha
        self.r = spec["rounds"] if rounds is None else rounds
        self.ks_literal = ks_literal
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
        return key_expand(self.F, self.alpha, tuple(K), self.r, self.ks_literal)

    def key_schedule_inverse(self, rk_i, i):
        """rk^{i-1} (or, for i = 1 with ks_literal, the seed state)."""
        return ks_inverse(self.F, self.alpha, rk_i, i, self.r, self.ks_literal)

    def round_constants(self):
        return round_constants(self.F, self.alpha, self.r)

    def random_key(self, rng):
        return tuple(int(v) for v in rng.integers(0, self.F.q, size=WORDS))

    # --- full / reduced-round cipher --------------------------------------
    def encrypt(self, P, rks, rounds=None, vec=False):
        r = self.r if rounds is None else rounds
        assert len(rks) >= r + 1
        x = self.ARK(tuple(P), rks[0], vec)
        for i in range(1, r):
            x = self.SL(x, vec)
            x = self.lin.L(x, 0, vec)
            x = self.ARK(x, rks[i], vec)
        x = self.SL(x, vec)
        return self.ARK(x, rks[r], vec)

    def decrypt(self, C, rks, rounds=None, vec=False):
        r = self.r if rounds is None else rounds
        x = self.ARK_inv(tuple(C), rks[r], vec)
        x = self.SL_inv(x, vec)
        for i in range(r - 1, 0, -1):
            x = self.ARK_inv(x, rks[i], vec)
            x = self.lin.L_inv(x, 0, vec)
            x = self.SL_inv(x, vec)
        return self.ARK_inv(x, rks[0], vec)

    def decrypt_layers(self, C, rks, layers, rounds=None, vec=False,
                       yield_all=False):
        """State after `layers` S^{-1} layers of decrypting the `rounds`-round
        cipher (default: full).  An "l-layer distinguisher" is a zero-sum on
        this state."""
        r = self.r if rounds is None else rounds
        assert 1 <= layers <= r
        states = []
        x = self.ARK_inv(tuple(C), rks[r], vec)
        x = self.SL_inv(x, vec)
        states.append(x)
        for k in range(1, layers):
            i = r - k
            x = self.ARK_inv(x, rks[i], vec)
            x = self.lin.L_inv(x, 0, vec)
            x = self.SL_inv(x, vec)
            states.append(x)
        return states if yield_all else x

    def encrypt_layers(self, P, rks, layers, rounds=None, vec=False,
                       yield_all=False):
        """Mirror of `decrypt_layers` for the designers' chosen-plaintext
        model: the state after S layer k (1 <= k <= r) of `encrypt`."""
        r = self.r if rounds is None else rounds
        assert 1 <= layers <= r
        states = []
        x = self.ARK(tuple(P), rks[0], vec)
        x = self.SL(x, vec)
        states.append(x)
        for k in range(1, layers):
            x = self.lin.L(x, 0, vec)
            x = self.ARK(x, rks[k], vec)
            x = self.SL(x, vec)
            states.append(x)
        return states if yield_all else x

    # --- O2 analogue (trivial for YuX: there is no key-dependent layer) -----
    def equivalent_fixedL0_keys(self, rks, rounds=None):
        """YuX has a single linear layer, so the "equivalent fixed-L0 cipher"
        is the cipher itself: the round keys are unchanged and the output
        rotation exponent is T = 0."""
        r = self.r if rounds is None else rounds
        return [tuple(rk) for rk in rks[:r + 1]], 0

    def encrypt_fixedL0(self, P, rks, rounds=None, vec=False):
        return self.encrypt(P, rks, rounds, vec)
