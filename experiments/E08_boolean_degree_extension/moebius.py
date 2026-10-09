"""W8 (E08) -- fast Moebius transform: truth table -> ANF, degree, monomials.

A Boolean function on m variables is a numpy uint8 array of length 2^m
(index = the variable assignment, bit i of the index = variable i).  The
Moebius transform is its own inverse and computes the ANF in place in
m passes of 2^{m-1} XORs each; m <= 24 needs 16 MB and runs in a few
seconds.

    anf = moebius(tt)            # anf[u] = coefficient of prod_{i in u} x_i
    d   = anf_degree(anf)        # max Hamming weight of a monomial present
    h   = degree_histogram(anf)  # how many monomials of each degree
"""
from __future__ import annotations

import numpy as np

_POPCOUNT16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


def popcount(idx: np.ndarray) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    out = np.zeros(idx.shape, dtype=np.int64)
    x = idx.copy()
    while True:
        out += _POPCOUNT16[(x & 0xFFFF).astype(np.int64)]
        x >>= 16
        if not np.any(x):
            break
    return out


def moebius(tt: np.ndarray) -> np.ndarray:
    """In-place-style Moebius transform of a truth table of length 2^m."""
    a = np.array(tt, dtype=np.uint8, copy=True)
    n = a.size
    m = n.bit_length() - 1
    assert n == 1 << m, "truth table length must be a power of two"
    a = a.reshape(-1)
    for i in range(m):
        step = 1 << i
        v = a.reshape(-1, 2 * step)
        v[:, step:] ^= v[:, :step]
    return a


def anf_degree(anf: np.ndarray) -> int:
    idx = np.nonzero(anf)[0]
    if len(idx) == 0:
        return -1
    return int(popcount(idx).max())


def degree_histogram(anf: np.ndarray) -> dict:
    idx = np.nonzero(anf)[0]
    if len(idx) == 0:
        return {}
    hw = popcount(idx)
    vals, cnt = np.unique(hw, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnt)}


def bits_of_words(words, nbits):
    """Pack a tuple of field elements into one integer index (word 0 lowest)."""
    idx = 0
    for i, w in enumerate(words):
        idx |= int(w) << (nbits * i)
    return idx


def unpack_index(idx: np.ndarray, nwords: int, nbits: int):
    """Inverse of bits_of_words for a whole array of indices."""
    mask = (1 << nbits) - 1
    return [(idx >> (nbits * i)) & mask for i in range(nwords)]
