"""
Partition combinatorics for the sampling algorithm.
"""

from functools import lru_cache
from typing import Tuple, List, Dict
from collections import Counter


@lru_cache(maxsize=None)
def partitions(n: int) -> Tuple[Tuple[int, ...], ...]:
    """All partitions of n in decreasing order."""
    if n == 0:
        return ((),)
    result = []
    def gen(rem, maxp, cur):
        if rem == 0:
            result.append(tuple(cur))
        else:
            for p in range(min(rem, maxp), 0, -1):
                gen(rem - p, p, cur + [p])
    gen(n, n, [])
    return tuple(result)


def smaller_partitions(lam: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """R(λ): partitions from removing one box."""
    result = set()
    for i in range(len(lam)):
        new = list(lam)
        new[i] -= 1
        if new[i] == 0:
            new.pop(i)
        result.add(tuple(sorted(new, reverse=True)))
    return list(result)


def r_index(lam: Tuple[int, ...], smaller: Tuple[int, ...]) -> int:
    """Index of decremented part (1-indexed, rightmost if tied)."""
    c1, c2 = Counter(lam), Counter(smaller)
    for size in c1:
        if c1[size] > c2.get(size, 0):
            for i in range(len(lam) - 1, -1, -1):
                if lam[i] == size:
                    return i + 1
    raise ValueError("Invalid smaller partition")


def multiplicity(lam: Tuple[int, ...], r: int) -> int:
    """Multiplicity of part λ_r in λ."""
    return lam.count(lam[r - 1])


def E_size(lam: Tuple[int, ...], smaller: Tuple[int, ...], q: int) -> int:
    """|E^{λ'}(λ)| = q^r - q^{r-m}."""
    r = r_index(lam, smaller)
    m = multiplicity(lam, r)
    return q**r - q**(r - m)


def block_starts(lam: Tuple[int, ...]) -> List[int]:
    """Starting indices of Jordan blocks."""
    starts, off = [], 0
    for size in lam:
        starts.append(off)
        off += size
    return starts
