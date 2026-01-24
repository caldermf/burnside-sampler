"""
Green polynomials Q_μ^λ(q) for counting flags in Springer fibers.
"""

from fractions import Fraction
from functools import lru_cache
from typing import Tuple, Dict
from collections import Counter

from partitions import partitions


def _z(lam: Tuple[int, ...]) -> Fraction:
    """Centralizer size z_λ = ∏(i^{m_i} · m_i!)."""
    counts = Counter(lam)
    result = Fraction(1)
    for part, mult in counts.items():
        fact = 1
        for k in range(1, mult + 1):
            fact *= k
        result *= part**mult * fact
    return result


def _binom(n: int, k: int) -> Fraction:
    """Binomial coefficient."""
    if k < 0 or k > n:
        return Fraction(0)
    num = den = 1
    for i in range(k):
        num *= (n - i)
        den *= (i + 1)
    return Fraction(num, den)


def _poly_mul(p1: list, p2: list) -> list:
    """Multiply two polynomials (coefficient lists)."""
    if not p1 or not p2:
        return [Fraction(0)]
    result = [Fraction(0)] * (len(p1) + len(p2) - 1)
    for i, a in enumerate(p1):
        for j, b in enumerate(p2):
            result[i + j] += a * b
    return result


@lru_cache(maxsize=None)
def green_X(lam: Tuple[int, ...], mu: Tuple[int, ...]) -> Tuple[Fraction, ...]:
    """Green polynomial X_λ^μ as coefficient tuple."""
    if sum(lam) != sum(mu):
        return (Fraction(0),)
    if len(lam) <= 1:
        return (Fraction(1),)
    
    lam0, lam_rest = lam[0], lam[1:]
    mu_counts = Counter(mu)
    n = sum(lam)
    
    # Generate sub-partitions of μ
    def subparts(items, idx=0, chosen=()):
        if idx == len(items):
            parts = sum(([s] * t for s, t in chosen), [])
            yield tuple(sorted(parts, reverse=True))
        else:
            s, m = items[idx]
            for t in range(m + 1):
                yield from subparts(items, idx + 1, chosen + ((s, t),))
    
    result = [Fraction(0)]
    
    for tau in subparts(list(mu_counts.items())):
        tau_counts = Counter(tau)
        binom_prod = Fraction(1)
        for s, m in mu_counts.items():
            binom_prod *= _binom(m, tau_counts.get(s, 0))
        if binom_prod == 0:
            continue
        
        rem = n - lam0 - sum(tau)
        for rho in partitions(rem):
            # z-inverse polynomial: (1/z_ρ) · ∏(1 - x^p)
            z_inv = [Fraction(1)]
            for p in rho:
                new = [Fraction(0)] * (len(z_inv) + p)
                for i, c in enumerate(z_inv):
                    new[i] += c
                    new[i + p] -= c
                z_inv = new
            z_inv = [c / _z(rho) for c in z_inv]
            
            concat = tuple(sorted(tau + rho, reverse=True))
            X_rec = list(green_X(lam_rest, concat))
            
            term = _poly_mul(z_inv, X_rec)
            sign = 1 if len(rho) % 2 == 0 else -1
            
            # Extend result if needed
            while len(result) < len(term):
                result.append(Fraction(0))
            for i, c in enumerate(term):
                result[i] += sign * binom_prod * c
    
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return tuple(result)


def green_Q(lam: Tuple[int, ...], mu: Tuple[int, ...]) -> Tuple[Fraction, ...]:
    """Q_λ^μ = involution of X_λ^μ (reverse coefficients)."""
    return tuple(reversed(green_X(lam, mu)))


def green_value(lam: Tuple[int, ...], q: int) -> int:
    """Q_{(1^n)}^λ(q): number of flags fixed by unipotent of type λ."""
    n = sum(lam)
    Q = green_Q(lam, (1,) * n)
    return int(sum(c * q**i for i, c in enumerate(Q)))


def precompute_green(n: int, q: int) -> Dict[Tuple[int, ...], int]:
    """Precompute green_value for all partitions up to n."""
    return {lam: green_value(lam, q) for k in range(n + 1) for lam in partitions(k)}
