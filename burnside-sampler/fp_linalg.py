"""
Linear algebra over finite fields F_p (p prime).
"""

import numpy as np
import random
from typing import List, Tuple


def mod_inv(a: int, p: int) -> int:
    """Multiplicative inverse via Fermat: a^{-1} = a^{p-2}."""
    return pow(a % p, p - 2, p)


def mat_mul(A: np.ndarray, B: np.ndarray, p: int) -> np.ndarray:
    """Matrix multiplication mod p."""
    return (A.astype(object) @ B.astype(object) % p).astype(int)


def mat_vec(M: np.ndarray, v: np.ndarray, p: int) -> np.ndarray:
    """Matrix-vector multiplication mod p."""
    return (M.astype(object) @ v.astype(object) % p).astype(int)


def mat_inv(M: np.ndarray, p: int) -> np.ndarray:
    """Matrix inverse mod p via Gaussian elimination."""
    n = M.shape[0]
    aug = np.hstack([M.copy(), np.eye(n, dtype=int)]) % p

    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r, col] % p), None)
        if pivot is None:
            raise ValueError("Singular matrix")
        aug[[col, pivot]] = aug[[pivot, col]]
        aug[col] = aug[col] * mod_inv(int(aug[col, col]), p) % p
        for row in range(n):
            if row != col and aug[row, col]:
                aug[row] = (aug[row] - int(aug[row, col]) * aug[col]) % p

    return aug[:, n:].astype(int)


def rank(M: np.ndarray, p: int) -> int:
    """Matrix rank mod p."""
    M = M.copy() % p
    m, n = M.shape
    r = 0
    for col in range(n):
        pivot = next((i for i in range(r, m) if M[i, col] % p), None)
        if pivot is None:
            continue
        M[[r, pivot]] = M[[pivot, r]]
        M[r] = M[r] * mod_inv(int(M[r, col]), p) % p
        for i in range(m):
            if i != r and M[i, col]:
                M[i] = (M[i] - int(M[i, col]) * M[r]) % p
        r += 1
    return r


def in_span(v: np.ndarray, vectors: List[np.ndarray], p: int) -> bool:
    """Check if v is in span of vectors."""
    if not vectors:
        return not np.any(v % p)
    M = np.column_stack(vectors + [v]) % p
    return rank(M[:, :-1], p) == rank(M, p)


def jordan_form(A: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
    """
    Jordan form of unipotent matrix A.
    Returns (J, P, partition) where A = P @ J @ P^{-1}.
    """
    n = A.shape[0]
    N = (A - np.eye(n, dtype=int)) % p
    
    # Compute powers N^k
    N_pows = [np.eye(n, dtype=int)]
    for _ in range(n):
        nxt = mat_mul(N_pows[-1], N, p)
        N_pows.append(nxt)
        if not np.any(nxt):
            break
    
    # Get partition from kernel dimensions
    dims = [n - rank(Nk, p) for Nk in N_pows]
    partition = []
    for k in range(len(dims) - 1, 0, -1):
        count = (dims[k] - dims[k-1]) - sum(1 for b in partition if b >= k)
        partition.extend([k] * count)
    partition = tuple(sorted(partition, reverse=True))
    
    # Build Jordan basis
    P_cols, used = [], []
    for size in sorted(set(partition), reverse=True):
        for _ in range(partition.count(size)):
            v = _find_cyclic(N, N_pows, size, used, p, n)
            chain = []
            cur = v.copy()
            for _ in range(size):
                chain.append(cur.copy())
                cur = mat_vec(N, cur, p)
            P_cols.extend(chain[::-1])
            used.extend(chain[::-1])
    
    P = np.column_stack(P_cols) % p
    J = np.eye(n, dtype=int)
    off = 0
    for size in partition:
        for i in range(size - 1):
            J[off + i, off + i + 1] = 1
        off += size
    
    return J, P, partition


def _find_cyclic(N, N_pows, k, used, p, n):
    """Find v with N^{k-1}v ≠ 0, N^k v = 0, independent of used."""
    for _ in range(200):
        v = np.array([random.randint(0, p-1) for _ in range(n)])
        if not np.any(v % p):
            continue
        if k < len(N_pows) and np.any(mat_vec(N_pows[k], v, p)):
            continue
        if k > 0 and not np.any(mat_vec(N_pows[k-1], v, p)):
            continue
        chain, cur, ok = [], v.copy(), True
        for _ in range(k):
            if used and in_span(cur, used + chain, p):
                ok = False
                break
            chain.append(cur.copy())
            cur = mat_vec(N, cur, p)
        if ok:
            return v
    raise ValueError("Could not find cyclic vector")
