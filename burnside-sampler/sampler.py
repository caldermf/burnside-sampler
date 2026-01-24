"""
Burnside process sampler for GL_n(F_q).

One step of the Markov chain on S_n:
  1. Sample a ∈ U_w uniformly (stabilizer of wB)
  2. Sample flag from Springer fiber of a
  3. Convert flag back to permutation
"""

import numpy as np
import random
from typing import Tuple, List, Dict, Optional
from collections import Counter

from fp_linalg import mat_mul, mat_vec, mat_inv, mod_inv, jordan_form, in_span
from partitions import smaller_partitions, E_size, block_starts
from green_polynomials import precompute_green


def sample_stabilizer(w: Tuple[int, ...], q: int) -> np.ndarray:
    """
    Sample uniformly from U_w.
    U_w = {u ∈ U : u_ij = 0 for inversions (i,j) of w}.
    """
    n = len(w)
    perm = [x - 1 for x in w]
    inv = [0] * n
    for i, v in enumerate(perm):
        inv[v] = i
    
    M = np.eye(n, dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if inv[i] < inv[j]:  # not an inversion
                M[i, j] = random.randint(0, q - 1)
    return M


def sample_eigenvector(lam: Tuple[int, ...], smaller: Tuple[int, ...], q: int) -> np.ndarray:
    """
    Sample v ∈ E^{smaller}(lam) uniformly.
    """
    n = sum(lam)
    c1, c2 = Counter(lam), Counter(smaller)
    
    # Find decremented block size
    dec = next(s for s in c1 if c1[s] > c2.get(s, 0))
    
    starts = block_starts(lam)
    eq_idx = [s for s, sz in zip(starts, lam) if sz == dec]
    gt_idx = [s for s, sz in zip(starts, lam) if sz > dec]
    
    v = np.zeros(n, dtype=int)
    
    # Nonzero in eq_idx positions
    while True:
        for i in eq_idx:
            v[i] = random.randint(0, q - 1)
        if any(v[i] for i in eq_idx):
            break
    
    # Random in gt_idx positions
    for i in gt_idx:
        v[i] = random.randint(0, q - 1)
    
    return v


def quotient(A: np.ndarray, v: np.ndarray, q: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Quotient representation A|_{V/⟨v⟩}.
    Returns (A_quot, basis) where basis[0] = v.
    """
    n = A.shape[0]
    v = v % q
    
    basis = [v.copy()]
    for i in range(n):
        e = np.zeros(n, dtype=int)
        e[i] = 1
        if not in_span(e, basis, q):
            basis.append(e)
        if len(basis) == n:
            break
    
    P = np.column_stack(basis) % q
    P_inv = mat_inv(P, q)
    A_new = mat_mul(mat_mul(P_inv, A, q), P, q)
    
    return A_new[1:, 1:] % q, basis


def sample_flag(A: np.ndarray, q: int, G: Dict) -> List[np.ndarray]:
    """
    Sample flag from Springer fiber of unipotent A.
    """
    n = A.shape[0]
    if n == 1:
        return [np.array([1])]
    
    J, P, lam = jordan_form(A, q)
    
    # Sample smaller partition with weights ∝ |E| · Q
    smaller = smaller_partitions(lam)
    weights = [E_size(lam, s, q) * G[s] for s in smaller]
    total = sum(weights)
    probs = [w / total for w in weights]
    chosen = random.choices(smaller, weights=probs)[0]
    
    # Sample eigenvector and transform to original coords
    v_jordan = sample_eigenvector(lam, chosen, q)
    v = mat_vec(P, v_jordan, q)
    
    # Recurse
    A_quot, basis = quotient(A, v, q)
    sub = sample_flag(A_quot, q, G)
    
    # Lift
    flag = [v]
    for u in sub:
        lifted = sum(int(c) * b for c, b in zip(u, basis[1:])) % q
        flag.append(lifted.astype(int))
    return flag


def flag_to_perm(flag: List[np.ndarray], q: int) -> Tuple[int, ...]:
    """
    Convert flag to permutation via Gaussian elimination (rightmost pivots).
    """
    n = len(flag)
    A = np.array([list(v) for v in flag]) % q
    pivots = []
    
    for i in range(n):
        for r, pc in enumerate(pivots):
            if A[i, pc]:
                A[i] = (A[i] - int(A[i, pc]) * mod_inv(int(A[r, pc]), q) * A[r]) % q
        
        # Rightmost nonzero
        pivot = next((c for c in range(n - 1, -1, -1) if A[i, c] % q), None)
        if pivot is None:
            raise ValueError("Dependent vectors")
        pivots.append(pivot)
    
    return tuple(c + 1 for c in pivots)


def step(w: Tuple[int, ...], q: int, G: Optional[Dict] = None) -> Tuple[int, ...]:
    """One step of the Burnside process."""
    if G is None:
        G = precompute_green(len(w), q)
    a = sample_stabilizer(w, q)
    flag = sample_flag(a, q, G)
    return flag_to_perm(flag, q)


def run(n: int, q: int, steps: int = 1000, start: Tuple[int, ...] = None) -> List[Tuple[int, ...]]:
    """Run the chain for multiple steps."""
    G = precompute_green(n, q)
    w = start or tuple(range(1, n + 1))
    history = [w]
    for _ in range(steps):
        w = step(w, q, G)
        history.append(w)
    return history


if __name__ == "__main__":
    from collections import Counter
    import math
    
    q, n = 97, 4
    print(f"Burnside process on S_{n} with q={q}")
    
    G = precompute_green(n, q)
    print(f"\nGreen values Q_{{(1^{n})}}^λ({q}):")
    from partitions import partitions
    for lam in partitions(n):
        print(f"  {lam}: {G[lam]}")
    
    print(f"\nRunning 1000 steps...")
    history = run(n, q, 1000)
    counts = Counter(history)
    print(f"Visited {len(counts)}/{math.factorial(n)} permutations")
    
    print("\nFirst 10 steps:")
    for i, w in enumerate(history[:11]):
        print(f"  {i}: {w}")