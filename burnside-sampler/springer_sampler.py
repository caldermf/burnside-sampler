"""
Springer Fiber Sampling for the Burnside Process on GL_n(F_q)
=============================================================

Pure Python implementation (no Sage dependency).

This implements the sampling algorithm from Section 5 of Diaconis-Morton:
a Markov chain on S_n arising from the action of unipotent upper-triangular
matrices U on the flag variety G/B for G = GL_n(F_q).

Algorithm (one step of chain from permutation w):
    1. Sample a ∈ U_w uniformly (the stabilizer of wB)
    2. Compute Jordan form of a (always unipotent, so eigenvalue 1)
    3. Sample a flag V_• from the Springer fiber of a:
       - Weighted sampling on R(λ) with weights ∝ |E^λ'(λ)| · Q_{(1^n)}^{λ'}(q)
       - Sample v ∈ E^{λ'}(λ) uniformly, recurse on quotient V/⟨v⟩
    4. Convert flag to permutation via Bruhat decomposition (Gaussian elimination)

All arithmetic is over GF(q) for prime q using modular arithmetic.

Usage:
    history = run_chain(n=4, q=5, steps=1000)
    w_new = next_step(w=(1,2,3,4), q=5)
"""

import numpy as np
from fractions import Fraction
from functools import lru_cache
from typing import List, Tuple, Dict, Optional
import random

# =============================================================================
# Default Parameters
# =============================================================================

Q_VAL = 5   # Field size (must be prime)
N_VAL = 4   # Dimension (S_n)


# =============================================================================
# Modular Arithmetic over GF(q)
# =============================================================================

def mod_inv(a: int, p: int) -> int:
    """
    Multiplicative inverse of a mod p using Fermat's little theorem.
    For prime p: a^{-1} ≡ a^{p-2} (mod p)
    """
    a = a % p
    if a == 0:
        raise ValueError("Cannot invert zero")
    return pow(a, p - 2, p)


def mat_mod(M: np.ndarray, p: int) -> np.ndarray:
    """Reduce matrix entries mod p."""
    return np.mod(M, p).astype(int)


def mat_mul(A: np.ndarray, B: np.ndarray, p: int) -> np.ndarray:
    """Matrix multiplication over GF(p)."""
    # Use Python ints to avoid overflow, then reduce
    result = np.zeros((A.shape[0], B.shape[1]), dtype=object)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            result[i, j] = sum(int(A[i, k]) * int(B[k, j]) for k in range(A.shape[1])) % p
    return result.astype(int)


def mat_vec_mul(M: np.ndarray, v: np.ndarray, p: int) -> np.ndarray:
    """Matrix-vector multiplication over GF(p)."""
    result = np.zeros(M.shape[0], dtype=int)
    for i in range(M.shape[0]):
        result[i] = sum(int(M[i, j]) * int(v[j]) for j in range(M.shape[1])) % p
    return result


def mat_inv(M: np.ndarray, p: int) -> np.ndarray:
    """
    Matrix inverse over GF(p) via Gaussian elimination.
    Returns M^{-1} such that M @ M^{-1} ≡ I (mod p).
    """
    n = M.shape[0]
    # Augment [M | I]
    aug = np.hstack([M.copy(), np.eye(n, dtype=int)])
    aug = mat_mod(aug, p)

    for col in range(n):
        # Find nonzero pivot
        pivot_row = None
        for row in range(col, n):
            if aug[row, col] % p != 0:
                pivot_row = row
                break
        if pivot_row is None:
            raise ValueError("Matrix is singular mod p")

        # Swap rows
        aug[[col, pivot_row]] = aug[[pivot_row, col]]

        # Scale pivot row so pivot = 1
        inv_pivot = mod_inv(int(aug[col, col]), p)
        aug[col] = mat_mod(aug[col] * inv_pivot, p)

        # Eliminate column
        for row in range(n):
            if row != col and aug[row, col] != 0:
                factor = int(aug[row, col])
                aug[row] = mat_mod(aug[row] - factor * aug[col], p)

    return aug[:, n:].astype(int)


# =============================================================================
# Green Polynomials Q_{(1^n)}^λ(q)
#
# These count the number of complete flags fixed by a unipotent element
# of Jordan type λ. Computed recursively over Q, then evaluated at q.
# =============================================================================

@lru_cache(maxsize=None)
def partitions(n: int) -> Tuple[Tuple[int, ...], ...]:
    """Generate all partitions of n as tuples in decreasing order."""
    if n == 0:
        return ((),)
    result = []
    def generate(remaining, max_part, current):
        if remaining == 0:
            result.append(tuple(current))
            return
        for part in range(min(remaining, max_part), 0, -1):
            generate(remaining - part, part, current + [part])
    generate(n, n, [])
    return tuple(result)


def z_lambda(partition: Tuple[int, ...]) -> Fraction:
    """
    Centralizer size: z_λ = ∏_i (i^{m_i} · m_i!)
    where m_i = multiplicity of i in λ.
    """
    from collections import Counter
    counts = Counter(partition)
    result = Fraction(1)
    factorial = 1
    for i, m in counts.items():
        for j in range(1, m + 1):
            factorial = 1
            for k in range(1, j + 1):
                factorial *= k
        result *= Fraction(i ** m * factorial)
        factorial = 1
        for k in range(1, m + 1):
            factorial *= k
        result *= Fraction(factorial)
    # Recompute properly
    result = Fraction(1)
    for part, mult in counts.items():
        result *= part ** mult
        fact = 1
        for k in range(1, mult + 1):
            fact *= k
        result *= fact
    return result


@lru_cache(maxsize=None)
def green_poly_X(lam: Tuple[int, ...], mu: Tuple[int, ...]) -> Tuple[Fraction, ...]:
    """
    Compute Green polynomial X_λ^μ as coefficient tuple.
    Returns tuple of coefficients [a_0, a_1, ..., a_d] representing a_0 + a_1*x + ...
    """
    n_lam = sum(lam)
    n_mu = sum(mu)
    
    if n_lam != n_mu:
        return (Fraction(0),)
    
    if len(lam) <= 1:
        return (Fraction(1),)
    
    lam_first = lam[0]
    lam_rest = lam[1:]
    
    # Get all sub-partitions of mu
    from collections import Counter
    mu_counts = Counter(mu)
    
    def subpartitions(counts_items, idx=0, chosen=None):
        if chosen is None:
            chosen = []
        if idx == len(counts_items):
            parts = []
            for size, take in chosen:
                parts.extend([size] * take)
            yield tuple(sorted(parts, reverse=True))
            return
        size, mult = counts_items[idx]
        for take in range(mult + 1):
            yield from subpartitions(counts_items, idx + 1, chosen + [(size, take)])
    
    counts_list = list(mu_counts.items())
    
    result_poly = [Fraction(0)] * (n_lam + 1)
    
    for tau in subpartitions(counts_list):
        tau_counts = Counter(tau)
        
        # Binomial product
        binom_prod = Fraction(1)
        for s, m in mu_counts.items():
            t = tau_counts.get(s, 0)
            # Compute binomial(m, t)
            if t > m:
                binom_prod = Fraction(0)
                break
            num = 1
            for i in range(t):
                num *= (m - i)
            den = 1
            for i in range(1, t + 1):
                den *= i
            binom_prod *= Fraction(num, den)
        
        if binom_prod == 0:
            continue
        
        remaining_sum = n_lam - lam_first - sum(tau)
        
        for rho in partitions(remaining_sum):
            # Compute z_inverse(rho) * X(lam_rest, tau + rho)
            concat = tuple(sorted(tau + rho, reverse=True))
            
            # z_inverse polynomial: (1/z_rho) * prod_{p in rho} (x^p - 1)
            z_rho = z_lambda(rho)
            
            # Product polynomial for (x^p - 1) terms
            prod_poly = [Fraction(1)]
            for p in rho:
                # Multiply by (x^p - 1)
                new_poly = [Fraction(0)] * (len(prod_poly) + p)
                for i, c in enumerate(prod_poly):
                    new_poly[i] -= c
                    new_poly[i + p] += c
                prod_poly = new_poly
            
            # Multiply by 1/z_rho
            prod_poly = [c / z_rho for c in prod_poly]
            
            # Get recursive result
            X_recursive = green_poly_X(lam_rest, concat)
            
            # Multiply polynomials
            term_poly = [Fraction(0)] * (len(prod_poly) + len(X_recursive) - 1)
            for i, c1 in enumerate(prod_poly):
                for j, c2 in enumerate(X_recursive):
                    term_poly[i + j] += c1 * c2
            
            # Multiply by binom_prod
            term_poly = [c * binom_prod for c in term_poly]
            
            # Sign: (-1)^{len(rho)}
            sign = 1 if len(rho) % 2 == 0 else -1
            
            # Add to result
            while len(result_poly) < len(term_poly):
                result_poly.append(Fraction(0))
            for i, c in enumerate(term_poly):
                result_poly[i] += sign * c
    
    # Trim trailing zeros
    while len(result_poly) > 1 and result_poly[-1] == 0:
        result_poly.pop()
    
    return tuple(result_poly)


def green_poly_Q(lam: Tuple[int, ...], mu: Tuple[int, ...]) -> Tuple[Fraction, ...]:
    """
    Compute Green polynomial Q_λ^μ = involution of X_λ^μ.
    Involution: if X = a_0 + a_1*x + ... + a_d*x^d, then Q = a_d + a_{d-1}*x + ... + a_0*x^d
    """
    X = green_poly_X(lam, mu)
    return tuple(reversed(X))


def green_polynomial_value(partition: Tuple[int, ...], q: int) -> int:
    """
    Compute Q_{(1^n)}^λ(q) - the number of flags fixed by a unipotent of type λ.
    """
    n = sum(partition)
    mu = tuple([1] * n)  # (1, 1, ..., 1)
    Q_coeffs = green_poly_Q(partition, mu)
    # Evaluate polynomial at q
    result = Fraction(0)
    for i, c in enumerate(Q_coeffs):
        result += c * (q ** i)
    return int(result)


# =============================================================================
# Partition Combinatorics (Section 5.1 of paper)
# =============================================================================

def smaller_partitions(partition: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """
    R(λ): Set of partitions obtained by removing one box from λ.
    Each is formed by decrementing one part by 1.
    """
    result = set()
    for i in range(len(partition)):
        new_parts = list(partition)
        new_parts[i] -= 1
        if new_parts[i] == 0:
            new_parts.pop(i)
        result.add(tuple(sorted(new_parts, reverse=True)))
    return list(result)


def r_index(partition: Tuple[int, ...], smaller: Tuple[int, ...]) -> int:
    """
    r(λ, λ'): Index of the part decremented (1-indexed).
    Convention: maximal index among parts of equal size.
    """
    # Find which part size was decremented
    from collections import Counter
    c1, c2 = Counter(partition), Counter(smaller)
    
    # Find the size that decreased
    for size in c1:
        diff = c1[size] - c2.get(size, 0)
        if diff > 0:
            # This size was decremented
            # Find the maximal index with this part size
            for i in range(len(partition) - 1, -1, -1):
                if partition[i] == size:
                    return i + 1  # 1-indexed
    raise ValueError("Not a valid smaller partition")


def multiplicity(partition: Tuple[int, ...], r: int) -> int:
    """
    m(λ, λ'): Multiplicity of part λ_r in λ.
    """
    part_size = partition[r - 1]  # r is 1-indexed
    return partition.count(part_size)


def flag_weight(partition: Tuple[int, ...], smaller: Tuple[int, ...], q: int) -> int:
    """
    Weight for sampling: |E^{λ'}(λ)| = q^r - q^{r-m}
    where r = r(λ, λ') and m = m(λ, λ').
    """
    r = r_index(partition, smaller)
    m = multiplicity(partition, r)
    return q ** r - q ** (r - m)


def compute_sampling_weights(partition: Tuple[int, ...], q: int) -> Dict[Tuple[int, ...], float]:
    """
    Compute normalized sampling weights for each λ' ∈ R(λ).
    Weight(λ') ∝ |E^{λ'}(λ)| · Q_{(1^n)}^{λ'}(q)
    """
    smaller = smaller_partitions(partition)
    weights = {}
    total = 0
    
    for s in smaller:
        w = flag_weight(partition, s, q) * green_polynomial_value(s, q)
        weights[s] = w
        total += w
    
    # Normalize
    return {s: w / total for s, w in weights.items()}


# =============================================================================
# Jordan Form for Unipotent Matrices
#
# For unipotent A (all eigenvalues = 1), find P such that P^{-1}AP = J
# where J is in Jordan normal form with blocks ordered by decreasing size.
# =============================================================================

def jordan_form(A: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
    """
    Compute Jordan form of unipotent matrix A over GF(p).
    
    Returns:
        J: Jordan normal form matrix
        P: Transformation matrix (A = P @ J @ P^{-1})
        partition: Tuple of block sizes in decreasing order
    
    For unipotent A, we have A = I + N where N is nilpotent.
    """
    n = A.shape[0]
    N = mat_mod(A - np.eye(n, dtype=int), p)
    
    # Compute kernels of N, N^2, N^3, ... to find Jordan structure
    # dim(ker(N^k)) - dim(ker(N^{k-1})) tells us about block sizes
    
    def kernel_basis(M: np.ndarray, p: int) -> List[np.ndarray]:
        """Find basis for kernel of M over GF(p)."""
        m, n_cols = M.shape
        # Row reduce M
        M = mat_mod(M.copy(), p)
        pivot_cols = []
        row = 0
        
        for col in range(n_cols):
            # Find pivot
            pivot_row = None
            for r in range(row, m):
                if M[r, col] % p != 0:
                    pivot_row = r
                    break
            if pivot_row is None:
                continue
            
            # Swap
            M[[row, pivot_row]] = M[[pivot_row, row]]
            pivot_cols.append(col)
            
            # Scale
            inv_piv = mod_inv(int(M[row, col]), p)
            M[row] = mat_mod(M[row] * inv_piv, p)
            
            # Eliminate
            for r in range(m):
                if r != row and M[r, col] != 0:
                    M[r] = mat_mod(M[r] - int(M[r, col]) * M[row], p)
            row += 1
        
        # Free variables give kernel basis
        free_cols = [c for c in range(n_cols) if c not in pivot_cols]
        basis = []
        
        for fc in free_cols:
            v = np.zeros(n_cols, dtype=int)
            v[fc] = 1
            for i, pc in enumerate(pivot_cols):
                v[pc] = (-int(M[i, fc])) % p
            basis.append(v)
        
        return basis
    
    # Compute nilpotent index and kernel dimensions
    N_power = np.eye(n, dtype=int)
    kernels = [kernel_basis(N_power, p)]  # ker(N^0) = {0}, represented by empty basis
    dims = [0]
    
    for k in range(1, n + 1):
        N_power = mat_mul(N_power, N, p)
        ker = kernel_basis(N_power, p)
        kernels.append(ker)
        dims.append(len(ker))
        if dims[-1] == n:
            break
    
    # Jordan block sizes: number of blocks of size >= k is dims[k] - dims[k-1]
    # So block sizes are determined by differences
    block_sizes = []
    for k in range(len(dims) - 1, 0, -1):
        count = dims[k] - dims[k - 1]
        block_sizes.extend([k] * (count - sum(1 for b in block_sizes if b >= k)))
    
    # Recompute properly
    block_sizes = []
    max_k = len(dims) - 1
    num_blocks_geq = [0] * (max_k + 2)
    for k in range(1, max_k + 1):
        num_blocks_geq[k] = dims[k] - dims[k - 1]
    
    for size in range(max_k, 0, -1):
        count = num_blocks_geq[size] - sum(1 for b in block_sizes if b >= size)
        block_sizes.extend([size] * count)
    
    partition = tuple(sorted(block_sizes, reverse=True))
    
    # Build transformation matrix P using generalized eigenvectors
    # For each block of size k, we need a chain: v, Nv, N^2v, ..., N^{k-1}v
    # where N^{k-1}v ≠ 0 but N^k v = 0
    
    P_cols = []
    used_space = []  # Track vectors we've used
    
    # Process blocks from largest to smallest
    for block_size in sorted(set(partition), reverse=True):
        count_needed = partition.count(block_size)
        count_found = 0
        
        # Find vectors in ker(N^{block_size}) but not in ker(N^{block_size-1})
        ker_k = kernels[block_size] if block_size < len(kernels) else kernels[-1]
        ker_k_minus_1 = kernels[block_size - 1] if block_size > 0 else []
        
        for v in ker_k:
            if count_found >= count_needed:
                break
            
            # Check v is not in span of ker(N^{k-1}) + used_space
            combined = ker_k_minus_1 + used_space
            if not in_span(v, combined, p):
                # Check that N^{k-1}v ≠ 0
                Nkv = v.copy()
                for _ in range(block_size - 1):
                    Nkv = mat_vec_mul(N, Nkv, p)
                
                if np.any(Nkv % p != 0):
                    # Found a good starting vector
                    # Build chain: N^{k-1}v, N^{k-2}v, ..., Nv, v (for Jordan block)
                    chain = []
                    current = v.copy()
                    for _ in range(block_size):
                        chain.append(current.copy())
                        current = mat_vec_mul(N, current, p)
                    
                    # Reverse so we go from N^{k-1}v to v
                    chain = chain[::-1]
                    P_cols.extend(chain)
                    used_space.extend(chain)
                    count_found += 1
        
        # If we didn't find enough, try random combinations
        while count_found < count_needed:
            # Generate random vector in ker(N^k)
            if ker_k:
                coeffs = [random.randint(0, p - 1) for _ in ker_k]
                v = sum(c * vec for c, vec in zip(coeffs, ker_k)) % p
                v = mat_mod(v.reshape(-1), p)
                
                combined = ker_k_minus_1 + used_space
                if not in_span(v, combined, p) or not combined:
                    Nkv = v.copy()
                    for _ in range(block_size - 1):
                        Nkv = mat_vec_mul(N, Nkv, p)
                    
                    if np.any(Nkv % p != 0):
                        chain = []
                        current = v.copy()
                        for _ in range(block_size):
                            chain.append(current.copy())
                            current = mat_vec_mul(N, current, p)
                        chain = chain[::-1]
                        P_cols.extend(chain)
                        used_space.extend(chain)
                        count_found += 1
            else:
                break
    
    # Build P matrix
    if len(P_cols) != n:
        # Fallback: use a simpler algorithm
        P, J, partition = jordan_form_simple(A, p)
        return J, P, partition
    
    P = np.column_stack(P_cols)
    P = mat_mod(P, p)
    
    # Build J matrix
    J = np.eye(n, dtype=int)
    offset = 0
    for size in partition:
        for i in range(size - 1):
            J[offset + i, offset + i + 1] = 1
        offset += size
    
    return J, P, partition


def in_span(v: np.ndarray, vectors: List[np.ndarray], p: int) -> bool:
    """Check if v is in the span of vectors over GF(p)."""
    if not vectors:
        return np.all(v % p == 0)
    
    # Build matrix and try to solve for coefficients
    M = np.column_stack(vectors)
    m, n_cols = M.shape
    
    # Augment with v
    aug = np.column_stack([M, v.reshape(-1, 1)])
    aug = mat_mod(aug, p)
    
    # Row reduce
    row = 0
    for col in range(n_cols):
        pivot_row = None
        for r in range(row, m):
            if aug[r, col] % p != 0:
                pivot_row = r
                break
        if pivot_row is None:
            continue
        
        aug[[row, pivot_row]] = aug[[pivot_row, row]]
        inv_piv = mod_inv(int(aug[row, col]), p)
        aug[row] = mat_mod(aug[row] * inv_piv, p)
        
        for r in range(m):
            if r != row and aug[r, col] != 0:
                aug[r] = mat_mod(aug[r] - int(aug[r, col]) * aug[row], p)
        row += 1
    
    # Check if last column is in span
    # v is in span iff after row reduction, the last column has no pivot
    for r in range(row, m):
        if aug[r, -1] % p != 0:
            return False
    return True


def jordan_form_simple(A: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
    """
    Simpler Jordan form algorithm for unipotent matrices.
    Uses a more direct approach to find the transformation matrix.
    """
    n = A.shape[0]
    N = mat_mod(A - np.eye(n, dtype=int), p)
    
    # Find Jordan block structure
    N_power = np.eye(n, dtype=int)
    dims = [0]
    
    for k in range(1, n + 1):
        N_power = mat_mul(N_power, N, p)
        # Compute rank
        rank = matrix_rank(N_power, p)
        dims.append(n - rank)
        if dims[-1] == n:
            break
    
    # Compute block sizes
    block_sizes = []
    for k in range(1, len(dims)):
        new_blocks = (dims[k] - dims[k-1]) - sum(1 for b in block_sizes if b >= k)
        block_sizes.extend([k] * new_blocks)
    
    partition = tuple(sorted(block_sizes, reverse=True))
    
    # Build J
    J = np.eye(n, dtype=int)
    offset = 0
    for size in partition:
        for i in range(size - 1):
            J[offset + i, offset + i + 1] = 1
        offset += size
    
    # For transformation matrix, use a constructive approach
    P = find_jordan_basis(N, partition, p)
    
    return J, P, partition


def matrix_rank(M: np.ndarray, p: int) -> int:
    """Compute rank of matrix over GF(p)."""
    M = mat_mod(M.copy(), p)
    m, n_cols = M.shape
    rank = 0
    
    for col in range(n_cols):
        pivot_row = None
        for r in range(rank, m):
            if M[r, col] % p != 0:
                pivot_row = r
                break
        if pivot_row is None:
            continue
        
        M[[rank, pivot_row]] = M[[pivot_row, rank]]
        inv_piv = mod_inv(int(M[rank, col]), p)
        M[rank] = mat_mod(M[rank] * inv_piv, p)
        
        for r in range(m):
            if r != rank and M[r, col] != 0:
                M[r] = mat_mod(M[r] - int(M[r, col]) * M[rank], p)
        rank += 1
    
    return rank


def find_jordan_basis(N: np.ndarray, partition: Tuple[int, ...], p: int) -> np.ndarray:
    """
    Find transformation matrix P for Jordan form.
    Returns P such that P^{-1} (I+N) P = J.
    """
    n = N.shape[0]
    
    # For each block size k, find vectors v where N^{k-1}v ≠ 0 but N^k v = 0
    P_cols = []
    used_flags = np.zeros(n, dtype=bool)
    
    # Compute powers of N
    N_powers = [np.eye(n, dtype=int)]
    for k in range(1, n + 1):
        N_powers.append(mat_mul(N_powers[-1], N, p))
        if np.all(N_powers[-1] % p == 0):
            break
    
    # Process blocks by size (largest first for cleaner basis)
    sizes_to_process = sorted(set(partition), reverse=True)
    
    for block_size in sizes_to_process:
        count_needed = partition.count(block_size)
        
        for _ in range(count_needed):
            # Find v in ker(N^k) \ ker(N^{k-1})
            v = find_cyclic_vector(N, N_powers, block_size, P_cols, p)
            if v is None:
                # Use random search
                v = random_cyclic_vector(N, N_powers, block_size, p, n)
            
            # Build chain
            chain = []
            current = v.copy()
            for j in range(block_size):
                chain.append(current.copy())
                current = mat_vec_mul(N, current, p)
            
            # Add in reverse order (eigenvector first)
            P_cols.extend(chain[::-1])
    
    P = np.column_stack(P_cols) if P_cols else np.eye(n, dtype=int)
    return mat_mod(P, p)


def find_cyclic_vector(N, N_powers, k, existing_cols, p):
    """Find a vector v with N^{k-1}v ≠ 0 and N^k v = 0."""
    n = N.shape[0]
    
    # Try standard basis vectors first
    for i in range(n):
        v = np.zeros(n, dtype=int)
        v[i] = 1
        
        # Check N^k v = 0
        if k < len(N_powers):
            Nkv = mat_vec_mul(N_powers[k], v, p)
            if np.any(Nkv % p != 0):
                continue
        
        # Check N^{k-1} v ≠ 0
        if k - 1 < len(N_powers):
            Nk1v = mat_vec_mul(N_powers[k-1], v, p)
            if np.all(Nk1v % p == 0):
                continue
        elif k > 1:
            continue
        
        # Check not in span of existing
        if existing_cols and in_span(v, existing_cols, p):
            continue
        
        return v
    
    return None


def random_cyclic_vector(N, N_powers, k, p, n):
    """Find cyclic vector via random search."""
    for _ in range(100):
        v = np.array([random.randint(0, p-1) for _ in range(n)], dtype=int)
        
        if np.all(v % p == 0):
            continue
        
        # Check N^k v = 0
        if k < len(N_powers):
            Nkv = mat_vec_mul(N_powers[k], v, p)
            if np.any(Nkv % p != 0):
                continue
        
        # Check N^{k-1} v ≠ 0
        if k - 1 < len(N_powers):
            Nk1v = mat_vec_mul(N_powers[k-1], v, p)
            if np.all(Nk1v % p == 0):
                continue
        
        return v
    
    # Fallback
    return np.zeros(n, dtype=int)


# =============================================================================
# Sampling Algorithm (Section 5.2 of paper)
# =============================================================================

def random_stabilizer_element(w: Tuple[int, ...], q: int) -> np.ndarray:
    """
    Sample uniformly from U_w ⊂ U (upper triangular unipotent stabilizer).
    
    By Proposition in paper:
    U_w = {(u_ij) ∈ U : u_ij = 0 for (i,j) ∈ Inv(w)}
    
    So we set u_ij = random for non-inversions above diagonal, 0 for inversions.
    """
    n = len(w)
    
    # Convert to 0-indexed if needed
    perm = list(w)
    if min(perm) == 1:
        perm = [x - 1 for x in perm]
    
    # Compute inverse permutation
    inv_perm = [0] * n
    for i, val in enumerate(perm):
        inv_perm[val] = i
    
    # Build upper triangular unipotent matrix
    M = np.eye(n, dtype=int)
    
    for i in range(n):
        for j in range(i + 1, n):
            # (i, j) is NOT an inversion of w iff w^{-1}(i) < w^{-1}(j)
            # i.e., inv_perm[i] < inv_perm[j]
            if inv_perm[i] < inv_perm[j]:
                M[i, j] = random.randint(0, q - 1)
            # else: M[i,j] = 0 (already set by eye)
    
    return M


def sample_first_vector(A: np.ndarray, q: int) -> np.ndarray:
    """
    Sample first vector of flag from Springer fiber.
    
    Algorithm:
    1. Compute Jordan form A = PJP^{-1}
    2. Get partition λ from Jordan block sizes
    3. Sample λ' ∈ R(λ) with prob ∝ |E^{λ'}(λ)| · Q_{(1^n)}^{λ'}(q)
    4. Sample v uniformly from E^{λ'}(λ)
    5. Return P·v (transform back to original coordinates)
    """
    n = A.shape[0]
    
    # Get Jordan form
    J, P, partition = jordan_form(A, q)
    
    # Compute sampling weights
    weights = compute_sampling_weights(partition, q)
    
    # Sample smaller partition
    partitions_list = list(weights.keys())
    probs = [weights[p] for p in partitions_list]
    chosen_partition = random.choices(partitions_list, weights=probs, k=1)[0]
    
    # Find which block size was decremented
    from collections import Counter
    c1 = Counter(partition)
    c2 = Counter(chosen_partition)
    
    dec_size = None
    for size in c1:
        if c1[size] > c2.get(size, 0):
            dec_size = size
            break
    
    # Compute I(λ): starting indices of Jordan blocks
    block_starts = []
    offset = 0
    for size in partition:
        block_starts.append(offset)
        offset += size
    
    # Partition block starts by size
    eq_indices = [s for s, size in zip(block_starts, partition) if size == dec_size]
    gt_indices = [s for s, size in zip(block_starts, partition) if size > dec_size]
    
    # Sample from E^{λ'}(λ):
    # - Coordinates at indices corresponding to blocks of size = dec_size: random nonzero
    # - Coordinates at indices corresponding to blocks of size > dec_size: random
    # - All other coordinates: 0
    
    v = np.zeros(n, dtype=int)
    
    # For equal-sized blocks: sample nonzero vector
    if eq_indices:
        # Sample random nonzero vector in GF(q)^{|eq_indices|}
        while True:
            subv = [random.randint(0, q - 1) for _ in eq_indices]
            if any(x != 0 for x in subv):
                break
        for idx, val in zip(eq_indices, subv):
            v[idx] = val
    
    # For larger blocks: sample random vector
    for idx in gt_indices:
        v[idx] = random.randint(0, q - 1)
    
    # Transform back: v_original = P @ v_jordan
    result = mat_vec_mul(P, v, q)
    return result


def quotient_representation(A: np.ndarray, v: np.ndarray, q: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Compute the quotient representation of A on V/span(v).
    
    Returns:
        A_quot: The (n-1) × (n-1) matrix representing A on quotient
        basis: Basis used for quotient (first element is v)
    """
    n = A.shape[0]
    
    # Build basis starting with v, then extend
    basis = [v.copy()]
    
    # Add standard basis vectors that extend to full basis
    for i in range(n):
        e_i = np.zeros(n, dtype=int)
        e_i[i] = 1
        
        if not in_span(e_i, basis, q):
            basis.append(e_i)
        
        if len(basis) == n:
            break
    
    # Change of basis matrix P: columns are basis vectors
    P = np.column_stack(basis)
    P = mat_mod(P, q)
    P_inv = mat_inv(P, q)
    
    # A' = P^{-1} A P
    A_prime = mat_mul(mat_mul(P_inv, A, q), P, q)
    
    # Quotient is the lower-right (n-1) × (n-1) block
    A_quot = A_prime[1:, 1:]
    
    return mat_mod(A_quot, q), basis


def lift_vector(v_quot: np.ndarray, basis: List[np.ndarray], q: int) -> np.ndarray:
    """
    Lift a vector from quotient space back to original space.
    v_quot is in coordinates relative to basis[1:], lift to original coordinates.
    """
    result = np.zeros(len(basis[0]), dtype=int)
    for coeff, b in zip(v_quot, basis[1:]):
        result = mat_mod(result + int(coeff) * b, q)
    return result


def springer_sample_flag(A: np.ndarray, q: int) -> List[np.ndarray]:
    """
    Sample a complete flag from the Springer fiber of unipotent A.
    
    Returns list of vectors [v_1, v_2, ..., v_n] where
    V_k = span(v_1, ..., v_k) is the k-th subspace of the flag.
    """
    n = A.shape[0]
    
    if n == 1:
        # Base case: only one vector (any nonzero)
        return [np.array([1], dtype=int)]
    
    # Sample first vector
    v = sample_first_vector(A, q)
    flag = [v]
    
    # Recurse on quotient
    A_quot, basis = quotient_representation(A, v, q)
    sub_flag = springer_sample_flag(A_quot, q)
    
    # Lift vectors back
    for v_quot in sub_flag:
        lifted = lift_vector(v_quot, basis, q)
        flag.append(lifted)
    
    return flag


def flag_to_permutation(flag: List[np.ndarray], q: int) -> Tuple[int, ...]:
    """
    Convert a flag to a permutation via Bruhat decomposition.
    
    Apply Gaussian elimination to find which standard basis vector
    each flag vector corresponds to (finding pivot columns from right).
    """
    n = len(flag)
    
    # Build matrix with flag vectors as rows
    A = np.array([list(v) for v in flag], dtype=int)
    A = mat_mod(A, q)
    
    pivots = []
    
    for i in range(n):
        # Eliminate using previous pivots
        for r, pc in enumerate(pivots):
            if A[i, pc] % q != 0:
                factor = A[i, pc] * mod_inv(int(A[r, pc]), q) % q
                A[i] = mat_mod(A[i] - factor * A[r], q)
        
        # Find rightmost nonzero entry (pivot column)
        pivot_col = None
        for c in range(n - 1, -1, -1):
            if A[i, c] % q != 0:
                pivot_col = c
                break
        
        if pivot_col is None:
            raise ValueError("Dependent row in flag")
        
        pivots.append(pivot_col)
    
    # Convert to 1-indexed permutation
    return tuple(c + 1 for c in pivots)


def next_step(w: Tuple[int, ...], q: int) -> Tuple[int, ...]:
    """
    One step of the Burnside process Markov chain.
    
    Given permutation w:
    1. Sample uniformly from stabilizer U_w
    2. Sample flag from Springer fiber
    3. Return permutation corresponding to new flag
    """
    n = len(w)
    
    # Sample from stabilizer
    u = random_stabilizer_element(w, q)
    
    # Sample flag from Springer fiber of u
    flag = springer_sample_flag(u, q)
    
    # Convert to permutation
    return flag_to_permutation(flag, q)


# =============================================================================
# Main / Testing
# =============================================================================

def run_chain(n: int, q: int, steps: int = 1000, start: Optional[Tuple[int, ...]] = None):
    """Run the Markov chain for given number of steps."""
    if start is None:
        start = tuple(range(1, n + 1))  # Identity permutation
    
    w = start
    history = [w]
    
    for _ in range(steps):
        w = next_step(w, q)
        history.append(w)
    
    return history


if __name__ == "__main__":
    # Test with small parameters
    q, n = Q_VAL, N_VAL
    print(f"Testing Springer sampler with q={q}, n={n}")
    
    # Test Green polynomial
    print("\nGreen polynomial values Q_{(1^n)}^λ(q):")
    for partition in partitions(n):
        val = green_polynomial_value(partition, q)
        print(f"  λ={partition}: {val}")
    
    # Test one step
    print("\nTesting one Markov chain step:")
    w = tuple(range(1, n + 1))
    print(f"  Start: {w}")
    w_new = next_step(w, q)
    print(f"  After one step: {w_new}")
    
    # Run a few steps
    print("\nRunning 10 steps of chain:")
    history = run_chain(n, q, steps=10)
    for i, perm in enumerate(history):
        print(f"  Step {i}: {perm}")
