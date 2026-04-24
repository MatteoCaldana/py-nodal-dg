"""
Implicit QR Algorithm with Wilkinson Shift — variable-precision implementation.

Works for any real floating-point dtype supported by NumPy:
  np.float16, np.float32, np.float64, np.float128 (where available), etc.

The algorithm finds all eigenvalues of a real symmetric matrix via:
  1. Householder reduction to symmetric tridiagonal form
  2. Implicit QR iteration with Wilkinson shift (symmetric / Francis shift)
  3. Deflation when off-diagonal elements become negligible

All arithmetic is performed in the dtype of the input array; no scalar
Python floats or hard-coded ``float`` / ``float64`` casts are used.

References:
    Golub & Van Loan, "Matrix Computations", 4th ed., §8.3
    Demmel, "Applied Numerical Linear Algebra", §4.4
"""

import numpy as np
from numpy import ndarray
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Internal dtype helpers
# ---------------------------------------------------------------------------


def _zero(dtype: np.dtype) -> np.generic:
    return dtype.type(0)


def _one(dtype: np.dtype) -> np.generic:
    return dtype.type(1)


def _two(dtype: np.dtype) -> np.generic:
    return dtype.type(2)


def _eps(dtype: np.dtype) -> np.generic:
    return np.finfo(dtype).eps


# ---------------------------------------------------------------------------
# Householder tridiagonalisation (Modified to accumulate V)
# ---------------------------------------------------------------------------


def _householder_tridiagonalise(A: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Returns (diag, off, V) where V is the orthogonal matrix s.t. V^T A V = T.
    """
    dtype = A.dtype
    n = A.shape[0]
    A = A.copy()
    V = np.eye(n, dtype=dtype)

    zero = _zero(dtype)
    two = _two(dtype)

    for k in range(n - 2):
        x = A[k + 1 :, k]
        sigma = np.linalg.norm(x).astype(dtype)
        if sigma == zero:
            continue

        v = x.copy()
        v[0] = v[0] + np.copysign(sigma, x[0])
        v = v / np.linalg.norm(v).astype(dtype)

        # Update A submatrix
        sub = A[k + 1 :, k + 1 :]
        p = np.dot(sub, v)
        vtpv = np.dot(v, p)
        w = p - vtpv * v
        sub -= two * (np.outer(v, w) + np.outer(w, v))
        A[k + 1 :, k + 1 :] = sub

        # Accumulate transformations in V
        # V_new = V * H_k -> Only columns k+1 and onwards are affected
        v_ext = V[:, k + 1 :]
        V[:, k + 1 :] = v_ext - two * np.outer(np.dot(v_ext, v), v)

        # Clean up tridiagonal part
        A[k + 1, k] = np.negative(np.copysign(sigma, x[0]))
        A[k, k + 1] = A[k + 1, k]
        A[k + 2 :, k] = zero
        A[k, k + 2 :] = zero

    return np.diag(A).copy(), np.diag(A, 1).copy(), V


# ---------------------------------------------------------------------------
# Wilkinson shift
# ---------------------------------------------------------------------------


def _wilkinson_shift(
    a_m1: np.generic,
    a_m: np.generic,
    b_m1: np.generic,
) -> np.generic:
    dtype = a_m.dtype
    two = _two(dtype)
    one = _one(dtype)
    zero = _zero(dtype)

    delta = (a_m1 - a_m) / two
    denom = np.abs(delta) + np.hypot(delta, b_m1)
    sign = np.where(delta != zero, np.copysign(one, delta), one)
    mu = np.where(denom == zero, a_m, a_m - sign * b_m1 * b_m1 / denom)
    return dtype.type(mu)


# ---------------------------------------------------------------------------
# Implicit symmetric QR step (Modified to update V)
# ---------------------------------------------------------------------------


def _implicit_qr_step(
    diag: ndarray,
    off: ndarray,
    V: ndarray,
    p: int,
    q: int,
) -> None:
    """One implicit symmetric QR step updating eigenvalues and eigenvectors."""
    dtype = diag.dtype
    zero = _zero(dtype)
    two = _two(dtype)

    mu = _wilkinson_shift(diag[q - 1], diag[q], off[q - 1])

    x = diag[p] - mu
    z = off[p]

    for k in range(p, q):
        r = np.hypot(x, z).astype(dtype)
        if r == zero:
            break
        c = x / r
        s = -z / r

        if k > p:
            off[k - 1] = r

        α = diag[k]
        β = diag[k + 1]
        γ = off[k]
        cs = c * s
        cc = c * c
        ss = s * s

        diag[k] = cc * α + ss * β - two * cs * γ
        diag[k + 1] = ss * α + cc * β + two * cs * γ
        off[k] = cs * (α - β) + (cc - ss) * γ

        # Update Eigenvectors: V = V * G_k
        # This rotates the k and k+1 columns of V
        v_k = V[:, k].copy()
        v_kp1 = V[:, k + 1].copy()
        V[:, k] = c * v_k - s * v_kp1
        V[:, k + 1] = s * v_k + c * v_kp1

        if k < q - 1:
            x = off[k]
            z = np.negative(s) * off[k + 1]
            off[k + 1] = c * off[k + 1]


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def eigh_qr(
    A: ndarray,
    tol: Optional[float] = None,
    max_iter: int = 30,
) -> Tuple[ndarray, ndarray]:
    """
    Eigenvalues and Eigenvectors of a real symmetric matrix.

    Returns
    -------
    eigenvalues  : (n,) array
    eigenvectors : (n, n) array where each column is an eigenvector
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square 2-D, got shape {A.shape}")

    dtype = np.result_type(A.dtype, np.float32)
    if not np.issubdtype(dtype, np.floating):
        raise ValueError(f"A must have a real floating dtype, got {A.dtype}")
    A = A.astype(dtype, copy=False)

    n = A.shape[0]
    if n == 1:
        return np.array([A[0, 0]], dtype=dtype), np.eye(1, dtype=dtype)

    # -- Tridiagonalise and get initial V ------------------------------------
    diag, off, V = _householder_tridiagonalise(A)

    # -- Deflation tolerance -------------------------------------------------
    if tol is None:
        eps = _eps(dtype)
        norm = np.max(np.abs(diag)) + (np.max(np.abs(off)) if n > 1 else _zero(dtype))
        tol = dtype.type(n) * dtype.type(eps) * norm
    else:
        tol = dtype.type(tol)

    # -- Main iteration ------------------------------------------------------
    q = n - 1

    while q > 0:
        # Peel off converged eigenvalues
        deflated = True
        while deflated and q > 0:
            deflated = False
            if np.abs(off[q - 1]) <= tol * (np.abs(diag[q - 1]) + np.abs(diag[q])):
                off[q - 1] = _zero(dtype)
                q -= 1
                deflated = True

        if q == 0:
            break

        # Find top of unreduced block
        p = q - 1
        while p > 0:
            if np.abs(off[p - 1]) <= tol * (np.abs(diag[p - 1]) + np.abs(diag[p])):
                break
            p -= 1

        # QR sweeps
        for _ in range(max_iter):
            _implicit_qr_step(diag, off, V, p, q)

            if np.abs(off[q - 1]) <= tol * (np.abs(diag[q - 1]) + np.abs(diag[q])):
                off[q - 1] = _zero(dtype)
                q -= 1
                break

            # Interior split
            for i in range(p, q - 1):
                if np.abs(off[i]) <= tol * (np.abs(diag[i]) + np.abs(diag[i + 1])):
                    off[i] = _zero(dtype)
                    break
        else:
            raise RuntimeError(f"Convergence failed for block [{p}:{q}].")

    return diag, V


###############################################################################


def lu_full_pivoting(A):
    dtype = A.dtype
    n = A.shape[0]
    P = np.eye(n, dtype=dtype)
    Q = np.eye(n, dtype=dtype)
    L = np.eye(n, dtype=dtype)
    U = A.copy()

    for i in range(n):
        # --- 1. Full Pivoting: Find max in the submatrix U[i:, i:] ---
        submatrix = np.abs(U[i:, i:])
        # Get the flat index of the maximum value
        idx = np.argmax(submatrix)
        # Convert flat index to 2D coordinates relative to the submatrix
        row_idx, col_idx = np.unravel_index(idx, submatrix.shape)

        # Adjust indices to the full matrix scale
        row_idx += i
        col_idx += i

        # --- 2. Swap Rows (P and U) ---
        if row_idx != i:
            U[[i, row_idx], i:] = U[[row_idx, i], i:]
            P[[i, row_idx], :] = P[[row_idx, i], :]
            L[[i, row_idx], :i] = L[[row_idx, i], :i]

        # --- 3. Swap Columns (Q and U) ---
        if col_idx != i:
            U[:, [i, col_idx]] = U[:, [col_idx, i]]
            Q[:, [i, col_idx]] = Q[:, [col_idx, i]]
            # Note: We don't swap columns in L because L tracks row multipliers

        # --- 4. Gaussian Elimination ---
        for j in range(i + 1, n):
            if U[i, i] != 0:
                factor = U[j, i] / U[i, i]
                L[j, i] = factor
                U[j, i:] -= factor * U[i, i:]

    return P, L, U, Q


def solve_lu(A, B):
    """
    Solves Ax = B using LU decomposition with full pivoting (PAQ = LU).
    B can be a vector (n,) or a matrix (n, k).
    """
    assert A.dtype == B.dtype
    dtype = A.dtype

    P, L, U, Q = lu_full_pivoting(A)
    n = A.shape[0]

    # 1. Permute B according to row swaps: B_prime = P @ B
    B_prime = P @ B

    # 2. Forward Substitution: LY = B_prime
    # L is lower triangular with 1s on diagonal
    Y = np.zeros_like(B_prime, dtype=dtype)
    for i in range(n):
        Y[i] = B_prime[i] - L[i, :i] @ Y[:i]

    # 3. Backward Substitution: UZ = Y
    # U is upper triangular
    Z = np.zeros_like(Y, dtype=dtype)
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0:
            raise ValueError("Matrix is singular.")
        Z[i] = (Y[i] - U[i, i + 1 :] @ Z[i + 1 :]) / U[i, i]

    # 4. Final Permutation: x = Q @ Z
    # Because PAQ = LU => A = P.T @ L @ U @ Q.T
    # So x = Q @ Z
    X = Q @ Z

    return X
