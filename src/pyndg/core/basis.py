import pyndg.backend as bkd
from pyndg.core.la import eigh_qr, solve_lu

import numpy as np


def gamma(n: int) -> int:
    n -= 1
    r = 1
    while n > 0:
        r *= n
        n -= 1
    return r


def pnleg(x: np.ndarray, n: int) -> np.ndarray:
    assert isinstance(x, np.ndarray) and x.dtype == bkd.np_prec
    assert isinstance(n, int)

    if n == 0:
        return np.ones_like(x, dtype=bkd.np_prec)
    else:
        p1 = np.ones_like(x, dtype=bkd.np_prec)
        p2, p3 = x, x
        for k in range(1, n):
            p3 = ((2 * k + 1) * x * p2 - k * p1) / (k + 1)
            p1, p2 = p2, p3
        return p3


# Golub-Welsch algorithm to find the nodes and weights for Jacobi-Gauss quadrature
def jacobi_gq(alpha: int, beta: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    assert isinstance(alpha, int)
    assert isinstance(beta, int)
    assert isinstance(N, int)

    a = np.array([alpha], dtype=bkd.np_prec)
    b = np.array([beta], dtype=bkd.np_prec)
    two = np.array([2], dtype=bkd.np_prec)
    if N == 0:
        return (b - a) / (a + b + 2), two

    h1 = 2 * np.arange(0, N + 1, dtype=bkd.np_prec) + a + b

    with np.errstate(divide="ignore", invalid="ignore"):
        diag = -(a * a - b * b) / (h1 + 2) / h1
    ii = np.arange(1, N + 1, dtype=bkd.np_prec)
    subdiag = (2.0 / (h1[:N] + 2)) * np.sqrt(
        ii * (ii + a + b) * (ii + a) * (ii + b) / (h1[:N] + 1) / (h1[:N] + 3)
    )
    J = np.diag(diag) + np.diag(subdiag, 1) + np.diag(subdiag, -1)
    if alpha + beta == 0:
        J[0, 0] = 0.0
    x, V = eigh_qr(J)
    w = (
        (V[0, :] ** 2 * (2 ** (alpha + beta + 1)) * gamma(alpha + 1) * gamma(beta + 1))
        / gamma(alpha + beta + 1)
    ) / (alpha + beta + 1)
    idxs = np.argsort(x)
    return x[idxs], w[idxs]


def jacobi_gl(alpha: int, beta: int, N: int) -> np.ndarray:
    assert isinstance(alpha, int)
    assert isinstance(beta, int)
    assert isinstance(N, int)

    one = np.array([1.0], dtype=bkd.np_prec)
    if N == 1:
        return np.array([-1.0, 1.0], dtype=bkd.np_prec)

    xint, _ = jacobi_gq(alpha + 1, beta + 1, N - 2)
    return np.concatenate([-one, xint, one])


def vandermonde_1d(N: int, r: np.ndarray) -> np.ndarray:
    assert isinstance(N, int)
    assert isinstance(r, np.ndarray) and r.dtype == bkd.np_prec
    assert r.ndim == 1

    V1D = np.empty((r.size, N + 1), dtype=bkd.np_prec)
    for j in range(0, N + 1):
        V1D[:, j] = jacobi_p(r, 0, 0, j)
    return V1D


def jacobi_p(x: np.ndarray, alpha: int, beta: int, N: int) -> np.ndarray:
    assert isinstance(alpha, int)
    assert isinstance(beta, int)
    assert isinstance(N, int)
    assert x.ndim == 1, "x must be a 1D array"

    n = x.shape[0]
    a = np.array([alpha], dtype=bkd.np_prec)
    b = np.array([beta], dtype=bkd.np_prec)

    # Initialize the results table (N+1 polynomials, n points)
    PL = np.empty((N + 1, n), dtype=bkd.np_prec)

    # Initial normalization factor (gamma0)
    # Using the same logic as the weight normalization in jacobi_gq
    gamma0_num = (2 ** (alpha + beta + 1)) * gamma(alpha + 1) * gamma(beta + 1)
    gamma0_den = gamma(alpha + beta + 1) * (alpha + beta + 1)
    gamma0 = bkd.np_prec(gamma0_num) / bkd.np_prec(gamma0_den)

    PL[0, :] = 1.0 / np.sqrt(gamma0)

    if N == 0:
        return PL[0, :]

    # First order polynomial
    gamma1 = (a + 1) * (b + 1) / (a + b + 3) * gamma0
    PL[1, :] = ((a + b + 2) * x / 2 + (a - b) / 2) / np.sqrt(gamma1)

    if N == 1:
        return PL[1, :]

    # Pre-compute initial 'a' coefficient for recurrence
    aold = (2 / (2 + a + b)) * np.sqrt((a + 1) * (b + 1) / (a + b + 3))

    # Recurrence relation for higher orders
    for i in range(N - 1):
        # Index shift: original code used i+1 in h1 to represent the current step
        h1 = 2 * (i + 1) + a + b

        anew = (2 / (h1 + 2)) * np.sqrt(
            (i + 2) * (i + 2 + a + b) * (i + 2 + a) * (i + 2 + b) / (h1 + 1) / (h1 + 3)
        )

        bnew = -(a * a - b * b) / (h1 * (h1 + 2))

        # Standard three-term recurrence
        PL[i + 2, :] = (1.0 / anew) * (-aold * PL[i, :] + (x - bnew) * PL[i + 1, :])
        aold = anew

    return PL[N, :]


def dmatrix_1d(N: int, r: np.ndarray, V: np.ndarray):
    assert isinstance(r, np.ndarray) and r.dtype == bkd.np_prec
    assert isinstance(V, np.ndarray) and V.dtype == bkd.np_prec
    assert isinstance(N, int)

    Vr = grad_vandermonde_1d(N, r)
    return solve_lu(V.T, Vr.T).T


def grad_vandermonde_1d(N: int, r: np.array):
    assert isinstance(N, int)
    assert isinstance(r, np.ndarray) and r.dtype == bkd.np_prec
    assert r.ndim == 1

    DVr = np.empty((r.size, N + 1), dtype=bkd.np_prec)
    for i in range(N + 1):
        DVr[:, i] = grad_jacobi_p(r, 0, 0, i)
    return DVr


def grad_jacobi_p(r: np.ndarray, alpha: int, beta: int, N: int):
    assert isinstance(alpha, int)
    assert isinstance(beta, int)
    assert isinstance(N, int)
    assert isinstance(r, np.ndarray) and r.dtype == bkd.np_prec

    if N == 0:
        return np.zeros_like(r, dtype=bkd.np_prec)

    coef = np.sqrt(N * (N + alpha + beta + 1), dtype=bkd.np_prec)
    return coef * jacobi_p(r, alpha + 1, beta + 1, N - 1)


##############################################################################
### 2D
##############################################################################

_ALPHA_OPT_2D = [
    None,
    0.0000,
    0.0000,
    1.4152,
    0.1001,
    0.2751,
    0.9800,
    1.0999,
    1.2832,
    1.3648,
    1.4773,
    1.4959,
    1.5743,
    1.5770,
    1.6223,
    1.6258,
]


def warp_2d(N, rout):
    LGLr = jacobi_gl(0, 0, N)
    req = np.linspace(-1, 1, N + 1, dtype=bkd.np_prec)
    Veq = vandermonde_1d(N, req)

    Nr = rout.size
    Pmat = np.empty((N + 1, Nr), dtype=bkd.np_prec)
    for i in range(N + 1):
        Pmat[i, :] = jacobi_p(rout, 0, 0, i)
    Lmat = solve_lu(Veq.T, Pmat)

    warp = np.matmul(Lmat.T, LGLr - req)

    zerof = np.abs(rout) < 1.0 - np.finfo(bkd.np_prec).eps * 10
    if np.any((np.abs(rout) >= 1.0 - 1e-5) & (np.abs(rout) != 1.0)):
        print("WARNING: rout values are close to 1 but not exactly 1")
        print("rout values:", rout)
    sf = 1.0 - (zerof * rout) ** 2
    return warp / sf + warp * (zerof - 1)


# here do not need extra precision, alpha is already approximated
def nodes_2d(N):
    alpha = _ALPHA_OPT_2D[N] if N < 16 else 5 / 3
    Np = (N + 1) * (N + 2) // 2
    L1, L3 = np.empty((Np,)), np.empty((Np,))
    idx = 0
    for n in range(N + 1):
        for m in range(N + 1 - n):
            L1[idx], L3[idx] = n / N, m / N
            idx += 1
    L2 = 1.0 - L1 - L3
    x = -L2 + L3
    y = (-L2 - L3 + 2 * L1) / np.sqrt(3.0)

    b1, b2, b3 = 4 * L2 * L3, 4 * L1 * L3, 4 * L1 * L2
    w1, w2, w3 = warp_2d(N, L3 - L2), warp_2d(N, L1 - L3), warp_2d(N, L2 - L1)

    wb1 = b1 * w1 * (1 + (alpha * L1) ** 2)
    wb2 = b2 * w2 * (1 + (alpha * L2) ** 2)
    wb3 = b3 * w3 * (1 + (alpha * L3) ** 2)

    x = x + 1 * wb1 + np.cos(2 * np.pi / 3) * wb2 + np.cos(4 * np.pi / 3) * wb3
    y = y + 0 * wb1 + np.sin(2 * np.pi / 3) * wb2 + np.sin(4 * np.pi / 3) * wb3

    x = x.astype(bkd.np_prec)
    y = y.astype(bkd.np_prec)

    # clipping
    sqrt3 = np.sqrt(3.0, dtype=bkd.np_prec)
    inv_sqrt3 = -1.0 / np.sqrt(3.0, dtype=bkd.np_prec)
    y = np.where(np.abs(y - inv_sqrt3) < 1e-10, inv_sqrt3, y)
    y = np.where(np.abs(y - 2 / sqrt3) < 1e-10, 2 / sqrt3, y)

    two_third = bkd.np_prec(2) / bkd.np_prec(3)
    x = np.where(np.abs(y - sqrt3 * (x + two_third)) < 1e-10, y / sqrt3 - two_third, x)
    x = np.where(np.abs(y + sqrt3 * (x - two_third)) < 1e-10, -y / sqrt3 + two_third, x)
    x = np.where(np.abs(x) < 1e-10, 0.0, x)

    return x, y


def xy_to_rs(x, y):
    assert isinstance(x, np.ndarray) and x.dtype == bkd.np_prec
    assert isinstance(y, np.ndarray) and y.dtype == bkd.np_prec
    assert x.ndim == 1 and y.ndim == 1
    assert x.shape == y.shape

    sqrt3 = np.sqrt(3.0, dtype=bkd.np_prec)
    L1 = (sqrt3 * y + 1.0) / 3.0
    L2 = (-3.0 * x - sqrt3 * y + 2.0) / 6.0
    L3 = (3.0 * x - sqrt3 * y + 2.0) / 6.0
    r = -L2 + L3 - L1
    s = -L2 - L3 + L1

    # clipping
    # start with a "soft" clip, move both, but division is not reliable
    idx = np.where(np.abs(r + s - 1.0) < 1e-10)[0]
    r[idx] = r[idx] / (r[idx] + s[idx])
    s[idx] = s[idx] / (r[idx] + s[idx])
    # hard clip r
    r = np.where(np.abs(r + s - 1.0) < 1e-10, 1.0 - s, r)
    # clip the orthogonal edges, easy
    r = np.where(np.abs(r + 1.0) < 1e-10, -1.0, r)
    s = np.where(np.abs(s + 1.0) < 1e-10, -1.0, s)
    return r, s


def rs_to_ab(r, s):
    assert isinstance(r, np.ndarray) and r.dtype == bkd.np_prec
    assert isinstance(s, np.ndarray) and s.dtype == bkd.np_prec
    assert r.ndim == 1 and s.ndim == 1
    assert r.shape == s.shape

    a = np.empty_like(r, dtype=bkd.np_prec)
    for n in range(r.size):
        den = np.abs(s[n] - 1.0)
        if den > 0.0:
            if den < 1e-14:
                print("WARNING: denominator close to zero in rs_to_ab!")
            a[n] = 2 * (1 + r[n]) / (1 - s[n]) - 1
        else:
            a[n] = -1
    return a, s


def vandermonde_2d(N, r, s):
    assert isinstance(r, np.ndarray) and r.dtype == bkd.np_prec
    assert isinstance(s, np.ndarray) and s.dtype == bkd.np_prec
    assert isinstance(N, int)

    V2D = np.empty((r.size, (N + 1) * (N + 2) // 2), dtype=bkd.np_prec)
    a, b = rs_to_ab(r, s)
    idx = 0
    for i in range(N + 1):
        for j in range(N - i + 1):
            V2D[:, idx] = simplex_2dp(a, b, i, j)
            idx += 1
    return V2D


def simplex_2dp(a: np.ndarray, b: np.ndarray, i: int, j: int):
    assert isinstance(a, np.ndarray) and a.dtype == bkd.np_prec
    assert isinstance(b, np.ndarray) and b.dtype == bkd.np_prec
    assert isinstance(i, int)
    assert isinstance(j, int)

    h1 = jacobi_p(a, 0, 0, i)
    h2 = jacobi_p(b, 2 * i + 1, 0, j)
    return np.sqrt(2.0, dtype=bkd.np_prec) * h1 * h2 * (1.0 - b) ** i


def dmatrices_2d(N: int, r: np.ndarray, s: np.ndarray, V: np.ndarray):
    assert isinstance(r, np.ndarray) and r.dtype == bkd.np_prec
    assert isinstance(s, np.ndarray) and s.dtype == bkd.np_prec
    assert isinstance(V, np.ndarray) and V.dtype == bkd.np_prec
    assert isinstance(N, int)

    Vr, Vs = grad_vandermonde_2d(N, r, s)
    return solve_lu(V.T, Vr.T).T, solve_lu(V.T, Vs.T).T


def grad_vandermonde_2d(N, r, s):
    assert isinstance(r, np.ndarray) and r.dtype == bkd.np_prec
    assert isinstance(s, np.ndarray) and s.dtype == bkd.np_prec
    assert isinstance(N, int)

    Np = (N + 1) * (N + 2) // 2
    V2Dr = np.empty((r.size, Np), dtype=bkd.np_prec)
    V2Ds = np.empty((r.size, Np), dtype=bkd.np_prec)
    a, b = rs_to_ab(r, s)

    idx = 0
    for i in range(N + 1):
        for j in range(N - i + 1):
            V2Dr[:, idx], V2Ds[:, idx] = grad_simplex_2dp(a, b, i, j)
            idx += 1
    return V2Dr, V2Ds


def grad_simplex_2dp(a, b, i, j):
    assert isinstance(a, np.ndarray) and a.dtype == bkd.np_prec
    assert isinstance(b, np.ndarray) and b.dtype == bkd.np_prec
    assert isinstance(i, int)
    assert isinstance(j, int)

    fa, dfa = jacobi_p(a, 0, 0, i), grad_jacobi_p(a, 0, 0, i)
    gb, dgb = jacobi_p(b, 2 * i + 1, 0, j), grad_jacobi_p(b, 2 * i + 1, 0, j)
    dmodedr = dfa * gb
    dmodeds = dfa * (gb * (0.5 * (1 + a)))
    if i > 0:
        fact = (0.5 * (1 - b)) ** (i - 1)
        dmodeds *= fact
        dmodedr *= fact
    else:
        fact = np.zeros_like(b, dtype=bkd.np_prec)
    dmodeds += fa * ((dgb * ((0.5 * (1 - b)) ** i) - 0.5 * i * gb * fact))
    return 2 ** bkd.np_prec(i + 0.5) * dmodedr, 2 ** bkd.np_prec(i + 0.5) * dmodeds


##############################################################################
### 3D
##############################################################################


_ALPHA_OPT_3D = [
    None,
    0,
    0,
    0,
    0.1002,
    1.1332,
    1.5608,
    1.3413,
    1.2577,
    1.1603,
    1.10153,
    0.6080,
    0.4523,
    0.8856,
    0.8717,
    0.9655,
]


def _equi_nodes_3d(N):
    Np = (N + 1) * (N + 2) * (N + 3) // 6
    nodes = np.empty((Np, 3), dtype=bkd.np_prec)
    idx = 0
    for n in range(N + 1):
        for m in range(N + 1 - n):
            for q in range(N + 1 - n - m):
                nodes[idx, 0] = -1.0 + q * 2.0 / N
                nodes[idx, 1] = -1.0 + m * 2.0 / N
                nodes[idx, 2] = -1.0 + n * 2.0 / N
                idx += 1
    return nodes[:, 0], nodes[:, 1], nodes[:, 2]


_SQRT3 = np.sqrt(3.0, dtype=bkd.np_prec)
_SQRT6 = np.sqrt(6.0, dtype=bkd.np_prec)

# vertices of the reference tetrahedron
_V1 = np.array([-1, -1 / _SQRT3, -1 / _SQRT6], dtype=bkd.np_prec)
_V2 = np.array([1, -1 / _SQRT3, -1 / _SQRT6], dtype=bkd.np_prec)
_V3 = np.array([0, 2 / _SQRT3, -1 / _SQRT6], dtype=bkd.np_prec)
_V4 = np.array([0, 0, 3 / _SQRT6], dtype=bkd.np_prec)



def warp_shift_face_3d(p, pval, L1, L2, L3):
    # compute blending function at each node for each edge
    blend1 = L2 * L3
    blend2 = L1 * L3
    blend3 = L1 * L2

    # amount of warp for each node, for each edge
    warpfactor1 = 4 * warp_2d(p, L3 - L2)
    warpfactor2 = 4 * warp_2d(p, L1 - L3)
    warpfactor3 = 4 * warp_2d(p, L2 - L1)

    # combine blend & warp
    warp1 = blend1 * warpfactor1 * (1 + (pval * L1) ** 2)
    warp2 = blend2 * warpfactor2 * (1 + (pval * L2) ** 2)
    warp3 = blend3 * warpfactor3 * (1 + (pval * L3) ** 2)

    # evaluate shift in equilateral triangle
    dx = 1 * warp1 + np.cos(2 * np.pi / 3) * warp2 + np.cos(4 * np.pi / 3) * warp3
    dy = 0 * warp1 + np.sin(2 * np.pi / 3) * warp2 + np.sin(4 * np.pi / 3) * warp3

    return dx, dy


# here do not need extra precision, alpha is already approximated
def nodes_3d(N):
    assert isinstance(N, int)

    alpha = _ALPHA_OPT_3D[N] if N < 16 else 1.0

    r, s, t = _equi_nodes_3d(N)

    L1 = (1 + t) / 2
    L2 = (1 + s) / 2
    L3 = -(1 + r + s + t) / 2
    L4 = (1 + r) / 2

    # orthogonal axis tangents on faces 1-4
    t1 = np.empty((4, 3), dtype=bkd.np_prec)
    t2 = np.empty((4, 3), dtype=bkd.np_prec)
    t1[0] = _V2 - _V1
    t1[1] = _V2 - _V1
    t1[2] = _V3 - _V2
    t1[3] = _V3 - _V1
    t2[0] = _V3 - 0.5 * (_V1 + _V2)
    t2[1] = _V4 - 0.5 * (_V1 + _V2)
    t2[2] = _V4 - 0.5 * (_V2 + _V3)
    t2[3] = _V4 - 0.5 * (_V1 + _V3)

    for i in range(4):
        t1[i] /= np.linalg.norm(t1[i])
        t2[i] /= np.linalg.norm(t2[i])

    # Warp and blend for each face (accumulated in shift)
    XYZ = np.outer(L3, _V1) + np.outer(L4, _V2) + np.outer(L2, _V3) + np.outer(L1, _V4)
    shift = np.zeros_like(XYZ, dtype=bkd.np_prec)

    PERMUTATIONS = [[0, 1, 2, 3], [1, 0, 2, 3], [2, 0, 3, 1], [3, 0, 2, 1]]
    TOL = 1e-7
    Ls = [L1, L2, L3, L4]
    for face in range(4):
        La, Lb, Lc, Ld = [Ls[PERMUTATIONS[face][i]] for i in range(4)]
        warp1, warp2 = warp_shift_face_3d(N, alpha, Lb, Lc, Ld)
        blend = Lb * Lc * Ld
        denom = (Lb + 0.5 * La) * (Lc + 0.5 * La) * (Ld + 0.5 * La)

        idx = np.where(denom > TOL)[0]
        blend[idx] = (1 + (alpha * La[idx]) ** 2) * blend[idx] / denom[idx]

        # compute warp & blend
        shift += np.outer(blend * warp1, t1[face]) + np.outer(blend * warp2, t2[face])

        # fix face warp
        ids = np.where((La < TOL) & ((Lb > TOL) + (Lc > TOL) + (Ld > TOL) < 3))[0]
        shift[ids] = np.outer(warp1[ids], t1[face]) + np.outer(warp2[ids], t2[face])

    XYZ += shift    
    return XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]


def xyz_to_rst(x, y, z):
    assert isinstance(x, np.ndarray) and x.dtype == bkd.np_prec
    assert isinstance(y, np.ndarray) and y.dtype == bkd.np_prec
    assert isinstance(z, np.ndarray) and z.dtype == bkd.np_prec
    assert x.ndim == 1 and y.ndim == 1 and z.ndim == 1
    assert x.shape == y.shape == z.shape

    # back out right tet nodes
    RHS = np.stack([x, y, z], axis=0) - 0.5 * (_V2 + _V3 + _V4 - _V1)[:, None]
    A = 0.5 * np.stack([_V2 - _V1, _V3 - _V1, _V4 - _V1], axis=1)
    RST = solve_lu(A, RHS)
    r = RST[0, :]
    s = RST[1, :]
    t = RST[2, :]

    # clipping

    # fix the face r+s+t=-1
    rst_sum = r + s + t
    mask = np.abs(rst_sum + 1.0) < 1e-10
    correction = (rst_sum[mask] + 1.0) / 3.0
    r[mask] -= correction
    s[mask] -= correction
    t[mask] -= correction

    mask = np.abs(r + s + t + 1.0) < 1e-10
    t[mask] = -1.0 - r[mask] - s[mask]

    # fix edges
    t = np.where(np.abs(t + s) < 1e-10, -s, t)
    r = np.where(np.abs(r + s) < 1e-10, -s, r)
    r = np.where(np.abs(r + t) < 1e-10, -t, r)

    # fix the orthogonal faces
    r = np.where(np.abs(r + 1.0) < 1e-10, -1.0, r)
    s = np.where(np.abs(s + 1.0) < 1e-10, -1.0, s)
    t = np.where(np.abs(t + 1.0) < 1e-10, -1.0, t)

    return r, s, t


def rst_to_abc(r, s, t):
    assert isinstance(r, np.ndarray) and r.dtype == bkd.np_prec
    assert isinstance(s, np.ndarray) and s.dtype == bkd.np_prec
    assert isinstance(t, np.ndarray) and t.dtype == bkd.np_prec
    assert r.ndim == 1 and s.ndim == 1 and t.ndim == 1
    assert r.shape == s.shape == t.shape

    a = np.empty_like(r, dtype=bkd.np_prec)
    b = np.empty_like(s, dtype=bkd.np_prec)
    for n in range(r.size):
        den = np.abs(t[n] + s[n])
        if den > 0.0:
            if den < 1e-14:
                print("WARNING: denominator 1 close to zero in rst_to_abc!")
            a[n] = -2.0 * (1 + r[n]) / (t[n] + s[n]) - 1
        else:
            a[n] = -1

        den = np.abs(t[n] - 1.0)
        if den > 0.0:
            if den < 1e-14:
                print("WARNING: denominator 2 close to zero in rst_to_abc!")
            b[n] = 2 * (1 + s[n]) / (1 - t[n]) - 1
        else:
            b[n] = -1

    return a, b, t


def vandermonde_3d(N: int, r: np.ndarray, s: np.ndarray, t: np.ndarray):
    assert isinstance(r, np.ndarray) and r.dtype == bkd.np_prec
    assert isinstance(s, np.ndarray) and s.dtype == bkd.np_prec
    assert isinstance(t, np.ndarray) and t.dtype == bkd.np_prec
    assert isinstance(N, int)

    V3D = np.empty((r.size, (N + 1) * (N + 2) * (N + 3) // 6), dtype=bkd.np_prec)
    a, b, c = rst_to_abc(r, s, t)
    idx = 0
    for i in range(N + 1):
        for j in range(N - i + 1):
            for k in range(N - i - j + 1):
                V3D[:, idx] = simplex_3dp(a, b, c, i, j, k)
                idx += 1

    return V3D


def grad_vandermonde_3d(N, r, s, t):
    assert isinstance(r, np.ndarray) and r.dtype == bkd.np_prec
    assert isinstance(s, np.ndarray) and s.dtype == bkd.np_prec
    assert isinstance(t, np.ndarray) and t.dtype == bkd.np_prec
    assert isinstance(N, int)

    Np = (N + 1) * (N + 2) * (N + 3) // 6
    V3Dr = np.empty((r.size, Np), dtype=bkd.np_prec)
    V3Ds = np.empty((r.size, Np), dtype=bkd.np_prec)
    V3Dt = np.empty((r.size, Np), dtype=bkd.np_prec)
    a, b, c = rst_to_abc(r, s, t)

    idx = 0
    for i in range(N + 1):
        for j in range(N - i + 1):
            for k in range(N - i - j + 1):
                V3Dr[:, idx], V3Ds[:, idx], V3Dt[:, idx] = grad_simplex_3dp(
                    a, b, c, i, j, k
                )
                idx += 1
    return V3Dr, V3Ds, V3Dt


def simplex_3dp(a, b, c, i, j, k):
    assert isinstance(a, np.ndarray) and a.dtype == bkd.np_prec
    assert isinstance(b, np.ndarray) and b.dtype == bkd.np_prec
    assert isinstance(c, np.ndarray) and c.dtype == bkd.np_prec
    assert isinstance(i, int)
    assert isinstance(j, int)
    assert isinstance(k, int)

    h1 = jacobi_p(a, 0, 0, i)
    h2 = jacobi_p(b, 2 * i + 1, 0, j)
    h3 = jacobi_p(c, 2 * (i + j) + 2, 0, k)

    return (
        2
        * np.sqrt(2, dtype=bkd.np_prec)
        * h1
        * h2
        * ((1 - b) ** i)
        * h3
        * ((1 - c) ** (i + j))
    )


def grad_simplex_3dp(a, b, c, i, j, k):
    assert isinstance(a, np.ndarray) and a.dtype == bkd.np_prec
    assert isinstance(b, np.ndarray) and b.dtype == bkd.np_prec
    assert isinstance(c, np.ndarray) and c.dtype == bkd.np_prec
    assert isinstance(i, int)
    assert isinstance(j, int)
    assert isinstance(k, int)

    fa = jacobi_p(a, 0, 0, i)
    dfa = grad_jacobi_p(a, 0, 0, i)
    gb = jacobi_p(b, 2 * i + 1, 0, j)
    dgb = grad_jacobi_p(b, 2 * i + 1, 0, j)
    hc = jacobi_p(c, 2 * (i + j) + 2, 0, k)
    dhc = grad_jacobi_p(c, 2 * (i + j) + 2, 0, k)

    # r-derivative
    V3Dr = dfa * gb * hc
    if i > 0:
        V3Dr *= (0.5 * (1 - b)) ** (i - 1)
    if i + j > 0:
        V3Dr *= (0.5 * (1 - c)) ** (i + j - 1)

    # s-derivative
    V3Ds = 0.5 * (1 + a) * V3Dr
    tmp = dgb * ((0.5 * (1 - b)) ** i)
    if i > 0:
        tmp += -i * gb * ((0.5 * (1 - b)) ** (i - 1)) / 2.0
    if i + j > 0:
        tmp *= (0.5 * (1 - c)) ** (i + j - 1)
    tmp *= fa * hc
    V3Ds += tmp

    # t-derivative
    V3Dt = 0.5 * (1 + a) * V3Dr + 0.5 * (1 + b) * tmp
    tmp = dhc * ((0.5 * (1 - c)) ** (i + j))
    if i + j > 0:
        tmp = tmp - 0.5 * (i + j) * (hc * ((0.5 * (1 - c)) ** (i + j - 1)))
    tmp = fa * (gb * tmp)
    tmp = tmp * ((0.5 * (1 - b)) ** i)
    V3Dt = V3Dt + tmp

    # normalize
    V3Dr = V3Dr * (bkd.np_prec(2) ** (2 * i + j + 1.5))
    V3Ds = V3Ds * (bkd.np_prec(2) ** (2 * i + j + 1.5))
    V3Dt = V3Dt * (bkd.np_prec(2) ** (2 * i + j + 1.5))

    return V3Dr, V3Ds, V3Dt


def dmatrices_3d(N: int, r: np.ndarray, s: np.ndarray, t: np.ndarray, V: np.ndarray):
    assert isinstance(r, np.ndarray) and r.dtype == bkd.np_prec
    assert isinstance(s, np.ndarray) and s.dtype == bkd.np_prec
    assert isinstance(t, np.ndarray) and t.dtype == bkd.np_prec
    assert isinstance(V, np.ndarray) and V.dtype == bkd.np_prec
    assert isinstance(N, int)

    Vrst = grad_vandermonde_3d(N, r, s, t)
    return [solve_lu(V.T, Vv.T).T for Vv in Vrst]
