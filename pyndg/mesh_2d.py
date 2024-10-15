# -*- coding: utf-8 -*-
import numpy as np
from scipy import sparse

from .scalar_param_2d import ScalarParam2D
from .euler_param_2d import EulerParam2D
from .mesh_1d import jacobiGL, vandermonde1D, jacobiP, grad_jacobiP
from .bc_2d import BC
from . import backend as bkd
from .backend import bkd as bk

MESH_TOL = 1e-12


def read_gmsh_file(filename):
    with open(filename, "r") as fp:
        data = fp.read().split("\n")
    assert data[0] == "$MeshFormat"
    assert data[1][0] == "2", "Support only version 2"
    assert data[2] == "$EndMeshFormat"
    assert data[3] == "$Nodes"
    Nv = int(data[4])
    VXY = np.empty((Nv, 2))
    for i in range(Nv):
        line_tokens = data[i + 5].split(" ")
        VXY[i, 0] = float(line_tokens[1])
        VXY[i, 1] = float(line_tokens[2])
    assert data[Nv + 5] == "$EndNodes"
    assert data[Nv + 6] == "$Elements"
    Ne = int(data[Nv + 7])
    Nb = 0
    line_tokens = data[Nv + 8].split(" ")
    b_faces = {}
    g_to_utag = {}
    while line_tokens[1] == "1":
        Nb += 1
        gftag = int(line_tokens[4])
        uftag = int(line_tokens[3])
        if gftag in g_to_utag:
            assert g_to_utag[gftag] == uftag
        else:
            g_to_utag[gftag] = uftag
        if uftag not in b_faces:
            b_faces[uftag] = []
        b_faces[uftag].append(int(line_tokens[-2]))
        b_faces[uftag].append(int(line_tokens[-1]))
        line_tokens = data[Nv + 8 + Nb].split(" ")
    for key in b_faces:
        b_faces[key] = np.array(b_faces[key]).reshape((-1, 2))
    K = Ne - Nb
    EToV = np.empty((K, 3), dtype=np.int32)
    for i in range(K - 1):
        EToV[i, 0] = int(line_tokens[-3])
        EToV[i, 1] = int(line_tokens[-2])
        EToV[i, 2] = int(line_tokens[-1])
        line_tokens = data[Nv + 8 + Nb + i + 1].split(" ")
    EToV[-1, 0] = int(line_tokens[-3])
    EToV[-1, 1] = int(line_tokens[-2])
    EToV[-1, 2] = int(line_tokens[-1])
    assert data[Nv + 8 + Nb + K] == "$EndElements"
    PerBToB, PerBFToF = {}, {}
    if data[Nv + 8 + Nb + K + 1] == "$Periodic":
        Np = int(data[Nv + 8 + Nb + K + 2])
        idx = 0
        for i in range(Np):
            line_tokens, idx = data[Nv + 8 + Nb + K + 3 + idx].split(" "), idx + 1
            n_ent, idx = int(data[Nv + 8 + Nb + K + 3 + idx]), idx + 1
            if line_tokens[0] == "1":
                bf1 = g_to_utag[int(line_tokens[1])]
                bf2 = g_to_utag[int(line_tokens[2])]
                nface = b_faces[bf1].shape[0]
                PerBToB[bf1] = bf2
                f2f = np.zeros((nface, nface))
                bf1_vlist = b_faces[bf1]
                bf2_vlist = b_faces[bf2]
                p2p = {}
                for j in range(n_ent):
                    line_tokens, idx = (
                        data[Nv + 8 + Nb + K + 3 + idx].split(" "),
                        idx + 1,
                    )
                    p1, p2 = int(line_tokens[0]), int(line_tokens[1])
                    p2p[p1] = p2
                    ind1 = np.where((bf1_vlist[:, 0] == p1) | (bf1_vlist[:, 1] == p1))
                    ind2 = np.where((bf2_vlist[:, 0] == p2) | (bf2_vlist[:, 1] == p2))
                    for i1 in ind1[0]:
                        for i2 in ind2[0]:
                            f2f[i1, i2] = f2f[i1, i2] + 1
                f2flist = np.empty((nface, 1), dtype=np.int32)
                for f1 in range(nface):
                    f2 = np.where(f2f[f1, :] == 2)[0]
                    f2flist[f1] = f2 * (
                        (p2p[bf1_vlist[f1, 0]] == bf1_vlist[f2, 0]) * 2 - 1
                    )
                PerBFToF[bf1] = f2flist
            else:
                for j in range(n_ent):
                    idx += 1
    EToV -= 1
    ax, ay = VXY[:, 0][EToV[:, 0]], VXY[:, 1][EToV[:, 0]]
    bx, by = VXY[:, 0][EToV[:, 1]], VXY[:, 1][EToV[:, 1]]
    cx, cy = VXY[:, 0][EToV[:, 2]], VXY[:, 1][EToV[:, 2]]
    D = (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)
    i = np.where(D < 0)[0]
    if i.size:
        EToV = EToV[i, [0, 2, 1]]
    return VXY, K, Nv, EToV, b_faces, PerBToB, PerBFToF


def make_gmsh(corner_sw, corner_ne, nps):
    assert len(corner_sw) == 2
    assert len(corner_ne) == 2
    assert len(nps) == 2

    X = np.linspace(corner_sw[0], corner_ne[0], nps[0])
    Y = np.linspace(corner_sw[1], corner_ne[1], nps[1])
    XX, YY = np.meshgrid(X, Y)

    VXY = np.stack([XX.flatten(), YY.flatten()]).T
    K = 2 * (nps[0] - 1) * (nps[1] - 1)
    Nv = VXY.shape[0]

    EToV = np.zeros((K, 3), dtype=np.int32)
    for k in range(K // 2):
        ie = np.array([k % (nps[0] - 1), k // (nps[1] - 1)], dtype=np.int32)
        iv1 = ie + np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int32)
        iv2 = ie + np.array([[0, 1], [1, 0], [1, 1]], dtype=np.int32)

        for j in range(3):
            EToV[2 * k, j] = iv1[j, 0] + iv1[j, 1] * nps[0]
            EToV[2 * k + 1, j] = iv2[j, 0] + iv2[j, 1] * nps[0]

    b_faces = {}
    b_faces[100001] = np.zeros((nps[1] - 1, 2), dtype=np.int32)
    b_faces[100002] = np.zeros((nps[1] - 1, 2), dtype=np.int32)
    b_faces[100003] = np.zeros((nps[0] - 1, 2), dtype=np.int32)
    b_faces[100004] = np.zeros((nps[0] - 1, 2), dtype=np.int32)
    for i in range(nps[0] - 1):
        b_faces[100003][i, :] = i + np.array([0, 1], dtype=np.int32)
        b_faces[100004][i, :] = Nv - i - 1 + np.array([0, -1], dtype=np.int32)

    for i in range(nps[1] - 1):
        b_faces[100001][-1 - i, :] = nps[0] * (i + np.array([1, 0], dtype=np.int32))
        b_faces[100002][i, :] = nps[0] * (i + 1 + np.array([0, 1], dtype=np.int32)) - 1

    for k in b_faces:
        b_faces[k] += 1

    PerBToB = {}
    PerBToB[100002] = 100001
    PerBToB[100003] = 100004

    PerBFToF = {}
    PerBFToF[100002] = np.arange(-len(b_faces[100002]), 0, dtype=np.int32) + 1
    PerBFToF[100003] = np.arange(-len(b_faces[100003]), 0, dtype=np.int32) + 1

    for i in [2, 3]:
        PerBFToF[100000 + i] = PerBFToF[100000 + i].reshape((-1, 1))

    return VXY, K, Nv, EToV, b_faces, PerBToB, PerBFToF


ALPHA_OPT = [
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


def nodes_2d(N):
    alpha = ALPHA_OPT[N] if N < 16 else 5 / 3
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
    w1, w2, w3 = warp(N, L3 - L2), warp(N, L1 - L3), warp(N, L2 - L1)

    wb1 = b1 * w1 * (1 + (alpha * L1) ** 2)
    wb2 = b2 * w2 * (1 + (alpha * L2) ** 2)
    wb3 = b3 * w3 * (1 + (alpha * L3) ** 2)

    x = x + 1 * wb1 + np.cos(2 * np.pi / 3) * wb2 + np.cos(4 * np.pi / 3) * wb3
    y = y + 0 * wb1 + np.sin(2 * np.pi / 3) * wb2 + np.sin(4 * np.pi / 3) * wb3

    return x, y


def warp(N, rout):
    LGLr = jacobiGL(0, 0, N)
    req = np.linspace(-1, 1, N + 1)
    Veq = vandermonde1D(N, req)

    Nr = rout.size
    Pmat = np.empty((N + 1, Nr))
    for i in range(N + 1):
        Pmat[i, :] = jacobiP(rout, 0, 0, i)
    Lmat = np.linalg.solve(Veq.T, Pmat)

    warp = np.matmul(Lmat.T, (LGLr - req))

    zerof = np.abs(rout) < 1.0 - 1e-10
    sf = 1.0 - (zerof * rout) ** 2
    return warp / sf + warp * (zerof - 1)


def xytors(x, y):
    L1 = (np.sqrt(3.0) * y + 1.0) / 3.0
    L2 = (-3.0 * x - np.sqrt(3.0) * y + 2.0) / 6.0
    L3 = (3.0 * x - np.sqrt(3.0) * y + 2.0) / 6.0
    return -L2 + L3 - L1, -L2 - L3 + L1


def vandermonde2D(N, r, s):
    V2D = np.empty((r.size, (N + 1) * (N + 2) // 2))
    a, b = rs_to_ab(r, s)
    idx = 0
    for i in range(N + 1):
        for j in range(N - i + 1):
            V2D[:, idx] = simplex2DP(a, b, i, j)
            idx += 1
    return V2D


def rs_to_ab(r, s):
    Np = r.size
    a = np.empty((Np,))
    for n in range(Np):
        if s[n] != 1:
            a[n] = 2 * (1 + r[n]) / (1 - s[n]) - 1
        else:
            a[n] = -1
    return a, s


def simplex2DP(a, b, i, j):
    h1 = jacobiP(a, 0, 0, i)
    h2 = jacobiP(b, 2 * i + 1, 0, j)
    return np.sqrt(2.0) * h1 * h2 * (1 - b) ** i


def dmatrices2D(N, r, s, V):
    Vr, Vs = grad_vandermonde2D(N, r, s)
    return np.linalg.solve(V.T, Vr.T).T, np.linalg.solve(V.T, Vs.T).T


def grad_vandermonde2D(N, r, s):
    Np = (N + 1) * (N + 2) // 2
    V2Dr, V2Ds = np.empty((r.size, Np)), np.empty((r.size, Np))
    a, b = rs_to_ab(r, s)

    idx = 0
    for i in range(N + 1):
        for j in range(N - i + 1):
            V2Dr[:, idx], V2Ds[:, idx] = grad_simplex2DP(a, b, i, j)
            idx += 1
    return V2Dr, V2Ds


def grad_simplex2DP(a, b, i, j):
    fa, dfa = jacobiP(a, 0, 0, i), grad_jacobiP(a, 0, 0, i)
    gb, dgb = jacobiP(b, 2 * i + 1, 0, j), grad_jacobiP(b, 2 * i + 1, 0, j)
    dmodedr = dfa * gb
    dmodeds = dfa * (gb * (0.5 * (1 + a)))
    if i > 0:
        fact = (0.5 * (1 - b)) ** (i - 1)
        dmodeds *= fact
        dmodedr *= fact
    else:
        fact = np.zeros_like(b)
    dmodeds += fa * ((dgb * ((0.5 * (1 - b)) ** i) - 0.5 * i * gb * fact))
    return 2 ** (i + 0.5) * dmodedr, 2 ** (i + 0.5) * dmodeds


def solve2x2(A, b):
    den = A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]
    return np.array(
        [
            (b[0] * A[1, 1] - A[0, 1] * b[1]) / den,
            (b[1] * A[0, 0] - A[1, 0] * b[0]) / den,
        ]
    )


def mapFtoCflat(vmap):
    # map Fortran flattening to C flattening for x
    mapFtoC = np.arange(np.max(vmap)).reshape((-1, vmap.shape[1])).flatten(order="F")
    # map C flattening to Fortran flattening for VtoE
    mapCtoF = np.arange(vmap.size).reshape(vmap.shape, order="F").flatten()
    # VtoE with 0-based indexing and C flattening order
    return mapFtoC[(vmap - 1).flatten(order="F")][mapCtoF].astype(np.int64)


class Mesh2D:
    def __init__(self, params):
        assert isinstance(params, ScalarParam2D) or isinstance(params, EulerParam2D)
        self.params = params
        self.is_init = False

    def initialize(self):
        if self.is_init:
            print("WARNING: mesh is already initialized")
            return
        params = self.params
        self.has_periodic_bc = params.has_periodic_bc
        self.bc = params.bc
        self.N = params.N
        self.N2 = np.array(self.N**2, dtype=np.float64)
        self.Nfp = self.N + 1
        self.Np = (self.N + 1) * (self.N + 2) // 2
        if isinstance(params.mesh_file, str):
            (
                self.VXY,
                self.K,
                self.Nv,
                self.EToV0,
                self.BFaces,
                self.PerBToB_map,
                self.PerBFToF0,
            ) = read_gmsh_file(params.mesh_file)
        else:
            print("Building mesh without gmsh")
            (
                self.VXY,
                self.K,
                self.Nv,
                self.EToV0,
                self.BFaces,
                self.PerBToB_map,
                self.PerBFToF0,
            ) = make_gmsh(*params.mesh_file)
            ca, cb, nn = params.mesh_file
            self.min_hK = np.array((cb[0] - ca[0]) / nn[0], dtype=np.float64)

        print("Pre-computing geometric quantities")
        self.PerBFToF_map = {k: self.PerBFToF0[k] - 1 for k in self.PerBFToF0}
        self.EToV = self.EToV0 + 1
        self.VX, self.VY = self.VXY[:, 0], self.VXY[:, 1]
        print("Pre-computing geometric quantities: node")
        self.__compute_nodes()
        print("Pre-computing geometric quantities: matrices")
        self.__compute_basic_matrices()
        print("Pre-computing geometric quantities: coordinates")
        self.__compute_nodes_coordinates()
        print("Pre-computing geometric quantities: face mask")
        self.__compute_cell_face_masks()
        print("Pre-computing geometric quantities: surface integrals")
        self.__compute_surface_integrals()
        print("Pre-computing geometric quantities: averages")
        self.__compute_averaging()
        print("Pre-computing geometric quantities: transform factors")
        self.__compute_geometric_transform_factor()
        print("Pre-computing geometric quantities: face normal data")
        self.__compute_face_normal_data()
        print("Pre-computing geometric quantities: radius incircle")
        self.__compute_radius_incircle()
        print("Pre-computing geometric quantities: connectivity")
        self.__compute_connectivity_matrix()
        print("Pre-computing geometric quantities: coundary conditions")
        self.__compute_bc_tags()
        print("Pre-computing geometric quantities: ghost elements")
        self.__compute_ghost_elements_essential()
        print("Pre-computing geometric quantities: face maps")
        self.__compute_face_maps()
        print("Pre-computing geometric quantities: weak operators")
        self.__compute_weak_operators()
        print("Pre-computing geometric quantities: interpolation")
        self.__compute_interpolation_matrix()
        print("Pre-computing geometric quantities: essential bcs")
        self.__compute_essential_bc()
        if len(self.bc_essential):
            self.__compute_ghost_elements()

        self.min_dx = np.min(self.dx)
        self.vmapP_C = mapFtoCflat(self.vmapP)
        self.vmapM_C = mapFtoCflat(self.vmapM)

        def flt(x):
            return x.reshape((-1,), order="F")

        self.xf, self.yf = [flt(t) for t in [self.x, self.y]]

        print("Pre-computing geometric quantities: slices")
        self.__compute_slices()
        print("Pre-computing geometric quantities: meshgrid interp")
        self.__compute_meshgrid_interp_matrix()

        print("Transform to constant")
        for field in [
            "x",
            "y",
            "N2",
            "min_dx",
            "vmapP_C",
            "vmapM_C",
            "LIFT",
            "Fscale",
            "nx",
            "ny",
            "Dr",
            "Ds",
            "Drw",
            "Dsw",
            "rx",
            "ry",
            "sx",
            "sy",
            "dx2",
            "invV",
            "J",
            "neighbors",
            "min_hK",
            "M",
        ]:
            self.__dict__[field] = bkd.to_const(self.__dict__[field])
        self.is_init = True

    def __compute_slices(self):
        self.Ne1D = int(self.Nv**0.5) - 1
        hh = np.arange(self.Ne1D) / self.Ne1D
        self.slices = np.array([*np.add.outer(hh, self.x[: self.N, 0]).flatten(), 1])
        self.slices_idx = []
        for s in self.slices:
            i = np.where(np.abs(self.yf - s) < 1e-10)[0]
            i = i[np.argsort(self.xf[i], kind="stable")]
            self.slices_idx.append(i)

    def __compute_meshgrid_interp_matrix(self):
        self.Ne = self.x.shape[1]
        self.meshgrid_interp_ijv = []
        for kx in range(self.Ne):
            i, j = (kx // 2) // self.Ne1D, (kx // 2) % self.Ne1D
            ky = 0

            for m in range(self.Nfp):
                for n in range(self.Nfp - m):
                    k = kx + self.Ne * ky
                    if kx % 2:
                        mgi, mgj = self.Nfp * i + n + m, 2 * j + 1 - int(n > 0)
                        mg = mgi + mgj * self.Ne1D * self.Nfp
                        v = 0.5 if m == 0 else 1.0
                        if n == 0 or (m == 0 and n == self.Nfp - 1):
                            self.meshgrid_interp_ijv.append(np.array([mg, k, v]))
                    else:
                        mgi, mgj = self.Nfp * i + n, 2 * j + int(m > 0)
                        mg = mgi + mgj * self.Ne1D * self.Nfp
                        v = 0.5 if self.Nfp - m == n + 1 else 1.0
                        if m == 0 or m == self.Nfp - 1:
                            self.meshgrid_interp_ijv.append(np.array([mg, k, v]))

                    ky += 1
        self.meshgrid_interp_ijv = np.array(self.meshgrid_interp_ijv)
        self.meshgrid_interp = sparse.coo_matrix(
            (
                self.meshgrid_interp_ijv[:, 2],
                (
                    self.meshgrid_interp_ijv[:, 0].astype(int),
                    self.meshgrid_interp_ijv[:, 1].astype(int),
                ),
            )
        )

        if bkd.BACKEND == bkd.TORCH:
            # tmp = self.meshgrid_interp.todense()
            self.meshgrid_interp = bk.sparse_coo_tensor(
                self.meshgrid_interp_ijv[:, :2].T.astype(int),
                self.meshgrid_interp_ijv[:, 2],
            ).coalesce()
            # assert np.all(tmp == self.meshgrid_interp.to_dense().numpy())

    def __compute_nodes(self):
        self.xrs, self.yrs = nodes_2d(self.N)
        self.r, self.s = xytors(self.xrs, self.yrs)
        self.rs, self.r1, self.s1 = self.r + self.s, self.r + 1, self.s + 1

    def __compute_basic_matrices(self):
        self.V = vandermonde2D(self.N, self.r, self.s)
        self.invV = np.linalg.inv(self.V)
        self.M = np.matmul(self.invV.T, self.invV)
        self.Dr, self.Ds = dmatrices2D(self.N, self.r, self.s, self.V)

    def __compute_nodes_coordinates(self):
        self.x = 0.5 * (
            -np.outer(self.rs, self.VX[self.EToV0[:, 0]])
            + np.outer(self.r1, self.VX[self.EToV0[:, 1]])
            + np.outer(self.s1, self.VX[self.EToV0[:, 2]])
        )
        self.y = 0.5 * (
            -np.outer(self.rs, self.VY[self.EToV0[:, 0]])
            + np.outer(self.r1, self.VY[self.EToV0[:, 1]])
            + np.outer(self.s1, self.VY[self.EToV0[:, 2]])
        )

    def __compute_cell_face_masks(self):
        fmask1 = np.where(np.abs(self.s1).reshape((-1,), order="F") < MESH_TOL)[0]
        fmask2 = np.where(np.abs(self.rs).reshape((-1,), order="F") < MESH_TOL)[0]
        fmask3 = np.where(np.abs(self.r1).reshape((-1,), order="F") < MESH_TOL)[0]
        self.Fmask0 = np.stack([fmask1, fmask2, fmask3], axis=1)
        self.Fx = self.x[self.Fmask0.reshape((-1,), order="F"), :]
        self.Fy = self.y[self.Fmask0.reshape((-1,), order="F"), :]
        self.Fmask = self.Fmask0 + 1

    def __compute_surface_integrals(self):
        self.Emat = np.zeros((self.Np, 3 * self.Nfp))
        self.Vmid1D = vandermonde1D(self.N, np.zeros_like(self.r))

        faceR = self.r[self.Fmask0[:, 0]]
        V1D = vandermonde1D(self.N, faceR)
        self.M1D_1 = np.linalg.inv(np.matmul(V1D, V1D.T))
        self.Emat[self.Fmask0[:, 0], : self.Nfp] = self.M1D_1
        self.facemid1 = np.linalg.solve(V1D.T, self.Vmid1D[0, :].T).T

        faceR = self.r[self.Fmask0[:, 1]]
        V1D = vandermonde1D(self.N, faceR)
        self.M1D_2 = np.linalg.inv(np.matmul(V1D, V1D.T))
        self.Emat[self.Fmask0[:, 1], self.Nfp : 2 * self.Nfp] = self.M1D_2
        self.facemid2 = np.linalg.solve(V1D.T, self.Vmid1D[0, :].T).T

        faceS = self.s[self.Fmask0[:, 2]]
        V1D = vandermonde1D(self.N, faceS)
        self.M1D_3 = np.linalg.inv(np.matmul(V1D, V1D.T))
        self.Emat[self.Fmask0[:, 2], 2 * self.Nfp :] = self.M1D_3
        self.facemid3 = np.linalg.solve(V1D.T, self.Vmid1D[0, :].T).T

        self.LIFT = np.matmul(self.V, np.matmul(self.V.T, self.Emat))

    def __compute_averaging(self):
        self.AVG2D = np.sum(self.M, axis=0) / 2
        self.AVG1D_1 = np.sum(self.M1D_1, axis=0) / 2
        self.AVG1D_2 = np.sum(self.M1D_2, axis=0) / 2
        self.AVG1D_3 = np.sum(self.M1D_3, axis=0) / 2

    def __compute_geometric_transform_factor(self):
        self.xr = np.matmul(self.Dr, self.x)
        self.xs = np.matmul(self.Ds, self.x)
        self.yr = np.matmul(self.Dr, self.y)
        self.ys = np.matmul(self.Ds, self.y)
        self.J = -self.xs * self.yr + self.xr * self.ys
        self.rx = self.ys / self.J
        self.sx = -self.yr / self.J
        self.ry = -self.xs / self.J
        self.sy = self.xr / self.J

    def __compute_face_normal_data(self):
        Fmask_flat = self.Fmask0.reshape((-1,), order="F")
        self.fxr = self.xr[Fmask_flat, :]
        self.fxs = self.xs[Fmask_flat, :]
        self.fyr = self.yr[Fmask_flat, :]
        self.fys = self.ys[Fmask_flat, :]

        self.nx = np.empty((3 * self.Nfp, self.K))
        self.ny = np.empty((3 * self.Nfp, self.K))
        fid1 = np.arange(self.Nfp)
        fid2 = np.arange(self.Nfp, 2 * self.Nfp)
        fid3 = np.arange(2 * self.Nfp, 3 * self.Nfp)

        self.nx[fid1, :] = self.fyr[fid1, :]
        self.ny[fid1, :] = -self.fxr[fid1, :]

        self.nx[fid2, :] = self.fys[fid2, :] - self.fyr[fid2, :]
        self.ny[fid2, :] = -self.fxs[fid2, :] + self.fxr[fid2, :]

        self.nx[fid3, :] = -self.fys[fid3, :]
        self.ny[fid3, :] = self.fxs[fid3, :]

        self.sJ = np.sqrt(self.nx**2 + self.ny**2)
        self.nx = self.nx / self.sJ
        self.ny = self.ny / self.sJ

        self.Fscale = self.sJ / self.J[Fmask_flat, :]

    def __compute_radius_incircle(self):
        vmask1 = np.where(np.abs(self.s + self.r + 2) < MESH_TOL)[0]
        vmask2 = np.where(np.abs(self.r - 1) < MESH_TOL)[0]
        vmask3 = np.where(np.abs(self.s - 1) < MESH_TOL)[0]
        vmask = np.hstack([vmask1, vmask2, vmask3])
        vx = self.x[vmask.reshape((-1,), order="F"), :]
        vy = self.y[vmask.reshape((-1,), order="F"), :]

        len1 = np.sqrt((vx[0, :] - vx[1, :]) ** 2 + (vy[0, :] - vy[1, :]) ** 2)
        len2 = np.sqrt((vx[1, :] - vx[2, :]) ** 2 + (vy[1, :] - vy[2, :]) ** 2)
        len3 = np.sqrt((vx[2, :] - vx[0, :]) ** 2 + (vy[2, :] - vy[0, :]) ** 2)
        sper = (len1 + len2 + len3) / 2.0
        area = np.sqrt(sper * (sper - len1) * (sper - len2) * (sper - len3))

        self.dx = area / sper
        self.dx2 = len1 * len2 * len3 / area / 4

    def __lfind(self, f):
        elem_listA, _ = np.where(self.EToV == f[0])
        elem_listB, _ = np.where(self.EToV == f[1])
        elem = list(set(elem_listA) & set(elem_listB))
        assert len(elem) == 1
        _, lind = np.where(self.EToV[elem, :] == f[0])
        return (
            elem,
            lind if (self.EToV[elem, lind % 3] == f[1]).all() else lind % 3,
        )

    def __compute_connectivity_matrix(self):
        total_faces = 3 * self.K
        i = np.repeat(np.arange(total_faces), 2)
        j = self.EToV0[:, [0, 1, 1, 2, 0, 2]].reshape((-1,))
        SpFToV = sparse.coo_matrix((np.ones_like(i), (i, j)))
        SpFToF = (SpFToV * SpFToV.T - 2 * sparse.eye(total_faces)).tocoo()

        idx = np.where(SpFToF.data == 2)[0]
        faces1, faces2 = SpFToF.row[idx], SpFToF.col[idx]
        face1, face2 = faces1 % 3, faces2 % 3
        element1, element2 = faces1 // 3, faces2 // 3

        self.EToE = np.outer(
            np.arange(self.K, dtype=np.int32), np.ones((3,), dtype=np.int32)
        )
        self.EToF = np.outer(
            np.ones((self.K,), dtype=np.int32), np.arange(3, dtype=np.int32)
        )
        self.EToE[element1, face1] = element2
        self.EToF[element1, face1] = face2

        self.PShift = {}
        if self.has_periodic_bc:
            for bf1 in sorted(self.PerBToB_map.keys()):  # sorted just to match Matlab
                bf2 = self.PerBToB_map[bf1]
                face_list1 = self.BFaces[bf1]
                face_list2 = self.BFaces[bf2]
                pflink = self.PerBFToF_map[bf1]
                for j in range(face_list1.shape[0]):
                    f1 = face_list1[j, :]
                    elem1, lfind1 = self.__lfind(f1)
                    f2 = face_list2[np.abs(pflink[j]) - 1, :].reshape((-1,))
                    elem2, lfind2 = self.__lfind(f2)

                    self.EToE[elem1, lfind1] = elem2
                    self.EToE[elem2, lfind2] = elem1
                    self.EToF[elem1, lfind1] = lfind2
                    self.EToF[elem2, lfind2] = lfind1

                    v1 = f1[0] - 1
                    v2 = f2[1 - (pflink[j] > 0)] - 1

                    x1, y1 = self.VX[v1], self.VY[v1]
                    x2, y2 = self.VX[v2], self.VY[v2]

                    def insert(key, e):
                        if key in self.PShift:
                            self.PShift[key] += e
                        else:
                            self.PShift[key] = e

                    insert(elem1[0], [elem2, x2 - x1, y2 - y1])
                    insert(elem2[0], [elem1, x1 - x2, y1 - y2])
        self.EToF, self.EToF0 = self.EToF + 1, self.EToF
        self.EToE, self.EToE0 = self.EToE + 1, self.EToE
        self.PShift0 = {
            k: np.concatenate(self.PShift[k]).reshape((-1, 3)) for k in self.PShift
        }
        self.PShift = {
            k + 1: np.hstack([self.PShift0[k][:, :1] + 1, self.PShift0[k][:, 1:]])
            for k in self.PShift0
        }

    def __compute_bc_tags(self):
        self.BCTag = np.zeros_like(self.EToV0)
        for key in self.BFaces:
            if self.bc[key] != BC.Periodic:
                bc_ind = np.zeros_like(self.EToV0, dtype=bool)
                face_list = self.BFaces[key]
                for j in range(face_list.shape[0]):
                    el, lf = self.__lfind(face_list[j, :])
                    bc_ind[el, lf] = True
                self.BCTag += int(self.bc[key]) * bc_ind

    def __compute_ghost_elements(self):
        self.__compute_ghost_elements1()
        self.__compute_ghost_elements2()
        self.__compute_ghost_elements3()

    def __compute_ghost_elements_essential(self):
        xyvs = [self.VXY[self.EToV0[:, i], j] for i in range(3) for j in range(2)]
        self.xv1, self.yv1, self.xv2, self.yv2, self.xv3, self.yv3 = xyvs
        self.fnx = np.vstack(
            [self.yv2 - self.yv1, self.yv3 - self.yv2, self.yv1 - self.yv3]
        )
        self.fny = np.vstack(
            [self.xv1 - self.xv2, self.xv2 - self.xv3, self.xv3 - self.xv1]
        )
        self.fL = np.sqrt(self.fnx**2 + self.fny**2)
        self.fcx = (
            np.vstack([self.xv1 + self.xv2, self.xv2 + self.xv3, self.xv3 + self.xv1])
            / 2
        )
        self.fcy = (
            np.vstack([self.yv1 + self.yv2, self.yv2 + self.yv3, self.yv3 + self.yv1])
            / 2
        )

        vind = np.array([2, 0, 1])
        xyvns = [np.empty((self.K,)) for _ in range(6)]
        for i in range(self.K):
            vns = [
                self.EToV0[self.EToE0[i, j], vind[self.EToF0[i, j]]] for j in range(3)
            ]
            for j in range(3):
                for k in range(2):
                    xyvns[2 * j + k][i] = self.VXY[vns[j], k]
            if i in self.PShift0 and self.has_periodic_bc:
                pairdata = self.PShift0[i]
                for j in range(3):
                    p_ind = np.where(pairdata[:, 0] == self.EToE0[i, j])[0]
                    if p_ind.size:
                        for k in range(2):
                            xyvns[2 * j + k][i] -= pairdata[p_ind, k + 1]
        ids = [np.where(self.BCTag[:, j])[0] for j in range(3)]
        A0 = 2 * np.matmul(self.AVG2D, self.J)
        for j in range(3):
            H = 2 * A0[ids[j]] / self.fL[0, ids[j]]
            xyvns[2 * j][ids[j]] += 2 * self.fnx[j, ids[j]] * H / self.fL[j, ids[j]]
            xyvns[2 * j + 1][ids[j]] += 2 * self.fny[j, ids[j]] * H / self.fL[j, ids[j]]
        self.xvn1, self.yvn1, self.xvn2, self.yvn2, self.xvn3, self.yvn3 = xyvns

        self.EToGE0 = -np.ones_like(self.EToE0)
        self.KG = sum([ids[j].size for j in range(3)])
        self.xG = np.empty((self.Np, self.KG))
        self.yG = np.empty((self.Np, self.KG))
        ranges = [0, ids[0].size, ids[0].size + ids[1].size, self.KG]
        perm = np.array([1, 2, 0])
        for j in range(3):
            self.EToGE0[ids[j], 0] = np.arange(ranges[j], ranges[j + 1])
            self.xG[:, ranges[j] : ranges[j + 1]] = 0.5 * (
                -np.outer(self.rs, xyvs[2 * j][ids[j]])
                + np.outer(self.r1, xyvns[2 * j][ids[j]])
                + np.outer(self.s1, xyvs[2 * perm[j]][ids[j]])
            )
            self.yG[:, ranges[j] : ranges[j + 1]] = 0.5 * (
                -np.outer(self.rs, xyvs[2 * j + 1][ids[j]])
                + np.outer(self.r1, xyvns[2 * j + 1][ids[j]])
                + np.outer(self.s1, xyvs[2 * perm[j] + 1][ids[j]])
            )
        self.EToGE = self.EToGE0 + 1

    def __compute_ghost_elements1(self):
        self.vcx = [
            self.xv1 + self.xv2 + self.xv3,
            self.xv1 + self.xv2 + self.xvn1,
            self.xv2 + self.xv3 + self.xvn2,
            self.xv3 + self.xv1 + self.xvn3,
        ]
        self.vcy = [
            self.yv1 + self.yv2 + self.yv3,
            self.yv1 + self.yv2 + self.yvn1,
            self.yv2 + self.yv3 + self.yvn2,
            self.yv3 + self.yv1 + self.yvn3,
        ]
        self.vcx, self.vcy = np.vstack(self.vcx) / 3, np.vstack(self.vcy) / 3

    def __compute_ghost_elements2(self):
        self.patch_alphas = np.empty((3, 3, self.K))
        self.patch_alphasA = np.empty((3, 2, 2, self.K))
        self.patch_alphasB = np.empty((3, 2, self.K))
        for i in range(self.K):
            cfx, cfy = self.fcx[:, i] - self.vcx[0, i], self.fcy[:, i] - self.vcy[0, i]
            cfL = np.sqrt(cfx**2 + cfy**2)

            c2cx, c2cy = (
                self.vcx[1:, i] - self.vcx[0, i],
                self.vcy[1:, i] - self.vcy[0, i],
            )
            c2cL = np.sqrt(c2cx**2 + c2cy**2)

            Kind = np.array([0, 1, 2, 0, 1])
            for j in range(3):
                A = np.array(
                    [
                        [c2cx[Kind[j]], c2cx[Kind[j + 1]]],
                        [c2cy[Kind[j]], c2cy[Kind[j + 1]]],
                    ]
                )
                b = np.array([cfx[Kind[j]], cfy[Kind[j]]])
                alphas = solve2x2(A, b)
                if alphas[0] < -MESH_TOL or alphas[1] < -MESH_TOL:
                    A = np.array(
                        [
                            [c2cx[Kind[j]], c2cx[Kind[j + 2]]],
                            [c2cy[Kind[j]], c2cy[Kind[j + 2]]],
                        ]
                    )
                    alphas = solve2x2(A, b)
                    assert (
                        alphas[0] >= -MESH_TOL and alphas[1] >= -MESH_TOL
                    ), "The mesh traingulation is not appropriate"
                    self.patch_alphas[j, -1, i] = Kind[j + 2]
                else:
                    self.patch_alphas[j, -1, i] = Kind[j + 1]
                self.patch_alphas[j, :2, i] = alphas

                self.patch_alphasA[j, :, :, i] = A
                self.patch_alphasB[j, :, i] = b
        self.patch_alphas0 = self.patch_alphas
        self.patch_alphas[:, -1, :] += 1

    def __compute_ghost_elements3(self):
        self.MMAP = np.empty((self.Np, 3))
        xx, yy = np.array([-1, 1, -1]), np.array([-1, -1, 1])
        vindm = np.array([[0, 2, 1], [1, 0, 2], [2, 1, 0]])
        xo = 0.5 * (-self.rs * xx[0] + self.r1 * xx[1] + self.s1 * xx[2])
        yo = 0.5 * (-self.rs * yy[0] + self.r1 * yy[1] + self.s1 * yy[2])
        for i in range(3):
            xm = 0.5 * (
                -self.rs * xx[vindm[i, 0]]
                + self.r1 * xx[vindm[i, 1]]
                + self.s1 * xx[vindm[i, 2]]
            )
            ym = 0.5 * (
                -self.rs * yy[vindm[i, 0]]
                + self.r1 * yy[vindm[i, 1]]
                + self.s1 * yy[vindm[i, 2]]
            )
            D = np.subtract.outer(xo, xm) ** 2 + np.subtract.outer(yo, ym) ** 2
            idM, idP = np.where(np.sqrt(np.abs(D)) < MESH_TOL)
            self.MMAP[:, i] = idM[np.argsort(idP)]
        self.MMAP0 = self.MMAP
        self.MMAP = self.MMAP0 + 1

    def __compute_face_maps(self):
        self.__compute_face_maps1()
        self.__compute_face_maps2()
        self.__compute_face_maps3()
        self.__compute_face_maps4()

    def __compute_face_maps1(self):
        self.nodeids = np.arange(self.K * self.Np).reshape((self.Np, self.K), order="F")
        self.gnodeids = np.arange(self.KG * self.Np).reshape(
            (self.Np, self.KG), order="F"
        )
        self.vmapM = np.empty((self.Nfp, 3, self.K), dtype=np.int32)
        self.vmapP = np.empty((self.Nfp, 3, self.K), dtype=np.int32)
        self.mapM = np.arange(self.K * self.Nfp * 3)
        self.mapP = np.arange(self.K * self.Nfp * 3).reshape(
            (self.Nfp, 3, self.K), order="F"
        )
        self.mapB = np.arange(self.KG * self.Nfp).reshape(
            (self.Nfp, self.KG), order="F"
        )
        self.vmapB = np.empty((self.Nfp, self.KG))

    def __compute_face_maps2(self):
        for k1 in range(self.K):
            for f1 in range(3):
                self.vmapM[:, f1, k1] = self.nodeids[self.Fmask0[:, f1], k1]

    def __compute_face_maps3(self):
        flat_x = self.x.reshape((-1,), order="F")
        flat_y = self.y.reshape((-1,), order="F")
        for k1 in range(self.K):
            for f1 in range(3):
                k2, f2 = self.EToE0[k1, f1], self.EToF0[k1, f1]
                v1, v2 = self.EToV0[k1, f1], self.EToV0[k1, (f1 + 1) % 3]
                refd = np.sqrt(
                    (self.VX[v1] - self.VX[v2]) ** 2 + (self.VY[v1] - self.VY[v2]) ** 2
                )

                xs, ys = 0, 0
                if k1 in self.PShift0 and self.has_periodic_bc:
                    pairdata = self.PShift0[k1]
                    p_ind = np.where(pairdata[:, 0] == k2)[0]
                    if p_ind.size:
                        assert p_ind.size == 1
                        xs, ys = pairdata[p_ind[0], 1:]
                vidM, vidP = self.vmapM[:, f1, k1], self.vmapM[:, f2, k2]
                x1, y1 = flat_x[vidM], flat_y[vidM]
                x2, y2 = flat_x[vidP] - xs, flat_y[vidP] - ys

                D = np.subtract.outer(x1, x2) ** 2 + np.subtract.outer(y1, y2) ** 2
                idM, idP = np.where(np.sqrt(np.abs(D)) < MESH_TOL * refd)
                self.vmapP[idM, f1, k1] = vidP[idP]
                self.mapP[idM, f1, k1] = idP + f2 * self.Nfp + k2 * 3 * self.Nfp

                if k2 == k1:
                    kg2 = self.EToGE0[k1, f1]
                    vidP = self.gnodeids[self.Fmask0[:, 3], kg2]
                    x2, y2 = (self.xG[vidP] - xs), (self.yG[vidP] - ys)
                    D = np.subtract.outer(x1, x2) ** 2 + np.subtract.outer(y1, y2) ** 2
                    idM, idP = np.where(np.sqrt(np.abs(D)) < MESH_TOL * refd)
                    self.vmapB[idP, kg2] = vidM[idM]
                    self.mapB[idP, kg2] = idM + f1 * self.Nfp + k1 * 3 * self.Nfp

    def __compute_face_maps4(self):
        self.vmapP = self.vmapP.reshape((-1), order="F")
        self.vmapM = self.vmapM.reshape((-1), order="F")
        self.mapP = self.mapP.reshape((-1), order="F")
        self.vmapM = self.vmapM.reshape((self.Nfp * 3, self.K), order="F")
        self.vmapP = self.vmapP.reshape((self.Nfp * 3, self.K), order="F")

        self.vmapM0, self.vmapP0 = self.vmapM, self.vmapP
        self.vmapM, self.vmapP = self.vmapM0 + 1, self.vmapP0 + 1

        self.mapP0, self.mapM0, self.mapB0 = self.mapP, self.mapM, self.mapB
        self.mapP, self.mapM, self.mapB = self.mapP0 + 1, self.mapM0 + 1, self.mapB0 + 1

    def __compute_weak_operators(self):
        self.Vr, self.Vs = grad_vandermonde2D(self.N, self.r, self.s)
        denT = np.matmul(self.V, self.V.T).T
        self.Drw = np.linalg.solve(denT, np.matmul(self.V, self.Vr.T).T).T
        self.Dsw = np.linalg.solve(denT, np.matmul(self.V, self.Vs.T).T).T

    def __compute_interpolation_matrix(self):
        self.__compute_interpolation_matrix1()
        self.__compute_interpolation_matrix2()

    def __compute_interpolation_matrix1(self):
        ij, idx = np.ones((self.VX.size * 6, 2), dtype=np.int32), 0

        self.EToV0T_flat = self.EToV0.T.reshape((-1,), order="F")
        for i in range(self.VX.size):
            js = np.where(self.EToV0T_flat == i)[0] // 3
            k = js.size
            ij[idx : idx + k, 0] *= i
            ij[idx : idx + k, 1] = js
            idx += k

        self.VToE = sparse.coo_matrix((np.ones((idx,)), (ij[:idx, 0], ij[:idx, 1])))
        self.neighbors = np.empty(self.VToE.shape[0])
        self.neighbors[:] = np.sum(self.VToE, axis=1).reshape((-1,))

        if bkd.BACKEND == bkd.TORCH:
            self.VToE = bk.sparse_coo_tensor(
                ij[:idx, :2].T.astype(int),
                np.ones((idx,)),
            ).coalesce()

    def __compute_interpolation_matrix2(self):
        coov = np.stack(
            [
                self.VX[self.EToV0T_flat],
                self.VY[self.EToV0T_flat],
                np.ones_like(self.EToV0T_flat),
            ],
            axis=1,
        )
        for i in range(self.K):
            coov[3 * i : 3 * (i + 1), :] = np.linalg.inv(coov[3 * i : 3 * (i + 1), :])
        i = np.tile(np.arange(coov.shape[0]), 3)
        j = np.repeat(
            np.concatenate([np.arange(i, 3 * self.K, 3) for i in range(3)]), 3
        )
        self.coov_inv = sparse.coo_matrix((coov.reshape((-1,), order="F"), (i, j)))

        coov = np.stack(
            [
                self.x.reshape((-1), order="F"),
                self.y.reshape((-1), order="F"),
                np.ones(
                    (self.x.size),
                ),
            ],
            axis=1,
        )
        i = np.tile(np.arange(coov.shape[0]), 3)
        j = np.repeat(
            np.concatenate([np.arange(i, 3 * self.K, 3) for i in range(3)]), self.Np
        )
        self.coov_dof = sparse.coo_matrix((coov.reshape((-1,), order="F"), (i, j)))

        self.interp_matrix = (self.coov_dof @ self.coov_inv).tocoo()

        if bkd.BACKEND == bkd.TORCH:
            self.interp_matrix = bk.sparse_coo_tensor(
                indices=bk.tensor(
                    [self.interp_matrix.row, self.interp_matrix.col], dtype=bk.int64
                ),
                values=self.interp_matrix.data,
                size=self.interp_matrix.shape,
            )

    def __compute_essential_bc(self):
        self.bc_essential = {
            k: self.bc[k] for k in self.bc if self.bc[k] != BC.Periodic
        }

        self.GEBC_list = {}
        self.mapBC_list = {}
        self.vmapBC_list = {}

        if len(self.bc_essential):
            ind = np.where(self.BCTag.reshape((-1,), order="F") != 0)[0]
            bct = np.zeros((self.KG,))
            assert np.all(self.EToGE0[ind] >= 0)
            bct[self.EToGE0[ind]] = self.BCTag[ind]
            bnodes = np.outer(np.ones((self.Nfp,)), bct.reshape((-1,), order="F"))
            bnodes = bnodes.reshape((-1,), order="F")

            for key in self.bc_essential:
                tag = int(self.bc_essential[key])
                bc_ind = np.where(bnodes == tag)[0]
                self.GEBC_list[tag] = np.where(bct == tag)[0]
                self.mapBC_list[tag] = self.mapB[bc_ind]
                self.vmapBC_list[tag] = self.vmapB[bc_ind]
