import math

import pyndg.backend as bkd

import pyndg.core.basis as basis
import pyndg.core.la as la

import numpy as np

_CELL_TOL = 1e-8


REF_NORMALS = [
    np.array([[1], [-1]]),
    np.array([[0, -1], [1, 1], [-1, 0]]),
    np.array([[0, 0, -1], [0, -1, 0], [1, 1, 1], [-1, 0, 0]]),
]


def nodes_1d_rst(N):
    return basis.jacobi_gl(0, 0, N).reshape((1, -1))


def nodes_2d_rst(N):
    return np.stack(basis.xy_to_rs(*basis.nodes_2d(N)))


def nodes_3d_rst(N):
    return np.stack(basis.xyz_to_rst(*basis.nodes_3d(N)))


def nodes(dim, N):
    nodes_fns = [nodes_1d_rst, nodes_2d_rst, nodes_3d_rst]
    return nodes_fns[dim - 1](N)


def vandermonde(dim, N, rst):
    assert rst.ndim == 2
    vandermonde_fns = [
        lambda *_: bkd.np_prec(1.0),
        basis.vandermonde_1d,
        basis.vandermonde_2d,
        basis.vandermonde_3d,
    ]
    return vandermonde_fns[dim](N, *[u for u in rst])


def dmatrices(dim, N, rst, V):
    assert rst.ndim == 2
    dmatrices_fns = [basis.dmatrix_1d, basis.dmatrices_2d, basis.dmatrices_3d]
    dmatrices = dmatrices_fns[dim - 1](N, *[u for u in rst], V)
    return np.stack(dmatrices).reshape((dim, *V.shape))


def grad_vandermonde(dim, N, rst):
    assert rst.ndim == 2
    grad_vandermonde_fns = [
        basis.grad_vandermonde_1d,
        basis.grad_vandermonde_2d,
        basis.grad_vandermonde_3d,
    ]
    grad_V = grad_vandermonde_fns[dim - 1](N, *[u for u in rst])
    return np.stack(grad_V).reshape((dim, rst.shape[1], -1))


_BARICENTRIC_COORDS_PERM = [
    [0, 1],
    [2, 0, 1],
    [3, 2, 0, 1],
]


def _compute_barycentric_coordinates(rst):
    dim = rst.shape[0]
    bary_coord = [-rst.sum(axis=0) - dim + 2.0] + [rst[d] + 1.0 for d in range(dim)]
    perm = _BARICENTRIC_COORDS_PERM[dim - 1]
    fmasks = [
        np.where(np.abs(bary_coord[perm[d]]) < _CELL_TOL)[0] for d in range(dim + 1)
    ]
    return np.stack(bary_coord), np.stack(fmasks)


# spatial dimension idxs that parametrize each face
_PARAM_INDXS = [
    [[0], [0]],
    [[0], [0], [1]],
    [[0, 1], [0, 2], [1, 2], [1, 2]],
]


class ReferenceElementOps:
    def __init__(self, dim: int, N: int):
        assert isinstance(dim, int) and dim in [1, 2, 3]
        assert isinstance(N, int) and N >= 1

        # number of nodes in the reference element
        self.Np = math.comb(N + dim, dim)
        # number of nodes on each face
        self.Nfp = math.comb(N + dim - 1, dim - 1)
        # number of faces
        self.Nf = dim + 1

        # nodes on the reference element centered in (-1,)*dim, bbox size=2
        self.rst = nodes(dim, N)

        # basic matrices
        self.V = vandermonde(dim, N, self.rst)
        self.invV = la.solve_lu(self.V, np.eye(*self.V.shape, dtype=bkd.np_prec))

        # mass matrix
        self.int_phiphi = self.invV.T @ self.invV

        # strong form differentiation matrix (Dr, Ds, Dt)
        self.Dphi = dmatrices(dim, N, self.rst, self.V)
        # weak form differentiation, aka S = M @ Dphi
        self.int_phiDphi = np.stack(
            [self.int_phiphi @ self.Dphi[i] for i in range(dim)]
        )

        # weak form differentiation matrix, equivalent to M^-1 @ int_phiDphi
        self.grad_V = grad_vandermonde(dim, N, self.rst)  # (Vr, Vs, Vt)
        self.Dphi_weak = np.stack(
            [(self.int_phiphi @ (self.grad_V[i] @ self.V.T)).T for i in range(dim)]
        )

        # stiffness components
        self.int_DphiDphi = np.empty((dim, dim, *self.V.shape), dtype=bkd.np_prec)
        for i in range(dim):
            for j in range(dim):
                self.int_DphiDphi[i, j] = (
                    self.Dphi[i].T @ self.int_phiphi @ self.Dphi[j]
                )

        # local element indices for nodes on faces
        self.bary_coord, self.fmasks = _compute_barycentric_coordinates(self.rst)
        Nfp = self.fmasks.shape[1]

        # face mass matrix
        self.face_V = np.empty((dim + 1, Nfp, Nfp), dtype=bkd.np_prec)
        self.face_int_phiphi = np.empty((dim + 1, Nfp, Nfp), dtype=bkd.np_prec)
        prolongation = np.zeros((dim + 1, self.V.shape[0], Nfp), dtype=bkd.np_prec)
        for d in range(dim + 1):
            # get face parameterization
            face_rst = self.rst[_PARAM_INDXS[dim - 1][d]][:, self.fmasks[d]]

            self.face_V[d] = vandermonde(dim - 1, N, face_rst)
            # prolong the face basis to the volume nodes
            # F[fmasks[d]] = x <==> prolongation[d] @ x == F
            prolongation[d, self.fmasks[d], np.arange(Nfp)] = 1.0
            # M_face = inv(V_face*V_face^T)
            self.face_int_phiphi[d] = la.solve_lu(
                self.face_V[d] @ self.face_V[d].T, np.eye(Nfp, dtype=bkd.np_prec)
            )

        # inv(mass matrix) \int_face (phi_i, phi_j)
        self.lift = np.hstack(
            [
                self.V @ (self.V.T @ (prolongation[d] @ self.face_int_phiphi[d]))
                for d in range(dim + 1)
            ]
        )

        # - face mass matrix
        # - lift
        # - limiter/av stuff
