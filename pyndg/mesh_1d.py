# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import gamma

from . import backend as bkd

from .utils import TOL
from .scalar_param_1d import ScalarParam1D


def generate_mesh_1d(xmin, xmax, K, mesh_per):
    Nv = K + 1
    h = (xmax - xmin) / K
    pert_scale = mesh_per * h
    VX = np.linspace(xmin, xmax, Nv)

    if mesh_per > 0:
        pert = (np.random.rand(Nv - 2) - 0.5) * pert_scale
        VX[1:-1] += pert
    hK = VX[1:] - VX[:-1]
    return Nv, VX, hK


def pnleg(x, n):
    if n == 0:
        return np.ones_like(x)
    else:
        p1 = np.ones_like(x)
        p2, p3 = x, x
        for k in range(1, n):
            p3 = ((2 * k + 1) * x * p2 - k * p1) / (k + 1)
            p1, p2 = p2, p3
        return p3


def jacobiGL(alpha, beta, N):
    if N == 1:
        return np.array([-1.0, 1.0])
    else:
        xint, w = jacobiGQ(alpha + 1, beta + 1, N - 2)
        return np.array([-1, *np.sort(xint), 1])


def jacobiGQ(alpha, beta, N):
    if N == 0:
        return np.array([(beta - alpha) / (alpha + beta + 2)]), np.array([2])
    else:
        h1 = 2 * np.arange(0, N + 1) + alpha + beta
        J = np.diag(-0.5 * (alpha ** 2 - beta ** 2) / (h1 + 2) / h1) + np.diag(
            2.0
            / (h1[:N] + 2)
            * np.sqrt(
                np.arange(1, N + 1)
                * (np.arange(1, N + 1) + alpha + beta)
                * (np.arange(1, N + 1) + alpha)
                * (np.arange(1, N + 1) + beta)
                / (h1[:N] + 1)
                / (h1[:N] + 3)
            ),
            1,
        )
        if alpha + beta < TOL:
            J[0, 0] = 0.0
        J = J + J.T
        x, V = np.linalg.eig(J)
        w = (
            V[0, :].T ** 2
            * 2 ** (alpha + beta + 1)
            / (alpha + beta + 1)
            * gamma(alpha + 1)
            * gamma(beta + 1)
            / gamma(alpha + beta + 1)
        )
        return x, w


def vandermonde1D(N, r):
    assert len(r.shape) == 1, "len(r.shape) != 1"
    V1D = np.empty((len(r), N + 1))
    for j in range(0, N + 1):
        V1D[:, j] = jacobiP(r, 0, 0, j)
    return V1D


def jacobiP(x, alpha, beta, N):
    xp = x
    assert len(xp.shape) == 1, "len(xp.shape) != 1"
    n = len(xp)
    PL = np.empty((N + 1, n))

    gamma0 = (
        2 ** (alpha + beta + 1)
        / (alpha + beta + 1)
        * gamma(alpha + 1)
        * gamma(beta + 1)
        / gamma(alpha + beta + 1)
    )
    PL[0, :] = 1.0 / np.sqrt(gamma0)
    if N == 0:
        return PL.reshape((n,))
    gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0
    PL[1, :] = ((alpha + beta + 2) * xp / 2 + (alpha - beta) / 2) / np.sqrt(gamma1)
    if N == 1:
        return PL[N, :].reshape((n,))
    aold = (
        2 / (2 + alpha + beta) * np.sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3))
    )

    for i in range(N - 1):
        h1 = 2 * (i + 1) + alpha + beta
        anew = (
            2
            / (h1 + 2)
            * np.sqrt(
                (i + 2)
                * (i + 2 + alpha + beta)
                * (i + 2 + alpha)
                * (i + 2 + beta)
                / (h1 + 1)
                / (h1 + 3)
            )
        )
        bnew = -(alpha ** 2 - beta ** 2) / h1 / (h1 + 2)
        PL[i + 2, :] = 1 / anew * (-aold * PL[i, :] + (xp - bnew) * PL[i + 1, :])
        aold = anew
    return PL[N, :].reshape((n,))


def dmatrix1D(N, r, V):
    Vr = grad_vandermonde1D(N, r)
    return np.linalg.solve(V.T, Vr.T).T


def grad_vandermonde1D(N, r):
    assert len(r.shape) == 1, "len(r.shape) != 1"
    DVr = np.empty((len(r), N + 1))
    for i in range(N + 1):
        DVr[:, i] = grad_jacobiP(r, 0, 0, i)
    return DVr


def grad_jacobiP(r, alpha, beta, N):
    if N == 0:
        return np.zeros((len(r),))
    else:
        return np.sqrt(N * (N + alpha + beta + 1)) * jacobiP(
            r, alpha + 1, beta + 1, N - 1
        )


class Mesh1D:
    def __init__(self, params):
        # assert isinstance(params, ScalarParam1D)
        self.N = params.N
        self.N2 = np.array(self.N ** 2, dtype=np.float64)
        self.K = params.K
        self.bnd = params.bnd
        self.bnd_l = params.bnd[0]
        self.bnd_r = params.bnd[1]
        self.mesh_pert = params.mesh_pert

        self.Nv, self.VX, self.hK = generate_mesh_1d(
            self.bnd[0], self.bnd[1], self.K, self.mesh_pert
        )
        self.min_hK = np.min(self.hK)
        self.Np = self.N + 1
        self.r = jacobiGL(0, 0, self.N)
        coef = 2 / (self.N * self.Np)
        self.w = coef / (pnleg(self.r, self.N) * pnleg(self.r, self.N))
        self.V = vandermonde1D(self.N, self.r)
        self.invV = np.linalg.inv(self.V)
        self.Dr = dmatrix1D(self.N, self.r, self.V)
        self.M = np.linalg.solve(self.V.T, np.linalg.inv(self.V.T).T).T
        self.invM = np.linalg.inv(self.M)
        self.S = np.matmul(self.M, self.Dr)
        self.ST = self.S.T
        self.int_metric = 2 * np.matmul(
            np.ones((self.Np, 1)), (1 / self.hK).reshape((1, -1))
        )
        self.Imat = np.eye(self.Np)
        self.avg1D = np.sum(self.M, axis=0) / 2
        tmp1 = np.matmul(np.ones((self.Np, 1)), self.VX[None, 0 : self.K])
        tmp2 = 0.5 * np.matmul(
            (self.r + 1).reshape((-1, 1)),
            (self.VX[1 : self.Nv] - self.VX[0 : self.Nv - 1]).reshape((1, -1)),
        )
        self.x = tmp1 + tmp2
        self.h = self.x[-1, :] - self.x[0, :]
        self.VtoE = np.zeros((2, self.K))
        for j in range(self.K):
            self.VtoE[0, j] = j * self.Np + 1
            self.VtoE[1, j] = (j + 1) * self.Np

        # map Fortran flattening to C flattening for x
        self.mapFtoC = np.arange(self.x.size).reshape(self.x.shape).flatten(order="F")
        # map C flattening to Fortran flattening for VtoE
        self.mapCtoF = (
            np.arange(self.VtoE.size).reshape(self.VtoE.shape, order="F").flatten()
        )
        # VtoE with 0-based indexing and C flattening order
        self.VtoE0 = self.mapFtoC[(self.VtoE.astype(np.int64) - 1).flatten(order="F")]
        self.VtoE0 = self.VtoE0[self.mapCtoF].astype(np.int64)

        for field in self.__dict__.keys():
            self.__dict__[field] = bkd.to_const(self.__dict__[field])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    from physic_model import Advection1D
    from viscosity_model import EV1D
    from time_integrator import SSP3
    from bc_1d import BC
    from test_matlab_1d import test_fields

    def test_mesh_perturbation(p):
        Nv, VX, hK = generate_mesh_1d(0, 1, 10, p)
        plt.plot(VX, np.zeros_like(VX), marker="o")

    def test_mesh(N, K):
        p = ScalarParam1D(
            model=Advection1D(),
            name=f"Test_N{N}_K{K}",
            N=N,
            K=K,
            u_IC=lambda x: x,
            bnd=(0.0, 1.0),
            mesh_pert=0.0,
            bc=((BC.Periodic, 0.0), (BC.Periodic, 0.0)),
            final_time=0.2,
            cfl=0.1,
            time_integrator=SSP3(),
            viscosity_model=EV1D(c_E=1.0, c_max=0.5),
        )
        mesh = Mesh1D(p)
        test_data = loadmat(f"test_data/mesh_1d/mesh_N{N}_K{K}.mat")
        test_fields(mesh, test_data)

    def test_all_mesh():
        for N in [1, 2, 3, 4]:
            for K in [1, 2, 3, 10, 16]:
                test_mesh(N, K)
        print("-----------------------------------")
        print("All mesh tests PASSED")
        print("-----------------------------------")

    assert bkd.BACKEND == bkd.NUMPY
    test_mesh_perturbation(0.5)
    test_all_mesh()
