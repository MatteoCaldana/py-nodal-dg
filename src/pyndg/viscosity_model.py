# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import math

from .backend import bkd as bk
from . import backend as bkd

if bkd.BACKEND == bkd.TORCH:
    import torch as th

from .utils import check_type
from .bc_1d import BC, apply_BC_1D


def lerp_1d_visc(mu, mesh):
    xref = mesh.r.reshape((-1, 1))
    lambda0 = (1 - xref) / 2
    lambda1 = (1 + xref) / 2
    return mu[0, :] * lambda0 + mu[1, :] * lambda1


def smooth_viscosity(mu, mesh):
    mu = mu.reshape((-1,))
    o = bk.ones((mesh.x.shape[0], 1))
    mu_ext = bk.stack([mu[0], *mu, mu[-1]])
    mu_left = (mu_ext[:-2] + mu_ext[1:-1]).reshape((1, -1)) / 2
    mu_right = (mu_ext[1:-1] + mu_ext[2:]).reshape((1, -1)) / 2
    return bk.matmul(o, mu_left) + (mesh.x - bk.matmul(o, mesh.x[0:1, :])) / mesh.h * (
        bk.matmul(o, (mu_right - mu_left))
    )


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def smooth_viscosity_2d(mu, mesh):
    assert mu.numel() == np.max(mu.shape)
    mu_avg = (mesh.VToE @ mu.T).reshape((-1,)) / mesh.neighbors
    mu_int = mesh.interp_matrix @ mu_avg[mesh.EToV0T_flat]
    return reshape_fortran(mu_int, (mesh.Np, mesh.K))


def bound_and_smooth_viscosity_2d(mu, mesh):
    mu = bk.maximum(mu, bk.zeros_like(mu))
    mu = smooth_viscosity_2d(mu, mesh)
    return bk.maximum(mu, bk.zeros_like(mu))


def bound_and_smooth_viscosity(mu, mesh):
    mu = bk.maximum(mu, bk.zeros_like(mu))
    mu = smooth_viscosity(mu, mesh)
    return bk.maximum(mu, bk.zeros_like(mu))


class ViscosityModel(ABC):
    @abstractmethod
    def compute(self, u, uold, dt, i, mesh):
        pass


class ViscosityModel1D(ViscosityModel):
    pass


class ViscosityModel2D(ViscosityModel):
    @staticmethod
    @abstractmethod
    def need_interpolation():
        pass


class NoViscosity(ViscosityModel1D, ViscosityModel2D):
    @staticmethod
    def need_interpolation():
        return False

    def compute(self, u, uold, dt, i, mesh, model, local_wave_sp):
        return bk.zeros_like(model.entropy(u))


@dataclass(frozen=True)
class EV1D(ViscosityModel1D):
    c_E: float
    c_max: float

    def __post_init__(self):
        check_type(self)

        if self.c_E <= 0:
            raise ValueError("Field 'c_E' must be positive.")
        if self.c_max <= 0:
            raise ValueError("Field 'c_max' must be positive.")

    def compute(self, u, uold, dt, i, mesh, model, local_wave_sp):
        E_old, E_new = model.entropy(uold), model.entropy(u)
        F_old, F_new = model.entropy_flux(uold), model.entropy_flux(u)

        mu_max = self.c_max * mesh.hK / mesh.N * local_wave_sp

        R = (E_new - E_old) / dt + (
            bk.matmul(mesh.Dr, F_old) + bk.matmul(mesh.Dr, F_new)
        ) / mesh.hK
        if i == 0:
            R = bk.zeros_like(R)
        F_ext = apply_BC_1D(
            F_new.reshape((-1,))[mesh.VtoE0].reshape(mesh.VtoE.shape),
            ((BC.Neumann, 0.0), (BC.Neumann, 0.0)),
        )

        cL = F_ext[1, : mesh.K] - F_ext[0, 1 : mesh.K + 1]
        cR = F_ext[1, 1 : mesh.K + 1] - F_ext[0, 2 : mesh.K + 2]
        J = bk.maximum(bk.abs(cL), bk.abs(cR)) / (mesh.hK / mesh.N)

        E_modal = bk.matmul(mesh.invV, E_new)
        Norm_E = bk.max(bk.abs(E_new - bk.sum(E_modal[0, :]) / math.sqrt(2) / mesh.K))

        DD = bk.maximum(bkd.maxval(bk.abs(R), axis=0), bk.abs(J)) / Norm_E
        mu_E = self.c_E * (mesh.hK / mesh.N) ** 2 * DD
        mu_piece = bk.minimum(mu_E, mu_max)
        return mu_piece


@dataclass(frozen=True)
class MDH1D(ViscosityModel1D):
    c_A: float
    c_k: float
    c_max: float

    def __post_init__(self):
        check_type(self)

        if self.c_A <= 0:
            raise ValueError("Field 'c_A' must be positive.")
        if self.c_k <= 0:
            raise ValueError("Field 'c_k' must be positive.")
        if self.c_max <= 0:
            raise ValueError("Field 'c_max' must be positive.")

    def compute(self, u, uold, dt, i, mesh, model):
        local_wave_sp = np.max(np.abs(model.dflux(u)))
        mu_max = self.c_max * mesh.hK / mesh.N * local_wave_sp
        u_modal = np.matmul(mesh.invV, u)
        S_K = u_modal[mesh.N, :] ** 2.0 / np.sum(u_modal**2)
        s_K = np.log10(S_K)
        s0 = -(self.c_A + 4 * np.log10(mesh.N))
        mu1 = ((s0 - self.c_k) <= s_K) * (s_K <= (s0 + self.c_k))
        mu2 = s_K > (s0 + self.c_k)
        mu_piece = mu_max * (
            mu1 / 2 * (1 + np.sin(np.pi / (2 * self.c_k) * (s_K - s0))) + mu2
        )
        mu_piece = np.where(np.isnan(mu_piece), 0, mu_piece)
        return mu_piece


# TODO: MDA 1D
# TODO: test MDH 1D

###############################################################################


@dataclass(frozen=True)
class EV2D(ViscosityModel2D):
    c_E: float
    c_max: float

    def __post_init__(self):
        check_type(self)

        if self.c_E <= 0:
            raise ValueError("Field 'c_E' must be positive.")
        if self.c_max <= 0:
            raise ValueError("Field 'c_max' must be positive.")

    @staticmethod
    def need_interpolation():
        return True

    def compute(self, u, uold, dt, i, mesh, model, local_wave_sp):
        E_old, E_new = model.entropy(uold), model.entropy(u)
        F_old_X, F_old_Y = model.entropy_flux(uold)
        F_new_X, F_new_Y = model.entropy_flux(u)

        mu_max = 2.0 * self.c_max * mesh.dx2 * bk.max(local_wave_sp) / mesh.N

        divFold = (
            mesh.rx * bk.matmul(mesh.Dr, F_old_X)
            + mesh.sx * bk.matmul(mesh.Ds, F_old_X)
            + mesh.ry * bk.matmul(mesh.Dr, F_old_Y)
            + mesh.sy * bk.matmul(mesh.Ds, F_old_Y)
        )
        divFnew = (
            mesh.rx * bk.matmul(mesh.Dr, F_new_X)
            + mesh.sx * bk.matmul(mesh.Ds, F_new_X)
            + mesh.ry * bk.matmul(mesh.Dr, F_new_Y)
            + mesh.sy * bk.matmul(mesh.Ds, F_new_Y)
        )

        R = (E_new - E_old) / dt + (divFold + divFnew) / 2
        if i == 0:
            R = bk.zeros_like(R)

        FXP = F_new_X.reshape((-1,))[mesh.vmapP_C].reshape(mesh.vmapP.shape)
        FXM = F_new_X.reshape((-1,))[mesh.vmapM_C].reshape(mesh.vmapM.shape)
        FYP = F_new_Y.reshape((-1,))[mesh.vmapP_C].reshape(mesh.vmapP.shape)
        FYM = F_new_Y.reshape((-1,))[mesh.vmapM_C].reshape(mesh.vmapM.shape)

        js = bk.abs(
            reshape_fortran(FXP - FXM, mesh.nx.shape) * mesh.nx
            + reshape_fortran(FYP - FYM, mesh.ny.shape) * mesh.ny
        ) / (
            2
            * bk.matmul(bk.ones(((mesh.N + 1) * 3, 1)), mesh.dx2.reshape((1, -1)))
            / mesh.N
        )

        Jump = bkd.maxval(js, axis=0)

        E_modal = bk.matmul(mesh.invV, E_new)
        Norm_E = bk.abs(
            E_new
            - bk.sum(E_modal[0, :] * mesh.J[0, :]) / bk.sum(mesh.J[0, :]) / np.sqrt(2)
        ).max()

        DD = bk.maximum(bkd.maxval(bk.abs(R), axis=0), bk.abs(Jump)) / Norm_E
        mu_E = self.c_E * (2 * mesh.dx2 / mesh.N) ** 2 * DD
        mu_piece = bk.minimum(mu_E, mu_max)
        return mu_piece
