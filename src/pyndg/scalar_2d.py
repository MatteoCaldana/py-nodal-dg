# -*- coding: utf-8 -*-
import numpy as np

from .scalar_param_2d import ScalarParam2D
from .mesh_2d import Mesh2D
from .backend import bkd as bk

from .time_integrator import LS54
from .physic_model import KPP2D, Advection2D
from .bc_2d import BC
from .limiters import Limiter
from .viscosity_model import bound_and_smooth_viscosity_2d, NoViscosity


class Scalar2D:
    def __init__(self, params, manual_reset=False):
        assert isinstance(params, ScalarParam2D)
        self.params = params
        self.mesh = Mesh2D(params)
        if not manual_reset:
            self.reset()

    def reset(self, state=None):
        if not self.mesh.is_init:
            self.mesh.initialize()
        if state is None:
            self.u = self.params.u_IC(self.mesh.x, self.mesh.y)
            self.u_old = bk.zeros_like(self.u)
            self.u_old_old = bk.zeros_like(self.u)
            self.mu_vals = bk.zeros_like(self.u)

            self.time = bk.zeros((1,))
            self.dt = self.params.dt
            self.tstep = 0

            self.done = False
            self.max_courant = -1
        else:
            for field in ["u", "u_old", "time", "dt", "tstep", "done"]:
                self.__dict__[field] = getattr(state, field)

    def step(self):
        if self.time - self.params.final_time >= 0:
            print("WARNING: stepping after final time")

        self.eig = self.params.model.eig(self.u)
        self.max_courant = max(
            self.max_courant, bk.max(self.eig) * self.dt / self.mesh.min_hK
        )
        if not isinstance(self.params.viscosity_model, NoViscosity):
            self.mu_raw = self.params.viscosity_model.compute(
                self.u,
                self.u_old,
                self.dt,
                self.tstep,
                self.mesh,
                self.params.model,
                self.eig,
            )
            self.mu_vals = bound_and_smooth_viscosity_2d(self.mu_raw, self.mesh)
            mu_max = bk.max(bk.abs(self.mu_vals))

        if self.params.dt <= 0:
            if self.params.dt_fixed:
                self.dt = self.params.cfl / (self.mesh.N2 / self.mesh.min_hK)
            else:
                self.dt = self.params.cfl * (
                    1
                    / (
                        bk.max(self.eig) * self.mesh.N2 / self.mesh.min_dx / 2
                        + mu_max * (self.mesh.N2 / self.mesh.min_dx / 2) ** 2
                    )
                )

        if self.time + self.dt >= self.params.final_time:
            self.dt = self.params.final_time - self.time
            assert self.dt >= 0
            self.done = True

        self.u_old_old = self.u_old
        self.u_old, self.u = (
            self.u,
            self.params.time_integrator.integrate(self.u, self.dt, self.__rhs_weak),
        )

        self.time = self.time + self.dt
        self.tstep += 1
        return not self.done

    def __calculate_uPM(self, u):
        uflat = u.reshape((-1,))
        uP = uflat[self.mesh.vmapP_C].reshape(self.mesh.vmapP.shape)
        uM = uflat[self.mesh.vmapM_C].reshape(self.mesh.vmapM.shape)
        return uP, uM, (uP + uM) / 2

    def __calculate_bd_flux_comp(self, uPM2):
        return (
            bk.matmul(self.mesh.LIFT, self.mesh.Fscale * (uPM2 * self.mesh.nx)),
            bk.matmul(self.mesh.LIFT, self.mesh.Fscale * (uPM2 * self.mesh.ny)),
        )

    def __calculate_int_flux_comp(self, u):
        Drwu, Dswu = bk.matmul(self.mesh.Drw, u), bk.matmul(self.mesh.Dsw, u)
        return (
            self.mesh.rx * Drwu + self.mesh.sx * Dswu,
            self.mesh.ry * Drwu + self.mesh.sy * Dswu,
        )

    def __calculate_gs(self, qX, qY):
        gX, gY = self.mu_vals * qX, self.mu_vals * qY
        gXf, gYf = gX.reshape((-1,)), gY.reshape((-1,))
        gXPM = (gXf[self.mesh.vmapP_C] + gXf[self.mesh.vmapM_C]).reshape(
            self.mesh.vmapP.shape
        )
        gYPM = (gYf[self.mesh.vmapP_C] + gYf[self.mesh.vmapM_C]).reshape(
            self.mesh.vmapP.shape
        )
        return gX, gY, gXPM, gYPM

    def __calculate_dGdF(self, u):
        F, G = self.params.model.flux(u)
        dFdr = bk.matmul(self.mesh.Drw, F)
        dFds = bk.matmul(self.mesh.Dsw, F)
        dGdr = bk.matmul(self.mesh.Drw, G)
        dGds = bk.matmul(self.mesh.Dsw, G)
        return dFdr, dFds, dGdr, dGds

    def __calculate_int_flux(self, dFdr, dFds, dGdr, dGds, gX, gY):
        return (
            (self.mesh.rx * dFdr + self.mesh.sx * dFds)
            + (self.mesh.ry * dGdr + self.mesh.sy * dGds)
            - (
                self.mesh.rx * bk.matmul(self.mesh.Drw, gX)
                + self.mesh.sx * bk.matmul(self.mesh.Dsw, gX)
                + self.mesh.ry * bk.matmul(self.mesh.Drw, gY)
                + self.mesh.sy * bk.matmul(self.mesh.Dsw, gY)
            )
        )

    def __calculate_flux(self, uM, uP):
        return self.params.model.flux(uM), self.params.model.flux(uP)

    def __calculate_lambda(self, uM, uP):
        lam = bk.maximum(self.params.model.eig(uM), self.params.model.eig(uP))
        if len(lam.shape) > 1:
            raise NotImplementedError("Not tested w.r.t. Matlab")
            lam = lam.reshape((self.mesh.Nfp, 3 * self.mesh.K))
            lam = bk.outer(bk.ones((self.mesh.Nfp,)), bk.max(lam, axis=0))
            lam = lam.reshape((self.mesh.Nfp * 3, self.mesh.K))
        return lam

    def __calculate_bd_flux(self, fP, fM, gP, gM, uM, uP, gXPM, gYPM, lam):
        flux = (
            self.mesh.nx * (fP + fM)
            + self.mesh.ny * (gP + gM)
            + lam * (uM - uP)
            - (gXPM * self.mesh.nx + gYPM * self.mesh.ny)
        )
        return bk.matmul(self.mesh.LIFT, self.mesh.Fscale * flux / 2)

    def __rhs_weak(self, u):
        # assert self.mesh.vmapM.shape == (self.mesh.Nfp * 3, self.mesh.K)
        # assert self.mesh.vmapP.shape == (self.mesh.Nfp * 3, self.mesh.K)
        # assert len(self.mesh.mapBC_list) == 0
        uP, uM, uPM2 = self.__calculate_uPM(u)
        bd_flux_X, bd_flux_Y = self.__calculate_bd_flux_comp(uPM2)
        int_flux_X, int_flux_Y = self.__calculate_int_flux_comp(u)
        qX, qY = bd_flux_X - int_flux_X, bd_flux_Y - int_flux_Y
        gX, gY, gXPM, gYPM = self.__calculate_gs(qX, qY)
        dFdr, dFds, dGdr, dGds = self.__calculate_dGdF(u)
        int_flux = self.__calculate_int_flux(dFdr, dFds, dGdr, dGds, gX, gY)
        (fM, gM), (fP, gP) = self.__calculate_flux(uM, uP)
        lam = self.__calculate_lambda(uM, uP)
        bd_flux = self.__calculate_bd_flux(fP, fM, gP, gM, uM, uP, gXPM, gYPM, lam)
        return int_flux - bd_flux


def kppic(x, y):
    return 3.5 * np.pi * (x * x + y * y < 1.0) + 0.25 * np.pi * (x * x + y * y >= 1)


def aic(x, y):
    p1 = 2.0 * (x > 0.2) * (x < 0.8) * (y > 0.2) * (y < 0.8)
    p2 = 1.0 * (x > 1.0) * (x < 1.5) * (y > 1.0) * (y < 1.5)
    return p1 + p2


def get_problem(viscosity_model, m, n, k, c, u, sv):
    if m == "RotatingWave":
        params = ScalarParam2D(
            model=KPP2D(),
            name="Test",
            N=int(n),
            mesh_file=((-2.5, -2.5), (2.5, 2.5), (int(k),) * 2),
            u_IC=kppic,
            bc={int(k + 1e5): BC.Periodic for k in [1, 2, 3, 4]},
            final_time=1.0,
            cfl=float(c),
            time_integrator=LS54(),
            viscosity_model=viscosity_model,
            limiter=Limiter(),
        )
    elif m == "Advection":
        params = ScalarParam2D(
            model=Advection2D(),
            name="Test",
            N=int(n),
            mesh_file=((0, 0), (2.0, 2.0), (int(k),) * 2),
            u_IC=aic,
            bc={int(k + 1e5): BC.Periodic for k in [1, 2, 3, 4]},
            final_time=1.0,
            cfl=float(c),
            time_integrator=LS54(),
            viscosity_model=viscosity_model,
            limiter=Limiter(),
        )
    else:
        raise NotImplementedError(f"{m}")
    return Scalar2D(params)


def linear_advection_solution_wrapper(func):
    return lambda x, y, t: func(x - t, y - t)


def get_solution(_, m):
    if m == "Advection":
        return linear_advection_solution_wrapper(aic)
    else:
        return None