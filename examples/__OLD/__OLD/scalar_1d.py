# -*- coding: utf-8 -*-
import numpy as np

from . import backend as bkd
from .backend import bkd as bk

from .scalar_param_1d import ScalarParam1D
from .mesh_1d import Mesh1D
from .bc_1d import apply_BC_1D, inv_BC
from .viscosity_model import (
    bound_and_smooth_viscosity,
    lerp_1d_visc,
    ExternalViscosity1D,
)


def centered_flux(u, v, flux):
    return 0.5 * (flux(u) + flux(v))


def lax_friedrich_flux(u, v, flux, dflux):
    return 0.5 * (
        flux(u) + flux(v) - bk.maximum(bk.abs(dflux(u)), bk.abs(dflux(v))) * (v - u)
    )


class Scalar1D:
    def __init__(self, params, manual_reset=False):
        assert isinstance(params, ScalarParam1D)
        self.params = params
        self.mesh = Mesh1D(params)
        if not manual_reset:
            self.reset()

    def reset(self, state=None):
        if state is None:
            self.u = self.params.u_IC(self.mesh.x)
            self.u_old = bk.zeros_like(self.u)
            self.u_old_old = bk.zeros_like(self.u)
            self.mu_vals = bk.zeros_like(self.u)

            self.time = bk.zeros((1,))
            self.dt = bkd.to_const(np.array(self.params.dt))
            self.iter = 0

            self.done = False
            self.max_courant = -1
        else:
            for field in ["u", "u_old", "time", "dt", "iter", "done"]:
                self.__dict__[field] = getattr(state, field)

    def step(self):
        if self.time - self.params.final_time > 0:
            print(
                f"WARNING: stepping after final time {self.params.final_time} at {self.time}."
            )

        local_wave_sp = bkd.maxval(bk.abs(self.params.model.dflux(self.u)), axis=0)
        self.max_courant = max(
            self.max_courant, local_wave_sp.max() * self.dt / self.mesh.min_hK
        )
        self.mu_raw = self.params.viscosity_model.compute(
            self.u,
            self.u_old,
            self.params.cfl / (self.mesh.N2 / self.mesh.min_hK),
            self.iter,
            self.mesh,
            self.params.model,
            local_wave_sp,
        )
        assert (
            (len(self.mu_raw.shape) == 1) and (self.mu_raw.shape[0] == self.u.shape[1])
        ) or (
            (len(self.mu_raw.shape) == 2) and (self.mu_raw.shape[1] == self.u.shape[1])
        )
        if self.mu_raw.shape == self.u.shape:
            self.mu_vals = self.mu_raw
        else:
            if (
                not isinstance(self.params.viscosity_model, ExternalViscosity1D)
                or self.params.smooth_viscosity
            ):
                self.mu_vals = bound_and_smooth_viscosity(self.mu_raw, self.mesh)
            elif len(self.mu_raw.shape) == 1 or self.mu_raw.shape[0] == 1:
                self.mu_vals = bk.tile(self.mu_raw, (self.u.shape[0], 1))
            elif self.mu_raw.shape[0] == 2:
                self.mu_vals = lerp_1d_visc(self.mu_raw, self.mesh)
            else:
                raise ValueError("Invalid mu_raw")
            assert self.mu_vals.shape == self.u.shape
        mu_max = bk.max(bk.abs(self.mu_vals))

        if self.params.cfl > 0:
            if self.params.dt_fixed:
                self.dt = self.params.cfl / (self.mesh.N2 / self.mesh.min_hK)
            else:
                self.speed = bk.max(bk.abs(self.params.model.dflux(self.u)))
                self.dt = self.params.cfl / (
                    self.speed * self.mesh.N2 / self.mesh.min_hK
                    + mu_max * self.mesh.N2**2 / self.mesh.min_hK**2
                )

        if self.time + self.dt >= self.params.final_time:
            self.dt = self.params.final_time - self.time
            self.done = True

        self.u_old_old = self.u_old
        self.u_old, self.u = (
            self.u,
            self.params.time_integrator.integrate(self.u, self.dt, self._rhs_weak),
        )

        self.time = self.time + self.dt
        if self.time - self.params.final_time >= 0:
            self.done = True
        self.iter += 1
        return not self.done

    def _rhs_weak(self, u):
        u_ext = apply_BC_1D(
            bk.reshape(bk.reshape(u, (-1,))[self.mesh.VtoE0], self.mesh.VtoE.shape),
            self.params.bc,
        )
        fluxr_u = centered_flux(
            u_ext[1, 1 : self.mesh.K + 1],
            u_ext[0, 2 : self.mesh.K + 2],
            lambda x: x,
        )
        fluxl_u = centered_flux(
            u_ext[1, : self.mesh.K], u_ext[0, 1 : self.mesh.K + 1], lambda x: x
        )

        qh = -bk.matmul(self.mesh.ST, u) + (
            bk.matmul(
                self.mesh.Imat[:, self.mesh.N : self.mesh.N + 1],
                fluxr_u.reshape((1, -1)),
            )
            - bk.matmul(self.mesh.Imat[:, 0:1], fluxl_u.reshape((1, -1)))
        )
        q = self.mu_vals * self.mesh.int_metric * bk.matmul(self.mesh.invM, qh)

        bc_cond_aux = (
            (inv_BC(self.params.bc[0][0]), 0.0),
            (inv_BC(self.params.bc[1][0]), 0.0),
        )

        q_ext = apply_BC_1D(
            bk.reshape(bk.reshape(q, (-1,))[self.mesh.VtoE0], self.mesh.VtoE.shape),
            bc_cond_aux,
        )

        fluxr_q = centered_flux(
            q_ext[1, 1 : self.mesh.K + 1],
            q_ext[0, 2 : self.mesh.K + 2],
            lambda x: x,
        )
        fluxl_q = centered_flux(
            q_ext[1, : self.mesh.K], q_ext[0, 1 : self.mesh.K + 1], lambda x: x
        )
        fluxr_fu = lax_friedrich_flux(
            u_ext[1, 1 : self.mesh.K + 1],
            u_ext[0, 2 : self.mesh.K + 2],
            self.params.model.flux,
            self.params.model.dflux,
        )
        fluxl_fu = lax_friedrich_flux(
            u_ext[1, : self.mesh.K],
            u_ext[0, 1 : self.mesh.K + 1],
            self.params.model.flux,
            self.params.model.dflux,
        )

        rhsu = (
            bk.matmul(self.mesh.ST, self.params.model.flux(u))
            - bk.matmul(self.mesh.ST, q)
            + (
                bk.matmul(
                    self.mesh.Imat[:, self.mesh.Np - 1 : self.mesh.Np],
                    fluxr_q.reshape((1, -1)),
                )
                - bk.matmul(self.mesh.Imat[:, 0:1], fluxl_q.reshape((1, -1)))
            )
            - (
                bk.matmul(
                    self.mesh.Imat[:, self.mesh.Np - 1 : self.mesh.Np],
                    fluxr_fu.reshape((1, -1)),
                )
                - bk.matmul(self.mesh.Imat[:, 0:1], fluxl_fu.reshape((1, -1)))
            )
        )
        rhsu = self.mesh.int_metric * bk.matmul(self.mesh.invM, rhsu)
        return rhsu


###############################################################################


def rect(x):
    return bk.where((x > 0.25) & (x < 0.75), 1.0, 0.0)


def t1(x):
    return (
        bk.where((x > 0.05) & (x < 0.125), 1.0, 0.0)
        + bk.where((x > 0.25) & (x < 0.5), 2 * x - 0.5, 0.0)
        + bk.where((x > 0.5) & (x < 0.75), 2 * bk.sqrt(0.0625 - (x - 0.5) ** 2), 0.0)
        + 0.5
    )


def t2(x):
    return -bk.exp(-400.0 * (x - 0.5) ** 2)


def t3(x):
    return bk.where((x > 0.0) & (x < 0.5), 20.0 * (0.25 - bk.abs(x - 0.25)), 0.0)


def t3_burgers_sol(x, t):
    return bk.where(
        t < 0.05,
        bk.where(x < 0.25 + 5 * t, 20.0 * x / (20.0 * t + 1), 0.0)
        + bk.where(
            (x >= 0.25 + 5 * t) & (x < 0.5), (10.0 - 20.0 * x) / (1 - 20.0 * t), 0.0
        ),
        bk.where(x < bk.sqrt(20.0 * t + 1.0) / 2**1.5, 20.0 * x / (20.0 * t + 1), 0.0),
    )


def t4(x):
    return (
        bk.where(x < 0.2, 10.0, 0.0)
        + bk.where((x >= 0.2) & (x < 0.4), 6.0, 0.0)
        + bk.where((x > 0.6) & (x < 1.0), -4.0, 0.0)
    )


def t5(x):
    return bk.where((x > 0.25) & (x < 0.75), 1.0, 0.0)


def t5_burgers_sol(x, t):
    return bk.where(
        t < 1.0,
        bk.where((x > 0.25) & (x < t + 0.25), (x - 0.25) / t, 0.0)
        + bk.where((x >= t + 0.25) & (x < t / 2 + 0.75), 1.0, 0.0),
        bk.where(
            (x > 0.25) & (x < (3.0 * bk.sqrt(t) + 1.0) / 4.0), (x - 0.25) / t, 0.0
        ),
    )


def t6(x):
    return bk.sin(2 * np.pi * x)


def t6_burgers_sol(x, t):
    u = t6(x)
    for _ in range(50):  # fixed point iterations
        u = t6(x - u * t)
    return u


def t7(x):
    return bk.where((x > 0.25) & (x < 0.5), bk.sin(4 * np.pi * x), 0.0) + bk.where(
        (x > 0.5) & (x < 0.75), bk.sin(8 * np.pi * x), 0.0
    )


def t8(x):
    return bk.where((x > 1 / 6) & (x < 5 / 6), -bk.sin(6 * np.pi * x), 0.0)


def t9(x):
    return (
        bk.where((x > 1 / 6) & (x < 1 / 3), 6 * (x - 1 / 6), 0.0)
        + bk.where((x > 1 / 3) & (x < 2 / 3), 6 * (x - 0.5), 0.0)
        + bk.where((x > 2 / 3) & (x < 5 / 6), 6 * (x - 5 / 6), 0.0)
    )


def t10(x):
    return bk.where((x > 0.25) & (x < 0.75), 16 * bk.abs(x - 0.5) - 2.0, 2.0)


def t10_burgers_sol(x, t):
    return bk.where(
        t < 0.0625,
        bk.where(
            (x > 0.25 + t * 2.0) & (x < 0.5 - 2.0 * t),
            (6.0 - 16.0 * x) / (1 - 16.0 * t),
            0.0,
        )
        + bk.where(
            (x >= 0.5 - 2.0 * t) & (x < 0.75 + t * 2.0),
            (16.0 * x - 10) / (1 + 16.0 * t),
            0.0,
        )
        + bk.where((x <= 0.25 + t * 2.0) | (x >= 0.75 + t * 2.0), 2.0, 0.0),
        bk.where(x < -bk.sqrt(16.0 * t + 1) / 2**1.5 + 2.0 * t + 0.75, 2.0, 0.0)
        + bk.where(
            x >= -bk.sqrt(16.0 * t + 1) / 2**1.5 + 2.0 * t + 0.75,
            (16.0 * x - 10) / (1 + 16.0 * t),
            0.0,
        ),
    )


def t11(x):
    return (
        bk.where((x > 0.25) & (x < 0.5), 1.0, 0.0)
        + bk.where((x >= 0.5) & (x < 0.75), 3.0, 0.0)
        - 1.0
    )


def t12(x):
    return bk.where(x < 0.5, 3.857143, 1 + 0.2 * bk.sin(5 * x))


def t13(x):
    return 0.125 + bk.where(x < 0.0, 0.875, 0.0)


def t14(x):
    x = x % 1.0
    return (
        bk.where((x > 0) & (x < 1 / 6), 6 * x, 0.0)
        + bk.where((x > 1 / 6) & (x < 1 / 3), 6 * (x - 1 / 3), 0.0)
        + bk.where((x > 1 / 3) & (x < 1 / 2), 2.0, 0.0)
        + bk.where((x > 1 / 2) & (x < 3 / 4), -0.5, 0.0)
    )


def t15(x):
    return (
        bk.where((x < 0.3) | (0.7 <= x), bk.sin(4 * np.pi * x), 0)
        + bk.where((0.3 < x) & (x <= 0.4), 3, 0)
        + bk.where((0.4 < x) & (x <= 0.5), 1, 0)
        + bk.where((0.5 < x) & (x <= 0.6), 3, 0)
        + bk.where((0.6 < x) & (x <= 0.7), 2, 0)
    )


def linear_advection_solution_wrapper(func):
    return lambda x, t: func(x - t)


IC_TABLE = {
    "Rect": {
        "ic": rect,
        "Advection": linear_advection_solution_wrapper(rect),
    },
    "T1": {
        "ic": t1,
        "Advection": linear_advection_solution_wrapper(t1),
        "Burgers": None,
    },
    "T2": {
        "ic": t2,
        "Advection": linear_advection_solution_wrapper(t2),
        "Burgers": None,
    },
    "T3": {
        "ic": t3,
        "Advection": linear_advection_solution_wrapper(t3),
        "Burgers": t3_burgers_sol,
    },
    "T4": {
        "ic": t4,
        "Advection": linear_advection_solution_wrapper(t4),
        "Burgers": None,
    },
    "T5": {
        "ic": t5,
        "Advection": linear_advection_solution_wrapper(t5),
        "Burgers": t5_burgers_sol,
    },
    "T6": {
        "ic": t6,
        "Advection": linear_advection_solution_wrapper(t6),
        "Burgers": t6_burgers_sol,
    },
    "T7": {
        "ic": t7,
        "Advection": linear_advection_solution_wrapper(t7),
        "Burgers": None,
    },
    "T8": {
        "ic": t8,
        "Advection": linear_advection_solution_wrapper(t8),
        "Burgers": None,
    },
    "T9": {
        "ic": t9,
        "Advection": linear_advection_solution_wrapper(t9),
        "Burgers": None,
    },
    "T10": {
        "ic": t10,
        "Advection": linear_advection_solution_wrapper(t10),
        "Burgers": t10_burgers_sol,
    },
    "T11": {
        "ic": t11,
        "Advection": linear_advection_solution_wrapper(t11),
        "Burgers": None,
    },
    "T12": {
        "ic": t12,
        "Advection": linear_advection_solution_wrapper(t12),
        "Burgers": None,
    },
    "T13": {
        "ic": t13,
        "Advection": linear_advection_solution_wrapper(t13),
        "Burgers": None,
    },
    "T14": {
        "ic": t14,
        "Advection": linear_advection_solution_wrapper(t14),
        "Burgers": None,
    },
    "T15": {
        "ic": t15,
        "Advection": linear_advection_solution_wrapper(t15),
        "Burgers": None,
    },
}
