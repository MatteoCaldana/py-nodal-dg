# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from . import  backend as bkd
from .backend import bkd as bk

from .euler_param_1d import EulerParam1D
from .mesh_1d import Mesh1D
from .bc_1d import BC, apply_BC_1D, inv_BC
from .time_integrator import LS54
from .viscosity_model import bound_and_smooth_viscosity


class Euler1D:
    def __init__(self, params, manual_reset=False):
        assert isinstance(params, EulerParam1D)
        self.params = params
        self.mesh = Mesh1D(params)
        if not manual_reset:
            self.reset()

    def reset(self, state=None):
        if state is None:
            rho = self.params.rho_IC(self.mesh.x)
            vel = self.params.vel_IC(self.mesh.x)
            pre = self.params.pre_IC(self.mesh.x)

            mmt = rho * vel
            energy = 0.5 * rho * vel * vel + pre / (self.params.gas_gamma - 1)

            # Creating vector of conserved variables
            self.u = bk.stack([rho, mmt, energy])
            self.u_old = bk.zeros_like(self.u)
            self.u_old_old = bk.zeros_like(self.u)

            self.mu_vals = bk.zeros_like(self.u[0])

            self.time = bk.zeros((1,))
            if self.params.dt != -1:
                self.dt = bkd.to_const(np.array(self.params.dt))
            else:
                self.dt = bkd.to_const(np.array(0))
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

        pre = (self.params.gas_gamma - 1) * (
            self.u[2] - 0.5 * self.u[1] * self.u[1] / self.u[0]
        )
        c_sound = bk.sqrt(bk.abs(self.params.gas_gamma * pre / self.u[0]))
        local_wave_sp = bkd.maxval(c_sound + bk.abs(self.u[1] / self.u[0]), axis=0)
        self.max_courant = max(self.max_courant, local_wave_sp.max() * self.dt / self.mesh.min_hK)
        self.mu_raw = self.params.viscosity_model.compute(
            self.u,
            self.u_old,
            self.dt,
            self.iter,
            self.mesh,
            self.params.model,
            local_wave_sp,
        )

        if self.mu_raw.shape == self.u[0].shape:
            self.mu_vals = self.mu_raw
        else:
            self.mu_vals = bound_and_smooth_viscosity(self.mu_raw, self.mesh)
            assert self.mu_vals.shape == self.u[0].shape
        mu_max = bk.max(bk.abs(self.mu_vals))

        if self.params.cfl > 0:
            if self.params.dt_fixed:
                self.dt = self.params.cfl / (self.mesh.N2 / self.mesh.min_hK)
            else:
                pre = (self.params.gas_gamma - 1) * (
                    self.u[2] - 0.5 * self.u[1] * self.u[1] / self.u[0]
                )
                c_sound = bk.sqrt(bk.abs(self.params.gas_gamma * pre / self.u[0]))
                self.speed = bk.max(c_sound + bk.abs(self.u[1] / self.u[0]))
                self.dt = self.params.cfl / (
                    self.speed * self.mesh.N2 / self.mesh.min_hK
                    + mu_max * self.mesh.N2 ** 2 / self.mesh.min_hK ** 2
                )

        self.u_old_old = self.u_old
        self.u_old, self.u = (  # idiomatic swap
            self.u,
            self.params.time_integrator.integrate(self.u, self.dt, self._rhs_weak),
        )

        self.time = self.time + self.dt
        if self.time - self.params.final_time >= 0:
            self.done = True
        self.iter += 1
        return not self.done


    def __get_flux_q(self, i, u):
        u_ext = apply_BC_1D(
            bk.reshape(bk.reshape(u, (-1,))[self.mesh.VtoE0], self.mesh.VtoE.shape),
            self.params.bcs[i],
        )
        fluxr_u = self.__centered_flux(
            u_ext[1, 1 : self.mesh.K + 1], u_ext[0, 2 : self.mesh.K + 2]
        )
        fluxl_u = self.__centered_flux(
            u_ext[1, : self.mesh.K], u_ext[0, 1 : self.mesh.K + 1]
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
            (inv_BC(self.params.bcs[i][0][0]), 0.0),
            (inv_BC(self.params.bcs[i][1][0]), 0.0),
        )

        q_ext = apply_BC_1D(
            bk.reshape(bk.reshape(q, (-1,))[self.mesh.VtoE0], self.mesh.VtoE.shape),
            bc_cond_aux,
        )
        fluxr_q = self.__centered_flux(
            q_ext[1, 1 : self.mesh.K + 1], q_ext[0, 2 : self.mesh.K + 2]
        )
        fluxl_q = self.__centered_flux(
            q_ext[1, : self.mesh.K], q_ext[0, 1 : self.mesh.K + 1]
        )
        return [u_ext, q, fluxl_q, fluxr_q]

    def __build_rhs(self, fu, q, fluxl_q, fluxr_q, fluxl_fu, fluxr_fu):
        return self.mesh.int_metric * bk.matmul(
            self.mesh.invM,
            bk.matmul(self.mesh.ST, fu)
            - bk.matmul(self.mesh.ST, q)
            - (
                bk.matmul(
                    self.mesh.Imat[:, self.mesh.Np - 1 : self.mesh.Np],
                    fluxr_fu.reshape((1, -1)),
                )
                - bk.matmul(self.mesh.Imat[:, 0:1], fluxl_fu.reshape((1, -1)))
            )
            + (
                bk.matmul(
                    self.mesh.Imat[:, self.mesh.Np - 1 : self.mesh.Np],
                    fluxr_q.reshape((1, -1)),
                )
                - bk.matmul(self.mesh.Imat[:, 0:1], fluxl_q.reshape((1, -1)))
            ),
        )

    def __centered_flux(self, u, v):
        return 0.5 * (u + v)

    def __lax_frierdrich(self, u, v):
        # flux for u
        preu = (self.params.gas_gamma - 1) * (u[2] - 0.5 * u[1] * u[1] / u[0])
        velu = u[1] / u[0]
        fu = bk.stack([u[1], (preu + u[0] * velu * velu), (u[2] + preu) * velu])

        # flux for v
        prev = (self.params.gas_gamma - 1) * (v[2] - 0.5 * v[1] * v[1] / v[0])
        velv = v[1] / v[0]
        fv = bk.stack([v[1], (prev + v[0] * velv * velv), (v[2] + prev) * velv])

        # Maximum eigenvalue at face
        lamu = bk.sqrt(bk.abs(self.params.gas_gamma * preu / u[0])) + bk.abs(velu)
        lamv = bk.sqrt(bk.abs(self.params.gas_gamma * prev / v[0])) + bk.abs(velv)
        lam = bk.maximum(lamu, lamv)

        return 0.5 * (fu + fv - lam * (v - u))

    def _rhs_weak(self, u):
        vel = u[1] / u[0]
        pre = (self.params.gas_gamma - 1) * (u[2] - 0.5 * u[0] * vel * vel)

        u_q_flux = [self.__get_flux_q(i, u[i]) for i in range(3)]

        fuu = [u[1], pre + vel * u[1], (u[2] + pre) * vel]

        # build LF flux
        fluxr_fu = self.__lax_frierdrich(
            bk.stack([u_q_flux[i][0][1, 1 : self.mesh.K + 1] for i in range(3)]),
            bk.stack([u_q_flux[i][0][0, 2 : self.mesh.K + 2] for i in range(3)]),
        )
        fluxl_fu = self.__lax_frierdrich(
            bk.stack([u_q_flux[i][0][1, 0 : self.mesh.K + 0] for i in range(3)]),
            bk.stack([u_q_flux[i][0][0, 1 : self.mesh.K + 1] for i in range(3)]),
        )

        rhsu = [
            self.__build_rhs(fuu[i], *u_q_flux[i][1:], fluxl_fu[i], fluxr_fu[i])
            for i in range(3)
        ]

        return bk.stack(rhsu)


# rho, velocity, pressure Shu Osher


def rso(x):
    return bk.where(x < -4, 3.857143, 1 + 0.2 * bk.sin(5 * x))


def vso(x):
    return bk.where(x < -4, 2.629369, 0.0)


def pso(x):
    return bk.where(x < -4, 10.33333, 1.0)


# rho, velocity, pressure Sod Tube
def rst(x):
    return 0.125 + bk.where(x < 0.5, 0.875, 0.0)


def vst(x):
    return 0.0 * x


def pst(x):
    return 0.1 + bk.where(x < 0.5, 0.9, 0.0)


def get_problem(viscosity_model, m, n, k, c, u, sv):
    if u == "ShuOsher":
        rho_IC = rso
        vel_IC = vso
        pre_IC = pso
        bcs = [
            ((BC.Dirichlet, 3.857143), (BC.Neumann, 0.0)),
            ((BC.Dirichlet, 10.141852), (BC.Dirichlet, 0.0)),
            ((BC.Dirichlet, 39.166661), (BC.Neumann, 0.0)),
        ]
        bnd=(-5.0, 5.0)
    elif u == "SodTube":
        rho_IC = rst
        vel_IC = vst
        pre_IC = pst
        bcs = [
            ((BC.Dirichlet, 1.0), (BC.Dirichlet, 0.125)),
            ((BC.Dirichlet, 0.0), (BC.Dirichlet, 0.0)),
            ((BC.Dirichlet, 2.5), (BC.Dirichlet, 0.25)),
        ]
        bnd=(0.0, 1.0)
    else:
        raise NotImplementedError("")

    params = EulerParam1D(
        gas_const=1.0,
        gas_gamma=1.4,
        name="Test",
        rho_IC=rho_IC,
        vel_IC=vel_IC,
        pre_IC=pre_IC,
        N=int(n),
        K=int(k),
        bnd=bnd,
        mesh_pert=0.0,
        bcs=bcs,
        final_time=2.00,
        cfl=float(c),
        time_integrator=LS54(),
        viscosity_model=viscosity_model,
        dt_fixed=True,
    )
    return Euler1D(params)


def plot(p):
    u = p.u.clone().detach().clone().numpy()
    vel = u[1] / u[0]
    pre = (p.params.gas_gamma - 1) * (u[2] - 0.5 * u[0] * vel * vel)
    f, axs = plt.subplots(1, 3)
    x = p.mesh.x.clone().detach().clone().numpy()
    axs[0].plot(x, u[0])
    axs[1].plot(x, pre)
    axs[2].plot(x, vel)