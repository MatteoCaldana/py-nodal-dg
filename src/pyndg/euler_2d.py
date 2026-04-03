import numpy as np

from . import backend as bkd
from .backend import bkd as bk

from .euler_param_2d import EulerParam2D
from .mesh_2d import Mesh2D
from .bc_2d import BC
from .viscosity_model import bound_and_smooth_viscosity_2d
from .time_integrator import LS54
import functools


@functools.partial(bkd.jit, static_argnums=(3, 4))
def calculate_uPM(u, vmapP_C, vmapM_C, vmapP_shape, vmapM_shape):
    uflat = u.reshape((-1,))
    uP = uflat[vmapP_C].reshape(vmapP_shape)
    uM = uflat[vmapM_C].reshape(vmapM_shape)
    return uP, uM, (uP + uM) / 2


@bkd.jit
def calculate_bd_flux_comp(uPM2, LIFT, Fscale, nx, ny):
    return (
        bk.matmul(LIFT, Fscale * (uPM2 * nx)),
        bk.matmul(LIFT, Fscale * (uPM2 * ny)),
    )


@bkd.jit
def calculate_int_flux_comp(u, Drw, Dsw, rx, sx, ry, sy):
    Drwu, Dswu = bk.matmul(Drw, u), bk.matmul(Dsw, u)
    return (
        rx * Drwu + sx * Dswu,
        ry * Drwu + sy * Dswu,
    )


@functools.partial(bkd.jit, static_argnums=(5,))
def calculate_gs(qX, qY, mu_vals, vmapP_C, vmapM_C, vmapP_shape):
    gX, gY = mu_vals * qX, mu_vals * qY
    gXf, gYf = gX.reshape((-1,)), gY.reshape((-1,))

    gXPM = (gXf[vmapP_C] + gXf[vmapM_C]).reshape(vmapP_shape)
    gYPM = (gYf[vmapP_C] + gYf[vmapM_C]).reshape(vmapP_shape)

    return gX, gY, gXPM, gYPM


@bkd.jit
def calculate_dGdF(F, G, Drw, Dsw):
    return (
        bk.matmul(Drw, F),
        bk.matmul(Dsw, F),
        bk.matmul(Drw, G),
        bk.matmul(Dsw, G),
    )


@bkd.jit
def calculate_int_flux(dFdr, dFds, dGdr, dGds, gX, gY, Drw, Dsw, rx, sx, ry, sy):
    return (
        (rx * dFdr + sx * dFds)
        + (ry * dGdr + sy * dGds)
        - (
            rx * bk.matmul(Drw, gX)
            + sx * bk.matmul(Dsw, gX)
            + ry * bk.matmul(Drw, gY)
            + sy * bk.matmul(Dsw, gY)
        )
    )


@bkd.jit
def calculate_bd_flux(fP, fM, gP, gM, uM, uP, gXPM, gYPM, lam, LIFT, Fscale, nx, ny):
    flux = nx * (fP + fM) + ny * (gP + gM) + lam * (uM - uP) - (gXPM * nx + gYPM * ny)
    return bk.matmul(LIFT, Fscale * flux / 2)


@functools.partial(bkd.jit, static_argnums=(0,))
def compute_flux(model, u):
    return model.flux(u)


class Euler2D:
    def __init__(self, params, manual_reset=False):
        assert isinstance(params, EulerParam2D)
        self.params = params
        self.mesh = Mesh2D(params)
        if not manual_reset:
            self.reset()

    def reset(self, state=None):
        if not self.mesh.is_init:
            self.mesh.initialize()
        for bc_key in self.params.bc:
            if self.params.bc[bc_key] != BC.Periodic:
                raise NotImplementedError()
        if state is None:
            self.u = self.params.uIC(
                self.mesh.x, self.mesh.y, self.params.gas_gamma, self.params.gas_const
            )
            self.u_old = bk.zeros_like(self.u)
            self.u_old_old = bk.zeros_like(self.u)

            self.mu_vals = bk.zeros_like(self.u[0])

            self.time = bkd.to_const(np.array(0))
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

    def _wave_speed(self):
        pre = self.params.model.pre(self.u)
        c_sound = bk.sqrt(bk.abs(self.params.gas_gamma * pre / self.u[0]))
        v_norm = bk.sqrt(self.u[1] * self.u[1] + self.u[2] * self.u[2]) / bk.abs(
            self.u[0]
        )
        return bkd.maxval(c_sound + v_norm, axis=0)

    def step(self):
        if self.time - self.params.final_time > 0:
            print("WARNING: stepping after final time.")

        self.wave_speed = self._wave_speed()
        self.max_courant = max(
            self.max_courant, self.wave_speed.max() * self.dt / self.mesh.min_hK
        )
        self.mu_raw = self.params.viscosity_model.compute(
            self.u,
            self.u_old,
            self.dt,
            self.iter,
            self.mesh,
            self.params.model,
            self.wave_speed,
        )

        if self.mu_raw.shape == self.u[0].shape:
            self.mu_vals = self.mu_raw
        else:
            self.mu_vals = bound_and_smooth_viscosity_2d(self.mu_raw, self.mesh)
            assert self.mu_vals.shape == self.u[0].shape
        mu_max = bk.max(bk.abs(self.mu_vals))

        if self.params.cfl > 0:
            if self.params.dt_fixed:
                self.dt = self.params.cfl / (self.mesh.N2 / self.mesh.min_hK)
            else:
                self.dt = self.params.cfl / (
                    self._wave_speed() * self.mesh.N2 / self.mesh.min_hK
                    + mu_max * self.mesh.N2**2 / self.mesh.min_hK**2
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

    def _rhs_weak(self, u):
        # TODO: implement non periodic BC

        # Solution traces
        uPuMuPM2 = [
            calculate_uPM(
                u[i],
                self.mesh.vmapP_C,
                self.mesh.vmapM_C,
                self.mesh.vmapP.shape,
                self.mesh.vmapM.shape,
            )
            for i in range(4)
        ]
        # Compute numerical fluxes of auxiliary equation
        bd_flux = [
            calculate_bd_flux_comp(
                uPuMuPM2[i][2],
                self.mesh.LIFT,
                self.mesh.Fscale,
                self.mesh.nx,
                self.mesh.ny,
            )
            for i in range(4)
        ]
        # Compute internal fluxes for auxiliary variable
        int_flux_c = [
            calculate_int_flux_comp(
                u[i],
                self.mesh.Drw,
                self.mesh.Dsw,
                self.mesh.rx,
                self.mesh.sx,
                self.mesh.ry,
                self.mesh.sy,
            )
            for i in range(4)
        ]
        # Compute auxiliary variables with viscosity and variable traces
        q = [[bd_flux[i][j] - int_flux_c[i][j] for j in range(2)] for i in range(4)]
        gXgYgXPMgYPM = [
            calculate_gs(
                q[i][0],
                q[i][1],
                self.mu_vals,
                self.mesh.vmapP_C,
                self.mesh.vmapM_C,
                self.mesh.vmapP.shape,
            )
            for i in range(4)
        ]
        # Compute volume contributions
        _, _, _, _, Fu, Gu = compute_flux(self.params.model, u)
        # Compute weak derivatives
        dFdrdFdsdGdrdGds = [
            calculate_dGdF(Fu[i], Gu[i], self.mesh.Drw, self.mesh.Dsw) for i in range(4)
        ]
        int_flux = [
            calculate_int_flux(
                *dFdrdFdsdGdrdGds[i],
                *gXgYgXPMgYPM[i][:2],
                self.mesh.Drw,
                self.mesh.Dsw,
                self.mesh.rx,
                self.mesh.sx,
                self.mesh.ry,
                self.mesh.sy,
            )
            for i in range(4)
        ]
        # Evaluate primitive variables & flux functions
        rhoP, vuP, vvP, pP, FuP, GuP = compute_flux(
            self.params.model, [uPuMuPM2[i][0] for i in range(4)]
        )
        rhoM, vuM, vvM, pM, FuM, GuM = compute_flux(
            self.params.model, [uPuMuPM2[i][1] for i in range(4)]
        )
        # Compute maximum wave speed on edges
        lam = bk.maximum(
            bk.sqrt(vuM * vuM + vvM * vvM)
            + bk.sqrt(bk.abs(self.params.gas_gamma * pM / rhoM)),
            bk.sqrt(vuP * vuP + vvP * vvP)
            + bk.sqrt(bk.abs(self.params.gas_gamma * pP / rhoP)),
        )

        max_lam_edges = []
        for i in range(lam.shape[0] // self.mesh.Nfp):
            max_lam_edge = bkd.maxval(
                lam[i * self.mesh.Nfp : (i + 1) * self.mesh.Nfp, :],
                axis=0,
            )
            for _ in range(self.mesh.Nfp):
                max_lam_edges.append(max_lam_edge)
        lam = bk.vstack(max_lam_edges)

        bd_flux = [
            calculate_bd_flux(
                FuP[i],
                FuM[i],
                GuP[i],
                GuM[i],
                uPuMuPM2[i][1],
                uPuMuPM2[i][0],
                gXgYgXPMgYPM[i][2],
                gXgYgXPMgYPM[i][3],
                lam,
                self.mesh.LIFT,
                self.mesh.Fscale,
                self.mesh.nx,
                self.mesh.ny,
            )
            for i in range(4)
        ]
        rhsu = bk.stack([int_flux[i] - bd_flux[i] for i in range(4)])
        return rhsu


def RPX_IC(p1, rho1, u1, v1, p2, rho2, u2, v2, p3, rho3, u3, v3, p4, rho4, u4, v4):
    def _RPX_IC(x, y, gas_gamma, gas_const):
        xc, yc = 0, 0
        pre = (
            p1 * (x > xc) * (y > yc)
            + p2 * (x <= xc) * (y > yc)
            + p3 * (x <= xc) * (y <= yc)
            + p4 * (x > xc) * (y <= yc)
        )
        u = (
            u1 * (x > xc) * (y > yc)
            + u2 * (x <= xc) * (y > yc)
            + u3 * (x <= xc) * (y <= yc)
            + u4 * (x > xc) * (y <= yc)
        )
        v = (
            v1 * (x > xc) * (y > yc)
            + v2 * (x <= xc) * (y > yc)
            + v3 * (x <= xc) * (y <= yc)
            + v4 * (x > xc) * (y <= yc)
        )
        rho = (
            rho1 * (x > xc) * (y > yc)
            + rho2 * (x <= xc) * (y > yc)
            + rho3 * (x <= xc) * (y <= yc)
            + rho4 * (x > xc) * (y <= yc)
        )
        e = 0.5 * rho * (u * u + v * v) + pre / (gas_gamma - 1)
        return bk.stack([rho, rho * u, rho * v, e])

    return _RPX_IC


RP4_IC = RPX_IC(
    p1=1.1,
    rho1=1.1,
    u1=0,
    v1=0,
    p2=0.35,
    rho2=0.5065,
    u2=0.8939,
    v2=0,
    p3=1.1,
    rho3=1.1,
    u3=0.8939,
    v3=0.8939,
    p4=0.35,
    rho4=0.5065,
    u4=0,
    v4=0.8939,
)

RP12_IC = RPX_IC(
    p1=0.4,
    rho1=0.5313,
    u1=0,
    v1=0,
    p2=1.0,
    rho2=1.0,
    u2=0.7276,
    v2=0,
    p3=1.0,
    rho3=0.8,
    u3=0.0,
    v3=0.0,
    p4=1.0,
    rho4=1.0,
    u4=0,
    v4=0.7276,
)


def shckVtx_IC(x, y, gas_gamma, gas_const):
    rho_L = 1.0
    u_L = gas_gamma**0.5
    v_L = 0
    p_L = 1.0
    p_R = 1.3
    rho_R = (
        rho_L
        * ((gas_gamma + 1) * p_R + gas_gamma - 1)
        / ((gas_gamma - 1) * p_R + gas_gamma + 1)
    )
    u_R = (
        gas_gamma**0.5
        + 2**0.5 * (1 - p_R) / (gas_gamma - 1 + p_R * (gas_gamma + 1)) ** 0.5
    )
    v_R = 0.0

    eps = 0.3
    rc = 0.05
    beta = 0.204
    xc = 0.25
    yc = 0.5
    r2 = ((x - xc) * (x - xc) + (y - yc) * (y - yc)) / rc / rc
    phir = bk.exp(beta * (1 - r2))
    delu = eps * (y - yc) / rc * phir
    delv = -eps * (x - xc) / rc * phir
    delT = -(gas_gamma - 1) * eps * eps / (4 * beta * gas_gamma) * phir * phir

    T_L_mod = p_L / gas_const / rho_L + delT
    rho_L_mod = (gas_const * T_L_mod) ** (1 / (gas_gamma - 1))
    p_L_mod = rho_L_mod**gas_gamma

    u = (u_L + delu) * (x < 0.5) + u_R * (x >= 0.5)
    v = (v_L + delv) * (x < 0.5) + v_R * (x >= 0.5)
    rho = (rho_L_mod) * (x < 0.5) + rho_R * (x >= 0.5)
    pre = (p_L_mod) * (x < 0.5) + p_R * (x >= 0.5)
    e = 0.5 * rho * (u * u + v * v) + pre / (gas_gamma - 1)
    return bk.stack([rho, rho * u, rho * v, e])


def get_problem(vm, m, n, k, cfl, uic_str, __):
    assert m == "Euler"
    if uic_str == "RP4":
        T = 0.25
        msh = ((-1.5, -1.5), (1.5, 1.5), (int(k),) * 2)
        uic = RP4_IC
    elif uic_str == "RP12":
        T = 0.3
        msh = ((-1.5, -1.5), (1.5, 1.5), (int(k),) * 2)
        uic = RP12_IC
    elif uic_str == "ShockVtx":
        T = 0.8
        msh = ((-1, 0), (3, 1), (int(k),) * 2)
        uic = shckVtx_IC
    else:
        raise NotImplementedError(f"uIC: {uic}")
    return Euler2D(
        EulerParam2D(
            gas_const=1.0,
            gas_gamma=1.4,
            name=uic_str,
            uIC=uic,
            N=n,
            mesh_file=msh,
            bc={int(k + 1e5): BC.Periodic for k in [1, 2, 3, 4]},
            final_time=T,
            cfl=cfl,
            dt_fixed=True,
            time_integrator=LS54(),
            viscosity_model=vm,
        )
    )
