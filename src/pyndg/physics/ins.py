from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from typing import NamedTuple

from pyndg.physics.poisson import Poisson
from pyndg.mesh.bc import BC, invert_dir_neu_bc
from pyndg.ops.meshops import div, grad, curl, apply_bc_maps
import pyndg.backend as bkd


# Splitting coefficients
class SplittingCoeffs(NamedTuple):
    g0: float
    a0: float
    a1: float
    b0: float
    b1: float


_SPLITTING_COEFFS = {
    "stage0": SplittingCoeffs(g0=1.0, a0=1.0, a1=0.0, b0=1.0, b1=0.0),
    "stage1": SplittingCoeffs(g0=1.5, a0=2.0, a1=-0.5, b0=2.0, b1=-1.0),
}


class IncNavierStokesState(NamedTuple):
    u_old: jax.Array
    u: jax.Array
    Nu: jax.Array
    bc_dudn: jax.Array
    rhs_bc_u: jax.Array
    bc_u: jax.Array

    p: jax.Array
    dpdn: jax.Array
    bc_p: jax.Array
    rhs_bc_p: jax.Array

    nu: jax.Array
    time: jax.Array
    dt: jax.Array
    timestep: int

    ref_bc_u: jax.Array
    ref_rhs_bc_u: jax.Array
    ref_bc_p: jax.Array
    ref_rhs_bc_p: jax.Array
    ref_bc_dudn: jax.Array


def _setup_pressure_and_velocity(mesh_ops, params):
    penalty = params["penalty"]
    p_bc_tags = invert_dir_neu_bc(params["bc_tags"])

    zero_fn = lambda x: jnp.zeros_like(x[0])

    pressure_pb = Poisson(mesh_ops, {"penalty": penalty, "bc_tags": p_bc_tags})
    pressure_pb.assemble()
    pr_rhs = pressure_pb.assemble_rhs(zero_fn, params["p_bc"])

    velocity_pb = Poisson(mesh_ops, {"penalty": penalty, "bc_tags": params["bc_tags"]})
    velocity_pb.assemble()
    vel_rhs = []
    for d in range(mesh_ops.mesh.dim):
        bc_fn = lambda *args: params["u_bc"](*args)[d]
        vel_rhs.append(velocity_pb.assemble_rhs(zero_fn, bc_fn, force_assemble=True))
    vel_rhs = jnp.stack(vel_rhs)

    mass = velocity_pb.mass_mat
    pr_sys = pressure_pb.stiff_mat
    vel_sys = velocity_pb.stiff_mat

    return mass, pr_sys, pr_rhs, vel_sys, vel_rhs


def step(
    state,
    mesh_data,
    mesh_dims,
    bc_type_map,
    u_time_fn,
    du_time_fn,
    p_time_fn,
    pressure_solver,
    velocity_solver,
    coeffs,
):
    state = _update_bc(state, u_time_fn, du_time_fn, p_time_fn)
    u_tilde, Nu = _advection_step(state, mesh_data, bc_type_map, mesh_dims, coeffs)
    dpdn, uTT, p_new = _pressure_step(
        state,
        mesh_data,
        mesh_dims,
        u_tilde,
        Nu,
        bc_type_map,
        pressure_solver,
        coeffs,
    )
    u_new = _viscous_step(state, mesh_data, mesh_dims, uTT, velocity_solver, coeffs)
    return state._replace(
        u_old=state.u,
        u=u_new,
        p=p_new,
        Nu=Nu,
        time=state.timestep * state.dt,
        timestep=state.timestep + 1,
        dpdn=dpdn,
    )


def _update_bc(state, u_time_fn, du_time_fn, p_time_fn):
    tfu = u_time_fn(state.time)
    tfu_new = u_time_fn(state.time + state.dt)
    tfp = p_time_fn(state.time)
    tfp_new = p_time_fn(state.time + state.dt)
    tfdu = du_time_fn(state.time)
    return state._replace(
        bc_u=state.ref_bc_u * tfu,
        rhs_bc_u=state.ref_rhs_bc_u * tfu_new,
        bc_p=state.ref_bc_p * tfp,
        rhs_bc_p=state.ref_rhs_bc_p * tfp_new,
        bc_dudn=state.ref_bc_dudn * tfdu,
    )


# @partial(jax.jit, static_argnames=["Nfp", "Nfaces"])
def _advection_step(state, mesh_ops, bc_type_map, mesh_dims, coeffs):
    # evaluate flux vector
    F = state.u[:, None] * state.u[None, :]
    # evaluate nonlinear term
    Nu = div(mesh_ops.J_rst_xyz, mesh_ops.Dphi, F)
    # get traces
    uM = state.u.reshape(mesh_dims.dim, -1)[:, mesh_ops.vmap_m]
    uP = state.u.reshape(mesh_dims.dim, -1)[:, mesh_ops.vmap_p]

    # apply BCs
    d_mask = bc_type_map[BC.Dirichlet]
    uP = uP.at[:, d_mask].set(state.bc_u[:, d_mask])

    FP = uP[:, None] * uP[None, :]
    FM = uM[:, None] * uM[None, :]

    # Normal velocity and Lax-Friedrichs/Rusonov flux
    unM = jnp.einsum("dij,dij->ij", uM, mesh_ops.nxyz)
    unP = jnp.einsum("dij,dij->ij", uP, mesh_ops.nxyz)
    maxvel = jnp.maximum(jnp.abs(unM), jnp.abs(unP))

    # Evaluate maximum normal velocity over each face
    maxvel = maxvel.reshape(mesh_dims.Nf, mesh_dims.Nfp, -1)
    maxvel = jnp.max(maxvel, axis=1, keepdims=True)
    maxvel = jnp.repeat(maxvel, mesh_dims.Nfp, axis=1)
    maxvel = maxvel.reshape(mesh_dims.Nf * mesh_dims.Nfp, -1)

    # Form Fluxes
    fluxu = -0.5 * (
        jnp.einsum("dij,dlij->lij", mesh_ops.nxyz, FM - FP) + maxvel * (uP - uM)
    )

    # Combine volume and surface terms
    Nu += mesh_ops.lift @ (mesh_ops.fscale[None, ...] * fluxu)

    # Compute intermediate velocity tilde
    u_tilde = (
        (coeffs.a0 * state.u + coeffs.a1 * state.u_old)
        - state.dt * (coeffs.b0 * Nu + coeffs.b1 * state.Nu)
    ) / coeffs.g0

    return u_tilde, Nu


def _pressure_step(
    state, mesh_ops, mesh_dims, u_tilde, Nu, bc_type_map, solver, coeffs
):
    div_u_tilde = div(mesh_ops.J_rst_xyz, mesh_ops.Dphi, u_tilde)

    # Compute dp/dn components
    curl_u = curl(mesh_ops.J_rst_xyz, mesh_ops.Dphi, state.u)
    curl_curl_u = curl(mesh_ops.J_rst_xyz, mesh_ops.Dphi, curl_u)
    res = Nu + state.nu * curl_curl_u

    # On Neumann nodes (Dirichlet on u):
    # dpdn(nbcmapD) = - n \dot (Du/Dt + nu curl curl u)
    d_mask = bc_type_map[BC.Dirichlet]
    dpdn = np.zeros_like(state.dpdn)
    res_on_face = res.reshape(mesh_dims.dim, -1)[:, mesh_ops.vmap_m[d_mask]]
    dpdn[d_mask] = -jnp.einsum("di,di->i", mesh_ops.nxyz[:, d_mask], res_on_face)

    # Update and subtract boundary forcing
    dpdn -= state.bc_dudn

    # Evaluate RHS for Pressure Poisson Equation
    term_vol = mesh_ops.J[None, :] * (-div_u_tilde * coeffs.g0 / state.dt)
    sJ = mesh_ops.sJ.reshape((-1, mesh_dims.K))
    term_sur = mesh_ops.lift @ (sJ * (coeffs.b0 * dpdn + coeffs.b1 * state.dpdn))
    p_rhs = mesh_ops.int_phiphi @ (term_vol + term_sur)

    # Add Dirichlet boundary forcing
    p_rhs += state.rhs_bc_p

    # Pressure Solve
    p_new = solver(p_rhs)

    # Compute (U~~, V~~) = (U~, V~) - dt*grad PR
    dp = grad(mesh_ops.J_rst_xyz, mesh_ops.Dphi, p_new)

    # Increment to (Ux~~, Uy~~)
    uTT = u_tilde - state.dt * dp / coeffs.g0
    return dpdn, uTT, p_new


def _viscous_step(state, mesh_ops, mesh_dims, uTT, solver, coeffs):
    mmUTT = mesh_ops.J * (mesh_ops.int_phiphi @ uTT)

    # Formulate the full RHS for the Helmholtz system
    u_rhs = (coeffs.g0 * mmUTT) / (state.nu * state.dt) + state.rhs_bc_u

    # Solve system
    u_new = [solver(u_rhs[d]) for d in range(mesh_dims.dim)]
    return jnp.stack(u_new)


class IncompressibleNavierStokes:
    def __init__(self, mesh_ops, params):
        self.mesh_ops = mesh_ops
        self.params = params

        self.mesh = mesh_ops.mesh
        self.N = mesh_ops.N

        self.dt = np.min(self.mesh.inradius) / (self.N + 1) / (self.N + 1)
        self.nsteps = int(np.ceil(params["final_time"] / self.dt))
        self.dt = params["final_time"] / self.nsteps
        self.time = 0.0

        self.nu = params["nu"]

        self.mesh_dims, self.mesh_data = mesh_ops.build_mesh_data()

        self.u = params["u_ic"](self.mesh_data.xyz)
        self.p = params["p_ic"](self.mesh_data.xyz)
        self.u_bc = params["u_bc"](
            self.mesh_data.fxyz, self.mesh_data.nxyz, self.mesh_data.bc_maps
        )
        self.p_bc = params["p_bc"](
            self.mesh_data.fxyz, self.mesh_data.nxyz, self.mesh_data.bc_maps
        )
        self.dudn_bc = params["dudn_bc"](
            self.mesh_data.fxyz, self.mesh_data.nxyz, self.mesh_data.bc_maps
        )

        self.mass, self.pr_sys, self.pr_rhs, self.vel_sys, self.vel_rhs = (
            _setup_pressure_and_velocity(mesh_ops, params)
        )

        _g0 = _SPLITTING_COEFFS["stage0"].g0
        self.adv_sys_0 = _g0 * self.mass / (self.dt * self.nu) + self.vel_sys
        _g0 = _SPLITTING_COEFFS["stage1"].g0
        self.adv_sys_1 = _g0 * self.mass / (self.dt * self.nu) + self.vel_sys

        self.bc_type_map = apply_bc_maps(self.mesh_ops, self.params["bc_tags"])
        self.bc_type_map = {
            key: jnp.array(val) for key, val in self.bc_type_map.items()
        }

        self.mesh_data = self.mesh_data._replace(
            vmap_m=self.mesh_data.vmap_m.reshape(-1, mesh_ops.K),
            vmap_p=self.mesh_data.vmap_p.reshape(-1, mesh_ops.K),
        )

    def _build_initial_state(self):
        return IncNavierStokesState(
            u_old=jnp.zeros_like(self.u, dtype=bkd.jnp_prec),
            u=jnp.array(self.u, dtype=bkd.jnp_prec),
            p=jnp.zeros_like(self.u[0], dtype=bkd.jnp_prec),
            dpdn=jnp.zeros_like(self.mesh_data.fxyz[0], dtype=bkd.jnp_prec),
            Nu=jnp.zeros_like(self.u, dtype=bkd.jnp_prec),
            nu=jnp.array(self.nu, dtype=bkd.jnp_prec),
            time=jnp.array(self.time, dtype=bkd.jnp_prec),
            dt=jnp.array(self.dt, dtype=bkd.jnp_prec),
            timestep=1,
            # will be filled from ref + time scaling functions
            bc_u=None,
            bc_dudn=None,
            bc_p=None,
            rhs_bc_u=None,
            rhs_bc_p=None,
            #
            ref_rhs_bc_u=jnp.array(self.vel_rhs, dtype=bkd.jnp_prec),
            ref_rhs_bc_p=jnp.array(self.pr_rhs, dtype=bkd.jnp_prec),
            ref_bc_u=jnp.array(self.u_bc, dtype=bkd.jnp_prec),
            ref_bc_p=jnp.array(self.p_bc, dtype=bkd.jnp_prec),
            ref_bc_dudn=jnp.array(self.dudn_bc, dtype=bkd.jnp_prec),
        )

    def run(self):
        state = self._build_initial_state()

        state = step(
            state,
            self.mesh_data,
            self.mesh_dims,
            self.bc_type_map,
            self.params["u_time_scale"],
            self.params["du_time_scale"],
            self.params["p_time_scale"],
            pressure_solver,
            velocity_solver,
            _SPLITTING_COEFFS["stage0"],
        )

        while state.timestep < self.nsteps:
            state = step(
                state,
                self.mesh_data,
                self.mesh_dims,
                self.bc_type_map,
                self.params["u_time_scale"],
                self.params["du_time_scale"],
                self.params["p_time_scale"],
                pressure_solver,
                velocity_solver,
                _SPLITTING_COEFFS["stage1"],
            )

        return state
