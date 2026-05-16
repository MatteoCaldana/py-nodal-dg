from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from typing import NamedTuple
from scipy.sparse.linalg import spsolve_triangular

from pyndg.physics.poisson import Poisson
from pyndg.mesh.bc import BC
from pyndg.ops.meshops import div, grad, curl

# Splitting coefficients
_g0: float = +1.5
_a0: float = +2.0
_a1: float = -0.5
_b0: float = +2.0
_b1: float = -1.0


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
    bc_u, bc_p = params["bc_fn"](mesh_ops.fxyz, mesh_ops.nxyz, mesh_ops.bc_maps)

    pressure_pb = Poisson(
        mesh_ops, {"penalty": params["penalty"], "bc_tags": params["pressure_bc_tags"]}
    )
    pressure_pb.assemble()

    velocity_pb = Poisson(
        mesh_ops, {"penalty": params["penalty"], "bc_tags": params["velocity_bc_tags"]}
    )
    velocity_pb.assemble()


def step(state):
    state = _update_bc(state)
    state = _advection_step(state)
    state = _pressure_step(state)
    state = _viscous_step(state)
    return state._replace(
        time=(state.timestep + 1) * state.dt,
        timestep=state.timestep + 1,
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
def _advection_step(state, mesh_ops, bc_type_map, mesh_dims):
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
        (_a0 * state.u + _a1 * state.u_old) - state.dt * (_b0 * Nu + _b1 * state.Nu)
    ) / _g0

    return u_tilde, Nu


def _pressure_step(state, mesh_ops, mesh_dims, u_tilde, Nu, bc_type_map, ss):
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
    term_vol = mesh_ops.J[None, :] * (-div_u_tilde * _g0 / state.dt)
    sJ = mesh_ops.sJ.reshape((-1, mesh_dims.K))
    term_sur = mesh_ops.lift @ (sJ * (_b0 * dpdn + _b1 * state.dpdn))
    p_rhs = mesh_ops.int_phiphi @ (term_vol + term_sur)

    # Add Dirichlet boundary forcing
    p_rhs += state.rhs_bc_p

    # Pressure Solve
    p_rhs = p_rhs.flatten(order="F")[ss.PRperm]
    tmp = spsolve_triangular(ss.PRsystemC.T, p_rhs, lower=True)
    p_sol = spsolve_triangular(ss.PRsystemC, tmp, lower=False)
    tmp[ss.PRperm] = p_sol
    p_new = tmp.reshape((mesh_dims.Np, mesh_dims.K), order="F")

    # Compute (U~~, V~~) = (U~, V~) - dt*grad PR
    dp = grad(mesh_ops.J_rst_xyz, mesh_ops.Dphi, p_new)

    # Increment to (Ux~~, Uy~~)
    uTT = u_tilde - state.dt * dp / _g0
    return dpdn, uTT, p_new


def _viscous_step(state, mesh_ops, mesh_dims, uTT, ss):
    mmUTT = mesh_ops.J * (mesh_ops.int_phiphi @ uTT)

    # Formulate the full RHS for the Helmholtz system
    u_rhs = (_g0 * mmUTT) / (state.nu * state.dt) + state.rhs_bc_u

    # Solve system
    u_new = []
    for d in range(mesh_dims.dim):
        u_rhs_flat = u_rhs[d].flatten(order="F")[ss.VELperm]
        tmp = spsolve_triangular(ss.VELsystemC.T, u_rhs_flat, lower=True)
        u_sol = spsolve_triangular(ss.VELsystemC, tmp, lower=False)
        tmp[ss.VELperm] = u_sol
        u_new.append(tmp.reshape((mesh_dims.Np, mesh_dims.K), order="F"))

    return jnp.stack(u_new)


class IncompressibleNavierStokes:
    def __init__(self, mesh_ops, params):
        self.mesh_ops = mesh_ops
        self.params = params

        self.mesh = mesh_ops.mesh
        self.N = mesh_ops.N

        self.dt = np.min(self.mesh.inradius) / (self.N + 1) / (self.N + 1)
        self.nsteps = int(params["final_time"] / self.dt)
        self.dt = params["final_time"] / self.nsteps
        self.time = 0.0

        self.nu = params["nu"]

        self.u, self.p = params["ic_fn"](mesh_ops.xyz)
