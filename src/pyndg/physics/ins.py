import numpy as np
import jax
import jax.numpy as jnp
from typing import NamedTuple

from pyndg.physics.poisson import Poisson


class SplittingCoeffs(NamedTuple):
    g0: float = 1.0
    a0: float = 1.0
    a1: float = 0.0
    b0: float = 1.0
    b1: float = 0.0


class IncNavierStokesState(NamedTuple):
    u: jax.Array
    p: jax.Array

    time: jax.Array
    dt: jax.Array
    timestep: int


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


def _update_bc(state):
    return state


def _advection_step(state, mesh_ops):
    # evaluate flux vector
    F = state.u[:, None] * state.u[None, :]
    # save old nonlinear term for use in pressure step
    nu_old = state.nu.copy()
    # evaluate nonlinear term
    nu = mesh_ops.divergence(F)

    return state


def _pressure_step(state):
    return state


def _viscous_step(state):
    return state


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
