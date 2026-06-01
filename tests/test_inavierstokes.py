import time
import jax
from pathlib import Path
import scipy.io
import numpy as np
import jax.numpy as jnp
from scipy.sparse.linalg import spsolve

from pyndg.mesh import read_mesh
from pyndg.ops.meshops import MeshOps
from pyndg.physics import ins
from pyndg.physics.ins import _SPLITTING_COEFFS, IncompressibleNavierStokes
from pyndg.mesh.bc import BC

PATH = "/home/matteo/Documents/nodal-dg/Codes1.1/"


@jax.jit
def u_time_fn(time):
    return jnp.sin(jnp.pi * time / 8)


@jax.jit
def du_time_fn(time):
    return (jnp.pi / 8) * jnp.cos(jnp.pi * time / 8)


@jax.jit
def p_time_fn(time):
    return (jnp.pi / 8) * jnp.cos(jnp.pi * time / 8)


@jax.jit
def u_bc(xyz, nxyz, maps):
    map_in = maps[11]
    y_in = xyz[1].reshape(-1)[map_in] + 0.20
    ux = (1 / 0.41) ** 2 * 6 * y_in * (0.41 - y_in)
    u = jnp.zeros((xyz.shape[0], xyz.shape[1] * xyz.shape[2]))
    u = u.at[0, map_in].set(ux)
    return u.reshape(xyz.shape)


@jax.jit
def p_bc(xyz, nxyz, maps):
    p = jnp.zeros(xyz.shape[1:])
    return p


@jax.jit
def dudn_bc(xyz, nxyz, maps):
    map_in = maps[11]
    y_in = xyz[1].reshape(-1)[map_in] + 0.20
    dudn = jnp.zeros(xyz.shape[1] * xyz.shape[2])
    dudn = dudn.at[map_in].set(-((1 / 0.41) ** 2) * 6 * y_in * (0.41 - y_in))
    return dudn.reshape(xyz.shape[1:])


@jax.jit
def u_ic(xyz):
    u = jnp.zeros_like(xyz)
    return u


@jax.jit
def p_ic(xyz):
    p = jnp.zeros(xyz.shape[1:])
    return p


def test_2d():
    run_test(N=1, tol=1e-14)
    run_test(N=2, tol=1e-13)
    run_test(N=3, tol=1e-12)
    run_test(N=4, tol=1e-11)


def run_test(N, tol):
    home_path = Path(__file__).resolve().parent.parent
    mesh_path = home_path / "mesh" / "gambit" / "cylinderA00075b.neu"
    data_path = home_path / "tests" / "data" / f"INS2D_N{N}_ts20.mat"

    mesh = read_mesh(mesh_path)

    mesh_ops = MeshOps(mesh, N)
    params = {
        "final_time": 8.0,
        "penalty": 20.0,
        "nu": 0.001,
        "u_ic": u_ic,
        "p_ic": p_ic,
        "u_bc": u_bc,
        "p_bc": p_bc,
        "dudn_bc": dudn_bc,
        "u_time_scale": u_time_fn,
        "du_time_scale": du_time_fn,
        "p_time_scale": p_time_fn,
        "bc_tags": {
            11: BC.Dirichlet,
            12: BC.Neumann,
            13: BC.Dirichlet,
            14: BC.Dirichlet,
        },
    }
    problem = IncompressibleNavierStokes(mesh_ops, params)

    def pressure_solver(p_rhs):
        p_sol = spsolve(problem.pr_sys, p_rhs.flatten(order="F"))
        return p_sol.reshape(p_rhs.shape, order="F")

    def velocity_solver_0(rhs):
        p_sol = spsolve(problem.adv_sys_0, rhs.flatten(order="F"))
        return p_sol.reshape(rhs.shape, order="F")

    def velocity_solver_1(rhs):
        p_sol = spsolve(problem.adv_sys_1, rhs.flatten(order="F"))
        return p_sol.reshape(rhs.shape, order="F")

    problem.nsteps = 20
    state = problem.run(pressure_solver, velocity_solver_0, velocity_solver_1)

    state_ref = scipy.io.loadmat(data_path)

    np.testing.assert_allclose(state_ref["Ux"], state.u[0], atol=tol)
    np.testing.assert_allclose(state_ref["Uy"], state.u[1], atol=tol)
    np.testing.assert_allclose(state_ref["NUx"], state.Nu[0], atol=tol)
    np.testing.assert_allclose(state_ref["NUy"], state.Nu[1], atol=tol)
    np.testing.assert_allclose(state_ref["dpdn"], state.dpdn, atol=tol * 100)
