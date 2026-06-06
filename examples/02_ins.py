import time
from sksparse.cholmod import cho_factor
import jax
from pathlib import Path
import scipy.io
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from pyndg.mesh import read_mesh
from pyndg.ops.meshops import MeshOps
from pyndg.physics import ins
from pyndg.physics.ins import _SPLITTING_COEFFS, IncompressibleNavierStokes
from pyndg.mesh.bc import BC
from pyndg.utils.plot import plot_2d

PATH = "/home/matteo/Documents/nodal-dg/Codes1.1/"


class State:
    def __init__(self, data):
        # Scalars
        self.time = data["time"].item()
        self.dt = data["dt"].item()
        self.tstep = data["tstep"].item()

        # State Vectors
        self.Ux = data["Ux"]
        self.Uy = data["Uy"]
        self.UxT = data["UxT"]
        self.UyT = data["UyT"]
        self.UxTT = data["UxTT"]
        self.UyTT = data["UyTT"]
        self.Uxold = data["Uxold"]
        self.Uyold = data["Uyold"]
        self.NUx = data["NUx"]
        self.NUy = data["NUy"]
        self.dpdn = data["dpdn"]


def load(file_path):
    data = scipy.io.loadmat(file_path)
    state = State(data)
    return state


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


def test(a, b):
    assert a.shape == b.shape, f"{a.shape} != {b.shape}"
    err = np.max(np.abs(a - b))
    print(f"Max difference: {err: .2e} | {err < 1e-10}")


def plot(axs, meshops, state):
    axs[0, 0].set_title("Velocity x")
    axs[0, 1].set_title("Velocity y")
    axs[1, 0].set_title("Velocity magnitude")
    axs[1, 1].set_title("Pressure")
    plot_2d(meshops, state.u[0], ax=axs[0, 0])
    plot_2d(meshops, state.u[1], ax=axs[0, 1])
    plot_2d(meshops, jnp.linalg.norm(state.u, axis=0), ax=axs[1, 0])
    plot_2d(meshops, state.p, ax=axs[1, 1])

    plt.pause(0.01)


if __name__ == "__main__":
    home_path = Path(__file__).resolve().parent.parent
    mesh_path = home_path / "mesh" / "gambit" / "cylinderA00075b.neu"

    mesh = read_mesh(mesh_path)
    # mesh.plot()

    N = 5

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
    Np, K = mesh_ops.Np, mesh_ops.K

    state0 = problem._build_initial_state()

    cho_factor_pr = cho_factor(problem.pr_sys.tocsc(), order="amd")
    cho_factor_vel_0 = cho_factor(problem.adv_sys_0.tocsc(), order="amd")
    cho_factor_vel_1 = cho_factor(problem.adv_sys_1.tocsc(), order="amd")

    def pressure_solver(p_rhs):
        p_rhs = p_rhs.flatten(order="F")
        p_sol = cho_factor_pr.solve(np.array(p_rhs))
        return p_sol.reshape((Np, K), order="F")

    def velocity_solver_0(rhs):
        rhs = rhs.flatten(order="F")
        sol = cho_factor_vel_0.solve(np.array(rhs))
        return sol.reshape((Np, K), order="F")

    def velocity_solver_1(rhs):
        rhs = rhs.flatten(order="F")
        sol = cho_factor_vel_1.solve(np.array(rhs))
        return sol.reshape((Np, K), order="F")

    state = ins.step(
        state0,
        problem.mesh_data,
        problem.mesh_dims,
        problem.bc_type_map,
        problem.params["u_time_scale"],
        problem.params["du_time_scale"],
        problem.params["p_time_scale"],
        pressure_solver,
        velocity_solver_0,
        _SPLITTING_COEFFS["stage0"],
    )

    print("=" * 70)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    plot(axs, mesh_ops, state)

    total_time = 0.0
    nsteps = 10000
    for step in range(2, nsteps):
        t0 = time.perf_counter()
        state = ins.step(
            state,
            problem.mesh_data,
            problem.mesh_dims,
            problem.bc_type_map,
            problem.params["u_time_scale"],
            problem.params["du_time_scale"],
            problem.params["p_time_scale"],
            pressure_solver,
            velocity_solver_1,
            _SPLITTING_COEFFS["stage1"],
        )
        state.u.block_until_ready()
        t1 = time.perf_counter()
        total_time += t1 - t0

        if step % 100 == 0:
            print(f"Step {step}/{nsteps} | Time: {state.time:.4f} seconds")
            plot(axs, mesh_ops, state)

    print(f"Total time: {total_time:.4f} seconds")
    print(f"Pressure solve time: {ins._pressure_solve_time:.4f} seconds")
    print(f"Velocity solve time: {ins._velocity_solve_time:.4f} seconds")
    non_solve_time = total_time - ins._pressure_solve_time - ins._velocity_solve_time
    print(f"Non-solve time: {non_solve_time:.4f} seconds")
