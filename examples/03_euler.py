from functools import partial
from pathlib import Path
import jax
import jax.numpy as jnp
import scipy
import numpy as np
import matplotlib.pyplot as plt

from pyndg.mesh import read_mesh
from pyndg.ops.meshops import MeshOps, apply_bc_maps
from pyndg.physics.euler import (
    _c2p,
    _dt,
    _rhs,
    EulerVarCons,
    _wave_speed,
    _wave_speed,
    enforce_positivity,
    step,
)
from pyndg.mesh.bc import BC
from pyndg.utils.plot import plot_2d


@jax.jit
def iso_vortex_sol_fn(xy, time):
    # Purpose: compute flow configuration given by
    #     Y.C. Zhou, G.W. Wei / Journal of Computational Physics 189 (2003) 159
    x = xy[0]
    y = xy[1]

    # based flow parameters
    xo = 5
    yo = 0
    beta = 5
    gamma = 1.4
    u = 1
    v = 0

    xmut = x - u * time
    ymvt = y - v * time
    r = jnp.sqrt((xmut - xo) ** 2 + (ymvt - yo) ** 2)

    # perturbed density
    u = u - beta * jnp.exp(1 - r**2) * (ymvt - yo) / (2 * jnp.pi)
    v = v + beta * jnp.exp(1 - r**2) * (xmut - xo) / (2 * jnp.pi)
    rho1 = (
        1
        - (
            (gamma - 1)
            * beta**2
            * jnp.exp(2 * (1 - r**2))
            / (16 * gamma * jnp.pi * jnp.pi)
        )
    ) ** (1 / (gamma - 1))
    p1 = rho1**gamma

    Q1 = rho1
    Q2 = rho1 * u
    Q3 = rho1 * v
    Q4 = p1 / (gamma - 1) + 0.5 * rho1 * (u**2 + v**2)
    return EulerVarCons(Q1, jnp.array([Q2, Q3]), Q4)


@partial(jax.jit, static_argnames=["mdims"])
def iso_vortex_bc_fn(mdata, mdims, var, time):
    bc_map = mdata.bc_maps[13]
    xy = mdata.fxyz.reshape(mdata.fxyz.shape[0], -1)[:, bc_map]
    sol_eval = iso_vortex_sol_fn(xy, time)

    def _apply_map(var, var_eval):
        return (
            var.reshape(*var.shape[:-3], -1)
            .at[..., bc_map]
            .set(var_eval.squeeze())
            .reshape(var.shape)
        )

    return jax.tree.map(_apply_map, var, sol_eval)


def _mat2py(var):
    return EulerVarCons(
        rho=jnp.array(var[:, :, 0]),
        rhou=jnp.array(var[:, :, 1:3].transpose(2, 0, 1)),
        E=jnp.array(var[:, :, 3]),
    )


def test(a, b):
    assert a.shape == b.shape, f"{a.shape} != {b.shape}"
    err = np.max(np.abs(a - b))
    magn = np.max(np.abs(b) + np.abs(a))
    print(f"Maximum error: {err:.2e} | {err < 1e-10} | rerr: {err / magn:.2e}")
    if a.min() == a.max():
        print("WARNING: constants detected, error may be misleading")


def test_iso_vortex(N):
    home_path = Path(__file__).resolve().parent.parent
    mesh_path = home_path / "mesh" / "gambit" / "vortexA04.neu"

    mesh = read_mesh(mesh_path)
    # mesh.plot()

    matlab_path = "/home/matteo/Documents/nodal-dg/Codes1.1/"
    matlab_data = scipy.io.loadmat(matlab_path + f"EulerRHS2D_N{N}_init.mat")

    q_matlab = _mat2py(matlab_data["Q"])

    meshops = MeshOps(mesh, N=N)
    mdims, mdata = meshops.build_mesh_data()

    print("time:", matlab_data["time"])

    rhs = _rhs(
        mdata,
        mdims,
        q_matlab,
        time=jnp.array(matlab_data["time"]),
        bd_flux_type="LF",
        gamma=1.4,
        bc_fn=iso_vortex_bc_fn,
    )

    K = mdims.K
    test(rhs.rho.reshape((-1, K)), matlab_data["rhsQ"][:, :, 0])
    test(rhs.rhou[0].reshape((-1, K)), matlab_data["rhsQ"][:, :, 1])
    test(rhs.rhou[1].reshape((-1, K)), matlab_data["rhsQ"][:, :, 2])
    test(rhs.E.reshape((-1, K)), matlab_data["rhsQ"][:, :, 3])


@partial(jax.jit, static_argnames=["mdims"])
def fw_step_bc_fn(mdata, mdims, var, time):
    gamma = 1.4
    rhoin = gamma
    uin = 3.0
    vin = 0.0
    pin = 1.0
    Ein = pin / (gamma - 1.0) + 0.5 * rhoin * (uin**2 + vin**2)

    def _rfw(var):
        return var.reshape(*var.shape[:-3], -1)

    # Inflow conditions -- uniform inflow
    map_inlet = mdata.bc_maps[11]
    rho = _rfw(var.rho).at[map_inlet].set(rhoin)
    rhou = _rfw(var.rhou)
    rhou = rhou.at[0, map_inlet].set(rhoin * uin)
    rhou = rhou.at[1, map_inlet].set(rhoin * vin)
    E = _rfw(var.E).at[map_inlet].set(Ein)

    # Outflow conditions -- supersonic outflow ( do nothing )

    # Wall conditions -- reflective, isothermal, i.e., n.u=0, T=T(t=0)
    # reverse flow in normal direction in ghost elements
    map_wall = mdata.bc_maps[13]
    rhou_w = rhou[:, map_wall]
    nxyz = mdata.nxyz.reshape((mdims.dim, -1))[:, map_wall]
    tmp = rhou_w - 2 * nxyz * jnp.sum(nxyz * rhou_w, axis=0)
    rhou = rhou.at[:, map_wall].set(tmp)

    return EulerVarCons(
        rho=rho.reshape(var.rho.shape),
        rhou=rhou.reshape(var.rhou.shape),
        E=E.reshape(var.E.shape),
    )


def fw_step_ic(xy):
    gamma = 1.4
    rhoin = gamma
    uin = 3.0
    vin = 0.0
    pin = 1.0
    Ein = pin / (gamma - 1.0) + 0.5 * rhoin * (uin**2 + vin**2)

    return EulerVarCons(
        rho=rhoin * jnp.ones(xy.shape[1:]),
        rhou=jnp.array([rhoin * uin, rhoin * vin])[:, None, None] * jnp.ones(xy.shape),
        E=Ein * jnp.ones(xy.shape[1:]),
    )


def test_fw_step(N, bd_flux_type):
    home_path = Path(__file__).resolve().parent.parent
    mesh_path = home_path / "mesh" / "gambit" / "fstepA001.neu"

    mesh = read_mesh(mesh_path)
    # mesh.plot()

    matlab_path = "/home/matteo/Documents/nodal-dg/Codes1.1/"
    matlab_data = scipy.io.loadmat(
        matlab_path + f"ShockEulerRHS2D_N{N}_init_{bd_flux_type}.mat"
    )

    q_matlab = _mat2py(matlab_data["Q"])

    meshops = MeshOps(mesh, N=N)
    mdims, mdata = meshops.build_mesh_data()

    print("time:", matlab_data["time"])

    rhs = _rhs(
        mdata,
        mdims,
        q_matlab,
        time=matlab_data["time"],
        bd_flux_type=bd_flux_type,
        gamma=1.4,
        bc_fn=fw_step_bc_fn,
    )

    K = mdims.K
    test(rhs.rho.reshape((-1, K)), matlab_data["rhsQ"][:, :, 0])
    test(rhs.rhou[0].reshape((-1, K)), matlab_data["rhsQ"][:, :, 1])
    test(rhs.rhou[1].reshape((-1, K)), matlab_data["rhsQ"][:, :, 2])
    test(rhs.E.reshape((-1, K)), matlab_data["rhsQ"][:, :, 3])


if __name__ == "__main__":
    # test_iso_vortex(4)
    # test_fw_step(1, "HLLC")
    # test_fw_step(1, "HLL")
    # test_fw_step(1, "Roe")

    N = 5

    home_path = Path(__file__).resolve().parent.parent
    mesh_path = home_path / "mesh" / "gambit" / "cylinderA00075b.neu"

    mesh = read_mesh(mesh_path)
    mesh.plot()
    
    mesh = mesh.refine()
    mesh.plot(show_elem_id=False, show_vtx_id=False)

    meshops = MeshOps(mesh, N=N)
    mdims, mdata = meshops.build_mesh_data()

    time = 0.0
    bd_flux_type = "HLL"
    gamma = 1.4
    stepper_type = "ssp2"

    q = iso_vortex_sol_fn(mdata.xyz, time)
    q = fw_step_ic(mdata.xyz)
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    def plot(q):
        p = _c2p(q, gamma)

        axs[0, 0].set_title("Density")
        axs[0, 1].set_title("Momentum x")
        axs[0, 2].set_title("Momentum y")
        axs[1, 0].set_title("Energy")
        axs[1, 1].set_title("Pressure")
        axs[1, 2].set_title("Velocity magnitude")
        plot_2d(meshops, jnp.log(q.rho), ax=axs[0, 0])
        plot_2d(meshops, q.rhou[0], ax=axs[0, 1])
        plot_2d(meshops, q.rhou[1], ax=axs[0, 2])
        plot_2d(meshops, q.E, ax=axs[1, 0])
        plot_2d(meshops, p.p, ax=axs[1, 1])
        plot_2d(meshops, jnp.linalg.norm(p.u, axis=0), ax=axs[1, 2])

        plt.pause(0.01)

    plot(q)

    for tstep in range(1, 10000):
        if tstep % 100 == 0:
            print(f"Time step {tstep}")

        dt = _dt(_wave_speed(_c2p(q, gamma), gamma), np.min(mesh.inradius), N)

        q = step(
            mdata,
            mdims,
            q,
            time,
            dt,
            bd_flux_type,
            gamma,
            fw_step_bc_fn,
            stepper_type,
        )
        time = dt * tstep

        if tstep % 100 == 0:
            plot(q)

    from pyndg.physics.euler import _PROFILER

    for key, value in _PROFILER.items():
        print(f"{key:<15}: {sum(value):.4f} seconds")

    plot(q)
    # print("time:", time)
    # plot(iso_vortex_sol_fn(mdata.xyz, time))
    plt.show()
