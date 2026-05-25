from pathlib import Path
import jax.numpy as jnp
import scipy
import numpy as np

from pyndg.mesh import read_mesh
from pyndg.ops.meshops import MeshOps, apply_bc_maps
from pyndg.physics.euler import _rhs, EulerVarCons
from pyndg.mesh.bc import BC


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

    bc_type_map = apply_bc_maps(meshops, {13: BC.Dirichlet})
    bc_type_map = {key: jnp.array(val) for key, val in bc_type_map.items()}

    def bc_fn(var, time):
        map = bc_type_map[BC.Dirichlet].reshape((mdims.Nf, mdims.Nfp, mdims.K))
        xy = mdata.fxyz.reshape((mdims.dim, mdims.Nf, mdims.Nfp, mdims.K))
        sol_eval = iso_vortex_sol_fn(xy, time)
        return EulerVarCons(
            rho=var.rho.at[map].set(sol_eval.rho[map]),
            rhou=var.rhou.at[:, map].set(sol_eval.rhou[:, map]),
            E=var.E.at[map].set(sol_eval.E[map]),
        )

    print("time:", matlab_data["time"])

    rhs, q_plus_tmp, q_plus, q_minus, bd_flux, rhs_new = _rhs(
        mdata,
        mdims,
        q_matlab,
        time=matlab_data["time"],
        bd_flux_type="LF",
        gamma=1.4,
        bc_fn=bc_fn,
    )

    test(rhs.rho, matlab_data["rhsQ_tmp"][:, :, 0])
    test(rhs.rhou[0], matlab_data["rhsQ_tmp"][:, :, 1])
    test(rhs.rhou[1], matlab_data["rhsQ_tmp"][:, :, 2])
    test(rhs.E, matlab_data["rhsQ_tmp"][:, :, 3])

    K = mdims.K
    test(q_plus_tmp.rho.reshape((-1, K)), matlab_data["QP_tmp"][:, :, 0])
    test(q_plus_tmp.rhou[0].reshape((-1, K)), matlab_data["QP_tmp"][:, :, 1])
    test(q_plus_tmp.rhou[1].reshape((-1, K)), matlab_data["QP_tmp"][:, :, 2])
    test(q_plus_tmp.E.reshape((-1, K)), matlab_data["QP_tmp"][:, :, 3])

    test(q_minus.rho.reshape((-1, K)), matlab_data["QM"][:, :, 0])
    test(q_minus.rhou[0].reshape((-1, K)), matlab_data["QM"][:, :, 1])
    test(q_minus.rhou[1].reshape((-1, K)), matlab_data["QM"][:, :, 2])
    test(q_minus.E.reshape((-1, K)), matlab_data["QM"][:, :, 3])

    test(q_plus.rho.reshape((-1, K)), matlab_data["QP"][:, :, 0])
    test(q_plus.rhou[0].reshape((-1, K)), matlab_data["QP"][:, :, 1])
    test(q_plus.rhou[1].reshape((-1, K)), matlab_data["QP"][:, :, 2])
    test(q_plus.E.reshape((-1, K)), matlab_data["QP"][:, :, 3])

    test(rhs_new.rho.reshape((-1, K)), matlab_data["rhsQ"][:, :, 0])
    test(rhs_new.rhou[0].reshape((-1, K)), matlab_data["rhsQ"][:, :, 1])
    test(rhs_new.rhou[1].reshape((-1, K)), matlab_data["rhsQ"][:, :, 2])
    test(rhs_new.E.reshape((-1, K)), matlab_data["rhsQ"][:, :, 3])


def test_fw_step(N):
    home_path = Path(__file__).resolve().parent.parent
    mesh_path = home_path / "mesh" / "gambit" / "fstepA001.neu"

    mesh = read_mesh(mesh_path)
    # mesh.plot()

    matlab_path = "/home/matteo/Documents/nodal-dg/Codes1.1/"
    matlab_data = scipy.io.loadmat(matlab_path + f"ShockEulerRHS2D_N{N}_init_HLL.mat")

    q_matlab = _mat2py(matlab_data["Q"])

    meshops = MeshOps(mesh, N=N)
    mdims, mdata = meshops.build_mesh_data()

    bc_type_map = apply_bc_maps(
        meshops, {13: BC.Dirichlet, 11: BC.Dirichlet, 12: BC.Dirichlet}
    )
    bc_type_map = {key: jnp.array(val) for key, val in bc_type_map.items()}
    gamma = 1.4

    def bc_fn(var, time):
        rhoin = gamma
        uin = 3.0
        vin = 0.0
        pin = 1.0
        Ein = pin / (gamma - 1.0) + 0.5 * rhoin * (uin**2 + vin**2)

        # Inflow conditions -- uniform inflow
        map_inlet = mdata.bc_maps[11].reshape((mdims.Nf, mdims.Nfp, mdims.K))
        rho = var.rho.at[map_inlet].set(rhoin)
        rhou = var.rhou.at[:, map_inlet].set(rhoin * jnp.array([uin, vin])[:, None])
        E = var.E.at[map_inlet].set(Ein)

        # Outflow conditions -- supersonic outflow ( do nothing )

        # Wall conditions -- reflective, isothermal, i.e., n.u=0, T=T(t=0)
        # reverse flow in normal direction in ghost elements
        map_wall = mdata.bc_maps[13].reshape((mdims.Nf, mdims.Nfp, mdims.K))
        rhouW = var.rhou[:, map_wall]
        nxyW = mdata.nxyz[:, mdata.bc_maps[13]]
        tmp = rhouW - 2 * nxyW * jnp.sum(nxyW * rhouW, axis=0)
        rhou = rhou.at[:, map_wall].set(tmp)

        return EulerVarCons(
            rho=rho,
            rhou=rhou,
            E=E,
        )

    print("time:", matlab_data["time"])

    rhs, q_plus_tmp, q_plus, q_minus, bd_flux, rhs_new = _rhs(
        mdata,
        mdims,
        q_matlab,
        time=matlab_data["time"],
        bd_flux_type="HLL",
        gamma=1.4,
        bc_fn=bc_fn,
    )

    test(rhs.rho, matlab_data["rhsQ_tmp"][:, :, 0])
    test(rhs.rhou[0], matlab_data["rhsQ_tmp"][:, :, 1])
    test(rhs.rhou[1], matlab_data["rhsQ_tmp"][:, :, 2])
    test(rhs.E, matlab_data["rhsQ_tmp"][:, :, 3])

    K = mdims.K
    test(q_plus_tmp.rho.reshape((-1, K)), matlab_data["QP_tmp"][:, :, 0])
    test(q_plus_tmp.rhou[0].reshape((-1, K)), matlab_data["QP_tmp"][:, :, 1])
    test(q_plus_tmp.rhou[1].reshape((-1, K)), matlab_data["QP_tmp"][:, :, 2])
    test(q_plus_tmp.E.reshape((-1, K)), matlab_data["QP_tmp"][:, :, 3])

    test(q_minus.rho.reshape((-1, K)), matlab_data["QM"][:, :, 0])
    test(q_minus.rhou[0].reshape((-1, K)), matlab_data["QM"][:, :, 1])
    test(q_minus.rhou[1].reshape((-1, K)), matlab_data["QM"][:, :, 2])
    test(q_minus.E.reshape((-1, K)), matlab_data["QM"][:, :, 3])

    test(q_plus.rho.reshape((-1, K)), matlab_data["QP"][:, :, 0])
    test(q_plus.rhou[0].reshape((-1, K)), matlab_data["QP"][:, :, 1])
    test(q_plus.rhou[1].reshape((-1, K)), matlab_data["QP"][:, :, 2])
    test(q_plus.E.reshape((-1, K)), matlab_data["QP"][:, :, 3])

    test(rhs_new.rho.reshape((-1, K)), matlab_data["rhsQ"][:, :, 0])
    test(rhs_new.rhou[0].reshape((-1, K)), matlab_data["rhsQ"][:, :, 1])
    test(rhs_new.rhou[1].reshape((-1, K)), matlab_data["rhsQ"][:, :, 2])
    test(rhs_new.E.reshape((-1, K)), matlab_data["rhsQ"][:, :, 3])


if __name__ == "__main__":
    test_iso_vortex(4)
    test_fw_step(1)
