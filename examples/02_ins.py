import jax

from pyndg.mesh import read_mesh
from pyndg.ops.meshops import MeshOps, apply_bc_maps
from pyndg.physics import ins
from pyndg.physics.ins import (
    _SPLITTING_COEFFS,
    IncNavierStokesState,
    IncompressibleNavierStokes,
)
from pyndg.mesh.bc import BC

from pathlib import Path
import scipy.io
import numpy as np
import jax.numpy as jnp
from scipy.sparse.linalg import spsolve_triangular

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


class StaticState:
    def __init__(self, data):
        self.nu = data["nu"].item()

        # Boundary / Reference Data
        self.refbcUx = data["refbcUx"]
        self.refrhsbcUx = data["refrhsbcUx"]
        self.refbcUy = data["refbcUy"]
        self.refrhsbcUy = data["refrhsbcUy"]
        self.refbcPR = data["refbcPR"]
        self.refrhsbcPR = data["refrhsbcPR"]
        self.refbcdUndt = data["refbcdUndt"]

        # system
        self.PRsystemC = data["PRsystemC"].tocsr()
        self.PRsystem = data["PRsystem"].tocsr()
        self.PRperm = data["PRperm"].astype(np.int32).squeeze() - 1
        self.VELsystemC = data["VELsystemC"].tocsr()
        self.VELsystem = data["VELsystem"].tocsr()
        self.VELperm = data["VELperm"].astype(np.int32).squeeze() - 1

        # splitting coef
        self.a0 = data["a0"].item()
        self.a1 = data["a1"].item()
        self.b0 = data["b0"].item()
        self.b1 = data["b1"].item()
        self.g0 = data["g0"].item()


class Mesh:
    def __init__(self, data):
        # Dimensions
        self.Np = int(data["Np"].item())
        self.Nfp = int(data["Nfp"].item())
        self.N = int(data["N"].item())
        self.K = int(data["K"].item())
        self.Nfaces = int(data["Nfaces"].item())
        self.NODETOL = data["NODETOL"].item()

        # Coordinates & Grid
        self.r = data["r"]
        self.s = data["s"]
        self.x = data["x"]
        self.y = data["y"]
        self.VX = data["VX"]
        self.VY = data["VY"]

        # Operators
        self.Dr = data["Dr"]
        self.Ds = data["Ds"]
        self.LIFT = data["LIFT"]
        self.Drw = data["Drw"]
        self.Dsw = data["Dsw"]
        self.MassMatrix = data["MassMatrix"]
        self.V = data["V"]
        self.invV = data["invV"]

        # Geometric Factors
        self.Fx = data["Fx"]
        self.Fy = data["Fy"]
        self.nx = data["nx"]
        self.ny = data["ny"]
        self.jac = data["jac"]
        self.J = data["J"]
        self.sJ = data["sJ"]
        self.Fscale = data["Fscale"]
        self.rx = data["rx"]
        self.ry = data["ry"]
        self.sx = data["sx"]
        self.sy = data["sy"]

        # Connectivity and Mapping (Corrected for 0-based indexing)
        idx_vars = [
            "vmapM",
            "vmapP",
            "vmapI",
            "vmapO",
            "vmapW",
            "vmapC",
            "mapB",
            "mapI",
            "mapO",
            "mapW",
            "mapC",
            "mapM",
            "mapP",
        ]

        def make_order_c(shape):
            range = np.arange(shape[0] * shape[1], dtype=np.int32)
            orderF = range.reshape(shape, order="F")
            orderC = np.empty(range.shape, dtype=int)
            orderC[orderF.flat] = range
            return orderC, orderF

        order_c_v, _ = make_order_c(self.x.shape)
        order_c_f, order_f_f = make_order_c(self.Fx.shape)

        for var in idx_vars:
            if var in data:
                # Subtract 1 from MATLAB indices to work in Python
                assert (data[var].shape[1] == 1) and (data[var].shape[0] > 1)
                data[var] = data[var].ravel().astype(np.int32)
                data[var] -= 1
                map = data[var]
                if var.startswith("v"):
                    mapC = order_c_v[map]
                    # plus/minus face mappings
                    # the ouput are faces in F order
                    if var.endswith("P") or var.endswith("M"):
                        mapC = mapC[order_f_f.flat]
                else:
                    mapC = order_c_f[map]

                setattr(self, var, map)
                setattr(self, var + "C", mapC)


def load(file_path, load_static=False):
    data = scipy.io.loadmat(file_path)
    if load_static:
        static_state = StaticState(data)
        try:
            mesh = Mesh(data)
            return static_state, mesh
        except:
            return static_state, None
    else:
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


def u_bc(xyz, nxyz, maps):
    u = jnp.zeros_like(xyz)
    y_in = xyz[1] + 0.20
    ux = jnp.where(maps[11], (1 / 0.41) ** 2 * 6 * y_in * (0.41 - y_in), 0)
    u = u.at[0].set(ux)
    return u


def p_bc(xyz, nxyz, maps):
    p = jnp.zeros(xyz.shape[1:])
    return p


@jax.jit
def dudn_bc(xyz, nxyz, maps):
    y_in = xyz[1] + 0.20
    dudn = jnp.where(maps[11], -((1 / 0.41) ** 2) * 6 * y_in * (0.41 - y_in), 0)
    return dudn


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


if __name__ == "__main__":
    home_path = Path(__file__).resolve().parent.parent
    mesh_path = home_path / "mesh" / "gambit" / "cylinderA00075b.neu"

    mesh = read_mesh(mesh_path)
    # mesh.plot()

    N = 1

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

    print(f"Loading reference mat N={N}")
    static_state, mesh_matlab = load(PATH + f"INS2D_N{N}_STATIC.mat", True)

    PRperm_inv = np.empty_like(static_state.PRperm)
    PRperm_inv[static_state.PRperm] = np.arange(len(PRperm_inv), dtype=np.int32)
    PRsystem_org = static_state.PRsystem[PRperm_inv][:, PRperm_inv]

    VELperm_inv = np.empty_like(static_state.VELperm)
    VELperm_inv[static_state.VELperm] = np.arange(len(VELperm_inv), dtype=np.int32)
    VELsystem_org = static_state.VELsystem[VELperm_inv][:, VELperm_inv]

    test(np.array(problem.nu), np.array(static_state.nu))

    test(problem.pr_sys, PRsystem_org)
    test(problem.adv_sys_1, VELsystem_org)

    test(mesh_ops.ref_elem_ops.Dphi[0], mesh_matlab.Dr)
    test(mesh_ops.ref_elem_ops.Dphi[1], mesh_matlab.Ds)
    test(mesh_ops.J_rst_xyz[0, 0], mesh_matlab.rx[0])
    test(mesh_ops.J_rst_xyz[1, 0], mesh_matlab.sx[0])
    test(mesh_ops.J_rst_xyz[0, 1], mesh_matlab.ry[0])
    test(mesh_ops.J_rst_xyz[1, 1], mesh_matlab.sy[0])
    test(mesh_ops.vmap_m.flatten(), mesh_matlab.vmapMC)
    test(mesh_ops.vmap_p.flatten(), mesh_matlab.vmapPC)
    test(np.where(mesh_ops.bc_maps[11].flatten())[0], np.sort(mesh_matlab.mapIC))
    test(np.where(mesh_ops.bc_maps[13].flatten())[0], np.sort(mesh_matlab.mapWC))
    test(np.where(mesh_ops.bc_maps[14].flatten())[0], np.sort(mesh_matlab.mapCC))
    test(mesh_ops.nxyz[0], mesh_matlab.nx)
    test(mesh_ops.nxyz[1], mesh_matlab.ny)
    test(mesh_ops.fscale.reshape((-1, mesh_ops.K)), mesh_matlab.Fscale)
    test(mesh_ops.ref_elem_ops.lift, mesh_matlab.LIFT)
    test(mesh_ops.J, mesh_matlab.J[0])
    test(mesh_ops.sJ.reshape((-1, mesh_ops.K)), mesh_matlab.sJ)
    test(mesh_ops.ref_elem_ops.int_phiphi, mesh_matlab.MassMatrix)

    print("=" * 70)

    state0 = problem._build_initial_state()
    mat_state0, _ = load(PATH + f"INS2D_N{N}_STATIC_PRESTAGE.mat", load_static=True)

    test(state0.ref_rhs_bc_p, mat_state0.refrhsbcPR.reshape(-1, mesh_ops.K, order="F"))
    test(
        state0.ref_rhs_bc_u[0], mat_state0.refrhsbcUx.reshape(-1, mesh_ops.K, order="F")
    )
    test(
        state0.ref_rhs_bc_u[1], mat_state0.refrhsbcUy.reshape(-1, mesh_ops.K, order="F")
    )
    test(state0.ref_bc_p, mat_state0.refbcPR.reshape(-1, mesh_ops.K, order="F"))
    test(state0.ref_bc_dudn, mat_state0.refbcdUndt.reshape(-1, mesh_ops.K, order="F"))
    test(state0.ref_bc_u[0], mat_state0.refbcUx)
    test(state0.ref_bc_u[1], mat_state0.refbcUy)

    VELsystem_org = mat_state0.VELsystem[VELperm_inv][:, VELperm_inv]
    test(problem.adv_sys_0, VELsystem_org)

    ss = mat_state0

    def pressure_solver(p_rhs):
        p_rhs = p_rhs.flatten(order="F")[ss.PRperm]
        tmp = spsolve_triangular(ss.PRsystemC.T, p_rhs, lower=True)
        p_sol = spsolve_triangular(ss.PRsystemC, tmp, lower=False)
        tmp[ss.PRperm] = p_sol
        return tmp.reshape((Np, K), order="F")

    def velocity_solver(u_rhs):
        u_rhs_flat = u_rhs.flatten(order="F")[ss.VELperm]
        tmp = spsolve_triangular(ss.VELsystemC.T, u_rhs_flat, lower=True)
        u_sol = spsolve_triangular(ss.VELsystemC, tmp, lower=False)
        tmp[ss.VELperm] = u_sol
        return tmp.reshape((Np, K), order="F")

    state = ins.step(
        state0,
        problem.mesh_data,
        problem.mesh_dims,
        problem.bc_type_map,
        problem.params["u_time_scale"],
        problem.params["du_time_scale"],
        problem.params["p_time_scale"],
        pressure_solver,
        velocity_solver,
        _SPLITTING_COEFFS["stage0"],
    )

    print("=" * 70)

    ss = static_state

    def pressure_solver(p_rhs):
        p_rhs = p_rhs.flatten(order="F")[ss.PRperm]
        tmp = spsolve_triangular(ss.PRsystemC.T, p_rhs, lower=True)
        p_sol = spsolve_triangular(ss.PRsystemC, tmp, lower=False)
        tmp[ss.PRperm] = p_sol
        return tmp.reshape((Np, K), order="F")

    def velocity_solver(u_rhs):
        u_rhs_flat = u_rhs.flatten(order="F")[ss.VELperm]
        tmp = spsolve_triangular(ss.VELsystemC.T, u_rhs_flat, lower=True)
        u_sol = spsolve_triangular(ss.VELsystemC, tmp, lower=False)
        tmp[ss.VELperm] = u_sol
        return tmp.reshape((Np, K), order="F")

    for step in range(2, 10):
        state = ins.step(
            state,
            problem.mesh_data,
            problem.mesh_dims,
            problem.bc_type_map,
            problem.params["u_time_scale"],
            problem.params["du_time_scale"],
            problem.params["p_time_scale"],
            pressure_solver,
            velocity_solver,
            _SPLITTING_COEFFS["stage1"],
        )

        ####

        state_ref = load(PATH + f"INS2D_N{N}_ts{step + 1}.mat")

        test(state_ref.Ux, state.u[0])
        test(state_ref.Uy, state.u[1])
        test(state_ref.NUx, state.Nu[0])
        test(state_ref.NUy, state.Nu[1])
        test(state_ref.dpdn, state.dpdn)

        print("================================")
