import jax

from pyndg.mesh import read_mesh
from pyndg.ops.meshops import MeshOps, apply_bc_maps
from pyndg.physics.ins import (
    IncNavierStokesState,
    IncompressibleNavierStokes,
    _advection_step,
    _viscous_step,
    _pressure_step,
    _update_bc,
)
from pyndg.mesh.bc import BC

from pathlib import Path
import scipy.io
from scipy.sparse.linalg import spsolve_triangular
import numpy as np
import jax.numpy as jnp
import time

PATH = "/home/matteo/Documents/nodal-dg/Codes1.1/"

system_solve_time, system_solve_cnt = 0, 0


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


system_solve_time = 0
system_solve_cnt = 0


def load(file_path, load_static=False):
    data = scipy.io.loadmat(file_path)
    if load_static:
        static_state = StaticState(data)
        mesh = Mesh(data)
        return static_state, mesh
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
        "ic_fn": lambda xyz: np.zeros_like(xyz),
    }
    problem = IncompressibleNavierStokes(mesh_ops, params)

    def test(a, b):
        assert a.shape == b.shape
        err = np.max(np.abs(a - b))
        print(f"Max difference: {err: .2e} | {err < 1e-10}")

    print(f"Loading reference mat N={N}")
    static_state, mesh_matlab = load(PATH + f"INS2D_N{N}_STATIC.mat", True)
    state = load(PATH + f"INS2D_N{N}_ts2.mat")

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

    mesh_dims, mesh_data = mesh_ops.build_mesh_data()
    bc_type_map = apply_bc_maps(
        mesh_ops,
        {
            11: BC.Dirichlet,
            13: BC.Dirichlet,
            14: BC.Dirichlet,
        },
    )
    mesh_data = mesh_data._replace(
        vmap_m=mesh_data.vmap_m.reshape(-1, mesh_ops.K),
        vmap_p=mesh_data.vmap_p.reshape(-1, mesh_ops.K),
    )

    jax_state = IncNavierStokesState(
        u_old=jnp.stack([state.Uxold, state.Uyold], axis=0),
        u=jnp.stack([state.Ux, state.Uy], axis=0),
        bc_u=None,
        bc_dudn=None,
        rhs_bc_u=None,
        p=None,
        bc_p=None,
        dpdn=jnp.array(state.dpdn),
        rhs_bc_p=None,
        Nu=jnp.stack([state.NUx, state.NUy], axis=0),
        nu=jnp.array(static_state.nu),
        time=jnp.array(state.time),
        dt=jnp.array(state.dt),
        timestep=state.tstep,
        ref_bc_u=jnp.stack([static_state.refbcUx, static_state.refbcUy], axis=0),
        ref_rhs_bc_u=jnp.stack(
            [
                static_state.refrhsbcUx.reshape(-1, mesh_dims.K, order="F"),
                static_state.refrhsbcUy.reshape(-1, mesh_dims.K, order="F"),
            ],
            axis=0,
        ),
        ref_bc_p=jnp.array(static_state.refbcPR),
        ref_rhs_bc_p=jnp.array(
            static_state.refrhsbcPR.reshape(-1, mesh_dims.K, order="F")
        ),
        ref_bc_dudn=jnp.array(static_state.refbcdUndt),
    )

    for step in range(2, 10):
        jax_state = _update_bc(jax_state, u_time_fn, du_time_fn, p_time_fn)

        u_tilde, Nu = _advection_step(jax_state, mesh_data, bc_type_map, mesh_dims)
        dpdn, uTT, p_new = _pressure_step(
            jax_state, mesh_data, mesh_dims, u_tilde, Nu, bc_type_map, static_state
        )
        u_new = _viscous_step(jax_state, mesh_data, mesh_dims, uTT, static_state)

        ####

        jax_state = jax_state._replace(
            u_old=jax_state.u,
            u=u_new,
            Nu=Nu,
            time=jax_state.timestep * jax_state.dt,
            timestep=jax_state.timestep + 1,
            dpdn=dpdn,
        )

        ####

        state_ref = load(PATH + f"INS2D_N{N}_ts{step + 1}.mat")

        test(jax_state.u[0], state_ref.Ux)
        test(jax_state.u[1], state_ref.Uy)
        test(jax_state.Nu[0], state_ref.NUx)
        test(jax_state.Nu[1], state_ref.NUy)
        test(jax_state.dpdn, state_ref.dpdn)
        test(state_ref.UxT, u_tilde[0])
        test(state_ref.UyT, u_tilde[1])
        test(state_ref.UxTT, uTT[0])
        test(state_ref.UyTT, uTT[1])

        print("================================")
