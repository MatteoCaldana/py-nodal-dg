from pyndg.mesh import read_mesh
from pyndg.ops.meshops import MeshOps
from pyndg.physics.ins import IncompressibleNavierStokes

from pathlib import Path
import scipy.io
from scipy.sparse.linalg import spsolve_triangular
import numpy as np
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


def ins2d_step(mesh, static_state, state):
    temporal_scaling(static_state, state)
    ins2d_advection(mesh, state, static_state)
    ins2d_pressure(mesh, state, static_state)
    ins2d_viscous(mesh, state, static_state)
    state.time = state.tstep * state.dt
    state.tstep += 1


def temporal_scaling(static_state, state):
    time = state.time
    dt = state.dt

    # Time factors
    tfac = np.sin(np.pi * time / 8)
    tfac1 = np.sin(np.pi * (time + dt) / 8)
    tpfac = (np.pi / 8) * np.cos(np.pi * time / 8)
    tpfac1 = (np.pi / 8) * np.cos(np.pi * time / 8)
    tpfac2 = (np.pi / 8) * np.cos(np.pi * time / 8)

    # Boundary condition calculations
    state.bcUx = tfac * static_state.refbcUx
    state.rhsbcUx = tfac1 * static_state.refrhsbcUx
    state.bcUy = tfac * static_state.refbcUy
    state.rhsbcUy = tfac1 * static_state.refrhsbcUy
    state.bcPR = tpfac1 * static_state.refbcPR
    state.rhsbcPR = tpfac2 * static_state.refrhsbcPR
    state.bcdUndt = tpfac * static_state.refbcdUndt


def Div2D(mesh, u, v):
    ur = mesh.Dr @ u
    us = mesh.Ds @ u
    vr = mesh.Dr @ v
    vs = mesh.Ds @ v
    return mesh.rx * ur + mesh.sx * us + mesh.ry * vr + mesh.sy * vs


def Curl2D(mesh, ux, uy):
    uxr = mesh.Dr @ ux
    uxs = mesh.Ds @ ux
    uyr = mesh.Dr @ uy
    uys = mesh.Ds @ uy
    return mesh.rx * uyr + mesh.sx * uys - mesh.ry * uxr - mesh.sy * uxs


def Grad2D(mesh, u):
    ur = mesh.Dr @ u
    us = mesh.Ds @ u

    ux = mesh.rx * ur + mesh.sx * us
    uy = mesh.ry * ur + mesh.sy * us
    return ux, uy


system_solve_time = 0
system_solve_cnt = 0


def ins2d_advection(mesh, state, ss):
    # 1. Evaluate flux vectors
    fxUx = state.Ux * state.Ux
    fyUx = state.Ux * state.Uy
    fxUy = state.Ux * state.Uy
    fyUy = state.Uy * state.Uy

    # 2. Save old nonlinear terms
    NUxold = state.NUx.copy()
    NUyold = state.NUy.copy()

    # 3. Evaluate inner-product (Assuming Div2D is defined elsewhere in your Python code)
    state.NUx = Div2D(mesh, fxUx, fyUx)
    state.NUy = Div2D(mesh, fxUy, fyUy)

    # 4. Interpolate velocity to face nodes
    UxM = state.Ux.flat[mesh.vmapMC].reshape((mesh.Nfp * mesh.Nfaces, mesh.K))
    UyM = state.Uy.flat[mesh.vmapMC].reshape((mesh.Nfp * mesh.Nfaces, mesh.K))
    UxP = state.Ux.flat[mesh.vmapPC].reshape((mesh.Nfp * mesh.Nfaces, mesh.K))
    UyP = state.Uy.flat[mesh.vmapPC].reshape((mesh.Nfp * mesh.Nfaces, mesh.K))

    # 5. Set '+' trace at boundary face nodes
    UxP.flat[mesh.mapIC] = state.bcUx.flat[mesh.mapIC]
    UxP.flat[mesh.mapWC] = state.bcUx.flat[mesh.mapWC]
    UxP.flat[mesh.mapCC] = state.bcUx.flat[mesh.mapCC]
    UyP.flat[mesh.mapIC] = state.bcUy.flat[mesh.mapIC]
    UyP.flat[mesh.mapWC] = state.bcUy.flat[mesh.mapWC]
    UyP.flat[mesh.mapCC] = state.bcUy.flat[mesh.mapCC]

    # 6. Evaluate flux vectors at face nodes
    fxUxM = UxM * UxM
    fyUxM = UyM * UxM
    fxUyM = UxM * UyM
    fyUyM = UyM * UyM
    fxUxP = UxP * UxP
    fyUxP = UyP * UxP
    fxUyP = UxP * UyP
    fyUyP = UyP * UyP

    # 7. Normal velocity and Lax-Friedrichs/Rusonov flux
    UDotNM = UxM * mesh.nx + UyM * mesh.ny
    UDotNP = UxP * mesh.nx + UyP * mesh.ny
    maxvel = np.maximum(np.abs(UDotNM), np.abs(UDotNP))

    # 8. Evaluate maximum normal velocity over each face
    for f in range(mesh.Nfaces):
        maxvel[f * mesh.Nfp : (f + 1) * mesh.Nfp, :] = np.max(
            maxvel[f * mesh.Nfp : (f + 1) * mesh.Nfp, :], axis=0
        )

    # 9. Form Fluxes
    fluxUx = 0.5 * (
        -mesh.nx * (fxUxM - fxUxP) - mesh.ny * (fyUxM - fyUxP) - maxvel * (UxP - UxM)
    )
    fluxUy = 0.5 * (
        -mesh.nx * (fxUyM - fxUyP) - mesh.ny * (fyUyM - fyUyP) - maxvel * (UyP - UyM)
    )

    # 10. Combine volume and surface terms
    # Use @ for matrix multiplication with LIFT
    state.NUx = state.NUx + mesh.LIFT @ (mesh.Fscale * fluxUx)
    state.NUy = state.NUy + mesh.LIFT @ (mesh.Fscale * fluxUy)

    # 11. Compute intermediate velocity (U~, V~)
    state.UxT = (
        (ss.a0 * state.Ux + ss.a1 * state.Uxold)
        - state.dt * (ss.b0 * state.NUx + ss.b1 * NUxold)
    ) / ss.g0
    state.UyT = (
        (ss.a0 * state.Uy + ss.a1 * state.Uyold)
        - state.dt * (ss.b0 * state.NUy + ss.b1 * NUyold)
    ) / ss.g0


def ins2d_pressure(mesh, state, ss):
    DivUT = Div2D(mesh, state.UxT, state.UyT)

    # 2. Compute dp/dn components
    CurlU = Curl2D(mesh, state.Ux, state.Uy)
    dCurlUdx, dCurlUdy = Grad2D(mesh, CurlU)

    res1 = -state.NUx - ss.nu * dCurlUdy
    res2 = -state.NUy + ss.nu * dCurlUdx

    # 3. Save old and compute new dp/dn
    dpdnold = state.dpdn.copy()

    # 4. Deciding Neumann nodes (Concatenating boundary maps)
    nbcmapD = np.concatenate([mesh.mapIC, mesh.mapWC, mesh.mapCC])
    vbcmapD = np.concatenate([mesh.vmapIC, mesh.vmapWC, mesh.vmapCC])

    # dpdn(nbcmapD) = nx.*res1 + ny.*res2
    state.dpdn = np.zeros_like(state.dpdn)
    state.dpdn.flat[nbcmapD] = (
        mesh.nx.flat[nbcmapD] * res1.flat[vbcmapD]
        + mesh.ny.flat[nbcmapD] * res2.flat[vbcmapD]
    )

    # Update and subtract boundary forcing
    state.dpdn -= state.bcdUndt

    # 5. Evaluate RHS for Pressure Poisson Equation
    term_vol = mesh.J * (-DivUT * ss.g0 / state.dt)
    term_sur = mesh.LIFT @ (mesh.sJ * (ss.b0 * state.dpdn + ss.b1 * dpdnold))
    PRrhs = mesh.MassMatrix @ (term_vol + term_sur)

    # 6. Add Dirichlet boundary forcing
    PRrhs_flat = PRrhs.ravel(order="F") + state.rhsbcPR.ravel(order="F")
    PRrhs_flat = PRrhs_flat[ss.PRperm]

    # 7. Pressure Solve (Assuming PRperm, PRsystemCT, PRsystemC are pre-computed)
    global system_solve_time, system_solve_cnt
    t0 = time.perf_counter()
    tmp = spsolve_triangular(ss.PRsystemC.T, PRrhs_flat, lower=True)
    PR_sol = spsolve_triangular(ss.PRsystemC, tmp, lower=False)
    t1 = time.perf_counter()
    system_solve_time += t1 - t0
    system_solve_cnt += 1

    # Reconstruct PR array using the permutation
    PR = np.empty_like(PR_sol)
    PR[ss.PRperm] = PR_sol
    PR = PR.reshape((mesh.Np, mesh.K), order="F")

    # 8. Compute (U~~, V~~) = (U~, V~) - dt*grad PR
    dPRdx, dPRdy = Grad2D(mesh, PR)

    # 9. Increment to (Ux~~, Uy~~)
    state.UxTT = state.UxT - state.dt * (dPRdx) / ss.g0
    state.UyTT = state.UyT - state.dt * (dPRdy) / ss.g0


def ins2d_viscous(mesh, state, ss):
    J_mean = np.mean(mesh.J, axis=0)

    mmUxTT = J_mean * (mesh.MassMatrix @ state.UxTT)
    mmUyTT = J_mean * (mesh.MassMatrix @ state.UyTT)

    # 2. Formulate the full RHS for the Helmholtz system
    Uxrhs_flat = (ss.g0 * mmUxTT.ravel(order="F")) / (
        ss.nu * state.dt
    ) + state.rhsbcUx.ravel(order="F")
    Uyrhs_flat = (ss.g0 * mmUyTT.ravel(order="F")) / (
        ss.nu * state.dt
    ) + state.rhsbcUy.ravel(order="F")

    # 3. Save current velocity to old variables
    state.Uxold = state.Ux.copy()
    state.Uyold = state.Uy.copy()

    # Backsolve twice (Assuming VELsystemCT and VELsystemC are the factored matrices)
    Uxrhs_flat = Uxrhs_flat[ss.VELperm]
    Uyrhs_flat = Uyrhs_flat[ss.VELperm]
    global system_solve_time, system_solve_cnt
    t0 = time.perf_counter()
    tmp_x = spsolve_triangular(ss.VELsystemC.T, Uxrhs_flat, lower=True)
    Ux_sol = spsolve_triangular(ss.VELsystemC, tmp_x, lower=False)
    tmp_y = spsolve_triangular(ss.VELsystemC.T, Uyrhs_flat, lower=True)
    Uy_sol = spsolve_triangular(ss.VELsystemC, tmp_y, lower=False)
    t1 = time.perf_counter()
    system_solve_time += t1 - t0
    system_solve_cnt += 2

    # Update the state variables
    tmp_Ux = np.empty_like(Ux_sol)
    tmp_Ux[ss.VELperm] = Ux_sol
    state.Ux = tmp_Ux.reshape((mesh.Np, mesh.K), order="F")
    tmp_Uy = np.empty_like(Uy_sol)
    tmp_Uy[ss.VELperm] = Uy_sol
    state.Uy = tmp_Uy.reshape((mesh.Np, mesh.K), order="F")


def load(file_path, load_static=False):
    data = scipy.io.loadmat(file_path)
    if load_static:
        static_state = StaticState(data)
        mesh = Mesh(data)
        return static_state, mesh
    else:
        state = State(data)
        return state


def compare(state, state_ref):
    for field in dir(state_ref):
        if field.startswith("__"):
            continue

        try:
            v = getattr(state, field)
            vr = getattr(state_ref, field)
            diff = np.max(np.abs(v - vr))
            print(f"{field:12}: {diff < 1e-12} {diff:.4e}")
        except:
            pass
    print("===============================")


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
        err = np.max(np.abs(a - b))
        print(f"Max difference: {err: .2e} | {err < 1e-10}")

    print(f"Loading reference mat N={N}")
    static_state, mesh_matlab = load(PATH + f"INS2D_N{N}_STATIC.mat", True)
    state = load(PATH + f"INS2D_N{N}_ts2.mat")

    test(mesh_ops.ref_elem_ops.Dphi[0], mesh_matlab.Dr)
    test(mesh_ops.ref_elem_ops.Dphi[1], mesh_matlab.Ds)
    test(mesh_ops.J_rst_xyz[0, 0], mesh_matlab.rx)
    test(mesh_ops.J_rst_xyz[1, 0], mesh_matlab.sx)
    test(mesh_ops.J_rst_xyz[0, 1], mesh_matlab.ry)
    test(mesh_ops.J_rst_xyz[1, 1], mesh_matlab.sy)
    test(mesh_ops.vmap_m.flat, mesh_matlab.vmapMC)
    test(mesh_ops.vmap_p.flat, mesh_matlab.vmapPC)
    test(np.where(mesh_ops.bc_maps[11].flat)[0], np.sort(mesh_matlab.mapIC))
    test(np.where(mesh_ops.bc_maps[13].flat)[0], np.sort(mesh_matlab.mapWC))
    test(np.where(mesh_ops.bc_maps[14].flat)[0], np.sort(mesh_matlab.mapCC))
    test(mesh_ops.nxyz[0], mesh_matlab.nx)
    test(mesh_ops.nxyz[1], mesh_matlab.ny)
    test(mesh_ops.fscale.reshape((-1, mesh_ops.K)), mesh_matlab.Fscale)
    test(mesh_ops.ref_elem_ops.lift, mesh_matlab.LIFT)
    test(mesh_ops.J, mesh_matlab.J)
    test(mesh_ops.sJ.reshape((-1, mesh_ops.K)), mesh_matlab.sJ)
    test(mesh_ops.ref_elem_ops.int_phiphi, mesh_matlab.MassMatrix)

    print("=" * 70)

    for step in range(2, 20):
        ins2d_step(mesh_matlab, static_state, state)
        state_ref = load(PATH + f"INS2D_N{N}_ts{step + 1}.mat")
        compare(state, state_ref)
