import scipy.io
from scipy.sparse.linalg import spsolve_triangular
import numpy as np
import time

from dev_poisson import Poisson2D, matvec_vmap, matvec_fori, matvec_segment
from sparse_solver import prepare, matvec_v0, matvec_v1, matvec_v2, matvec_v3

from pyndg.mesh_2d import Mesh2D
from pyndg.bc_2d import BC

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

jax.config.update("jax_enable_x64", True)

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
    ins2d_render(mesh, state)
    ins2d_lift_drag(mesh, state)
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


def ins2d_render(mesh, state):
    pass


def ins2d_lift_drag(mesh, state):
    pass


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
            print(f"{field:12}: {diff < 1e-14} {diff:.4e}")
        except:
            pass
    print("===============================")


# TODO:
# - build Poisson matrix in python
# - check matvec product timings in CRS vs COO vs Block Native vs Block Synth both CPU and GPU
# - GPU scalable solver ideas, to test **scaling** for different N:
# --- richardson on Cholesky + SPAI
# --- CG + SPAI [separate and together Chol factors]
# --- pGMG


class DummyParams:
    def __init__(self, **kwargs):
        self.has_periodic_bc = False
        self.bc = {
            12: BC.Dirichlet, # 12: Out
            11: BC.Neumann,   # 11: In
            13: BC.Neumann,   # 13: Wall
            15: BC.Neumann,   # 15: Cyl
        }
        for k in kwargs:
            setattr(self, k, kwargs[k])

def run_benchmark(name, func, args, iters=100):
    # Warmup (compilation happens here)
    _ = func(*args).block_until_ready()

    t0 = time.time()
    for _ in range(iters):
        func(*args).block_until_ready()  # Crucial for JAX timing!
    t1 = time.time()

    print(f"{name:16} {(t1 - t0) * 1000 / iters:8.2f} [ms]")

if __name__ == "__main__":
    import cProfile
    import pstats
    import matplotlib.pyplot as plt

    DIR = "/home/matteo/Documents/nodal-dg/Codes1.1/"

    N = 10

    print(f"Loading reference mat N={N}")
    static_state, mesh = load(DIR + f"INS2D_N{N}_STATIC.mat", True)
    state = load(DIR + f"INS2D_N{N}_ts2.mat")

    print("Reading mesh")
    params = DummyParams(N=N, mesh_file=DIR + "Grid/CFD/cylinderA00075b.neu")
    pyndg_mesh = Mesh2D(params)
    pyndg_mesh.initialize()

    poisson_pb = Poisson2D(None, pyndg_mesh)
    poisson_pb.assemble()

    PRperm_inv = np.empty_like(static_state.PRperm) 
    PRperm_inv[static_state.PRperm] = np.arange(len(PRperm_inv), dtype=np.int32)

    PRsystem_org = static_state.PRsystem[PRperm_inv][:, PRperm_inv]
    err = PRsystem_org - poisson_pb.stiff_mat
    print("Pressure System Error:", np.max(np.abs(err)))


    print("="*70)
    print("="*70)

    n_iters = 100

    x_np = np.ones(PRsystem_org.shape[1])
    x_jx = jnp.ones(PRsystem_org.shape[1])

    mat_scipy_csr = static_state.PRsystem
    mat_scipy_coo = static_state.PRsystem.tocoo()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        mat_scipy_coo @ x_np
    t1 = time.perf_counter()
    print(f"Scipy COO matvec     {(t1 - t0) * 1000 / n_iters:.2f} [ms]")

    t0 = time.perf_counter()
    for _ in range(n_iters):
        mat_scipy_csr @ x_np
    t1 = time.perf_counter()
    print(f"Scipy CSR matvec     {(t1 - t0) * 1000 / n_iters:.2f} [ms]")

    t0 = time.perf_counter()
    for _ in range(n_iters):
        poisson_pb.matvec(x_np)
    t1 = time.perf_counter()
    print(f"Block numpy matvec   {(t1 - t0) * 1000 / n_iters:.2f} [ms]")


    matvec_vmap_ = jax.jit(lambda a, b, c: matvec_vmap(a, b, c, mesh.Np))
    matvec_segment_ = jax.jit(lambda a, b, c: matvec_segment(a, b, c, mesh.Np))
    matvec_fori_ = jax.jit(lambda a, b, c: matvec_fori(a, b, c, mesh.Np))

    ij_jx = jnp.array(poisson_pb.ij)
    stiff_jx = jnp.array(poisson_pb.stiff)
    run_benchmark("JAX Vmap", matvec_vmap_, (x_jx, ij_jx, stiff_jx))
    run_benchmark(
        "JAX Segment Sum",
        matvec_segment_,
        (x_jx, ij_jx, stiff_jx),
    )
    run_benchmark("JAX Fori Loop", matvec_fori_, (x_jx, ij_jx, stiff_jx))

    bcoo_mat = jsparse.BCOO.from_scipy_sparse(mat_scipy_csr, index_dtype=jnp.int32)
    bcsr_mat = jsparse.BCSR.from_scipy_sparse(mat_scipy_csr, index_dtype=jnp.int32)

    matvec_plain = jax.jit(lambda m, v: m @ v)

    matvec_plain(bcoo_mat, x_jx).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        matvec_plain(bcoo_mat, x_jx).block_until_ready()
    t1 = time.perf_counter()
    print(f"JAX COO matvec       {(t1 - t0) * 1000 / n_iters:.2f} [ms]")

    matvec_plain(bcoo_mat, x_jx).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        matvec_plain(bcsr_mat, x_jx).block_until_ready()
    t1 = time.perf_counter()
    print(f"JAX CSR matvec       {(t1 - t0) * 1000 / n_iters:.2f} [ms]")


    d_jax, c_jax, _, max_bw = prepare(mat_scipy_csr)
    print("BW", max_bw)
    run_benchmark("JAX Synth blk V0", matvec_v0, (d_jax, c_jax, x_jx))
    run_benchmark("JAX Synth blk V1", matvec_v1, (d_jax, c_jax, x_jx))
    run_benchmark("JAX Synth blk V2", matvec_v2, (d_jax, c_jax, x_jx))
    run_benchmark("JAX Synth blk V3", matvec_v3, (d_jax, c_jax, x_jx))



    d_jax, c_jax, _, max_bw = prepare(static_state.PRsystemC)
    print("BW", max_bw)
    run_benchmark("JAX Synth blk V0", matvec_v0, (d_jax, c_jax, x_jx))
    run_benchmark("JAX Synth blk V1", matvec_v1, (d_jax, c_jax, x_jx))
    run_benchmark("JAX Synth blk V2", matvec_v2, (d_jax, c_jax, x_jx))
    run_benchmark("JAX Synth blk V3", matvec_v3, (d_jax, c_jax, x_jx))

    # do_check = True
    # do_profile = False

    # run_time = 0

    # def run():
    #     # N = 5, 0.60[s] for 100 steps, 0.45 for backslash -> 1.5ms / system
    #     # N = 7, 1.47[s] for 100 steps, 1.21 for backslash -> 4.0ms / system
    #     # N = 8, 2.00[s] for 100 steps, 1.75 for backslash -> 6.0ms / system
    #     global run_time
    #     t0 = time.time()
    #     for step in range(2, 20):
    #         t0 = time.perf_counter()
    #         ins2d_step(mesh, static_state, state)
    #         run_time += time.perf_counter() - t0
    #         if do_check:
    #             state_ref = load(DIR + f"INS2D_N{N}_ts{step + 1}.mat")
    #             compare(state, state_ref)

    # if do_profile:
    #     with cProfile.Profile() as pr:
    #         run()

    #     stats = pstats.Stats(pr)
    #     stats.sort_stats("cumulative").print_stats()
    # else:
    #     run()

    # avg_solve_time = system_solve_time / system_solve_cnt
    # print(
    #     f"System solve time: {system_solve_time:.2f} [s] ({avg_solve_time*1000:.2f} [ms] / sys)"
    # )
    # print(f"Run time: {run_time:.2f}")
