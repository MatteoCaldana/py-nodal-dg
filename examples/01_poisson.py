import jax
import jax.numpy as jnp

from pyndg.mesh import read_mesh
from pyndg.mg.smoother import jacobi, chebyshev, chebyshev_v2, bjacobi
from pyndg.mg.cg import pcg, pcg_np
from pyndg.mg.mg import build_prolongator_restrictor, mg_iter
from pyndg.ops.meshops import MeshOps
from pyndg.mesh.bc import BC
from pyndg.physics.poisson import Poisson, _block_assemble_kernel
from pyndg.utils.plot import plot_2d
import pyndg.backend

from pathlib import Path
import scipy.io
import numpy as np

PATH = "/home/matteo/Documents/nodal-dg/Codes1.1/"

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)


freq = 1.0


def sol_fn(x, y):
    return np.sin(freq * np.pi * x) * np.sin(freq * np.pi * y)


def dir_fn(x, y):
    return sol_fn(x, y)


def neu_fn(x, y):
    un_x = freq * np.pi * np.cos(freq * np.pi * x) * np.sin(freq * np.pi * y)
    un_y = freq * np.pi * np.sin(freq * np.pi * x) * np.cos(freq * np.pi * y)
    return un_x, un_y


def rhs_fn(x, y):
    return 2 * freq * freq * np.pi * np.pi * sol_fn(x, y)


_A_CALLS = 0
_P_CALLS = 0


def reset_counters():
    global _A_CALLS, _P_CALLS
    _A_CALLS = 0
    _P_CALLS = 0


import time

if __name__ == "__main__":
    N = 10

    home_path = Path(__file__).resolve().parent.parent
    mesh_path = home_path / "mesh" / "gambit" / "circA01.neu"

    mesh = read_mesh(mesh_path)
    mesh.face_tag = np.where(mesh.face_tag == 15, 1, mesh.face_tag)
    mesh_ops = MeshOps(mesh, N)
    params = {"penalty": 20.0, "bc_tags": {1: BC.Dirichlet}}
    problem = Poisson(params, mesh_ops)

    start_time = time.perf_counter()
    problem.assemble()
    end_time = time.perf_counter()
    print(f"Assemble time: {end_time - start_time:.3f} seconds")

    mesh_dims, mesh_data = mesh_ops.build_mesh_data()

    y = _block_assemble_kernel(
        mesh_data, params["penalty"], mesh_dims
    ).stiff.block_until_ready()
    time_start = time.perf_counter()
    for _ in range(10):
        y = _block_assemble_kernel(
            mesh_data, params["penalty"], mesh_dims
        ).stiff.block_until_ready()
    time_end = time.perf_counter()
    print(f"Block assemble time: {(time_end - time_start) / 10:.3f} seconds")

    data = scipy.io.loadmat(PATH + f"Poisson2D_N{N}.mat")
    print(np.max(np.abs(problem.mass_mat - data["M"])))
    print(np.max(np.abs(problem.stiff_mat - data["A"])))

    rhs = problem.assemble_rhs(rhs_fn, dir_fn, neu_fn)
    print(np.max(np.abs(rhs - data["rhs"])))

    uh = scipy.sparse.linalg.spsolve(problem.stiff_mat, rhs.flatten(order="F"))

    def Afn(x):
        global _A_CALLS
        _A_CALLS += 1
        return problem.stiff_mat @ x

    diag = problem.stiff_mat.diagonal()

    def P_jacobi(x):
        global _P_CALLS
        _P_CALLS += 1
        return x / diag

    bdiag_inv = np.linalg.inv(problem.stiff[: mesh.K])

    def P_bjacobi(x):
        global _P_CALLS
        _P_CALLS += 1
        x_shape = (bdiag_inv.shape[0], bdiag_inv.shape[2], 1)
        return (bdiag_inv @ x.reshape(x_shape)).reshape(-1)

    rhs_flat = rhs.flatten(order="F")

    uh_v2 = pcg_np(Afn, rhs_flat, tol=1e-8, max_iter=10000)
    print("CG [Plain]  :", uh_v2.iterations, uh_v2.converged)
    print(f"A calls: {_A_CALLS}, P calls: {_P_CALLS}")
    reset_counters()

    uh_v2 = pcg_np(Afn, rhs_flat, M=P_jacobi, tol=1e-8)
    print("CG [Jacobi] :", uh_v2.iterations, uh_v2.converged)
    print(f"A calls: {_A_CALLS}, P calls: {_P_CALLS}")
    reset_counters()

    uh_v2 = pcg_np(Afn, rhs_flat, M=P_bjacobi, tol=1e-8)
    print("CG [bJacobi]:", uh_v2.iterations, uh_v2.converged)
    print(f"A calls: {_A_CALLS}, P calls: {_P_CALLS}")
    reset_counters()

    print("Diff with ref (PCG)", np.max(np.abs(uh_v2.x - uh)))

    ##########################################################################
    # multigrid

    mesh_ops_coarse = MeshOps(mesh, N - 3)
    problem_coarse = Poisson(params, mesh_ops_coarse)
    problem_coarse.assemble()

    P, R = build_prolongator_restrictor(mesh_ops_coarse, mesh_ops)
    R = P.T

    Np_f = mesh_ops.Np
    K = mesh_ops.K
    Np_c = mesh_ops_coarse.Np

    uh = uh.reshape(mesh_ops.Np, mesh_ops.K, order="F")

    stiff_c = R @ (problem.stiff @ P)

    # choice: asseble from blocks or use coarse stiffness matrix
    stiff_mat_c = problem.assemble_stiff_from_blocks(mesh_ops.mesh, stiff_c)
    # stiff_mat_c = problem_coarse.stiff_mat

    smooth_iters = 1
    omega = 0.8
    D_inv = 1.0 / diag
    Db_inv = bdiag_inv
    lambda_max = scipy.sparse.linalg.eigsh(
        problem.stiff_mat, k=1, which="LM", return_eigenvectors=False
    )
    lambda_max = 1.1 * lambda_max
    lambda_min = 0.05 * lambda_max

    def sm_jacobi(AAfn, bb, xx):
        global _P_CALLS
        _P_CALLS += smooth_iters
        return jacobi(AAfn, D_inv, bb, xx, smooth_iters, omega=omega)

    def sm_chebyshev(AAfn, bb, xx):
        global _P_CALLS
        _P_CALLS += smooth_iters
        return chebyshev_v2(AAfn, bb, xx, smooth_iters, lambda_min, lambda_max)

    def sm_bjacobi(AAfn, bb, xx):
        global _P_CALLS
        _P_CALLS += smooth_iters
        return bjacobi(AAfn, Db_inv, bb, xx, smooth_iters, omega=1.0)

    # choice: smoother
    smoother = sm_bjacobi

    def mg_prec(rhs):
        x0 = np.zeros_like(rhs)
        for _ in range(1):
            x0 = mg_iter(
                Afn,
                rhs,
                x0,
                smoother,
                P,
                R,
                stiff_mat_c,
            )
        return x0

    reset_counters()
    mgcg = pcg_np(Afn, rhs_flat, M=mg_prec, tol=1e-8, max_iter=300)
    uh_mg = mgcg.x.reshape(mesh_ops.Np, mesh_ops.K, order="F")
    print("CG [MG]:", mgcg.iterations, mgcg.converged)
    print(f"A calls: {_A_CALLS}, P calls: {_P_CALLS}")
    reset_counters()

    ##########################################################################

    uh = uh.reshape(mesh_ops.Np, mesh_ops.K, order="F")
    print("Diff with ref", np.max(np.abs(uh - data["u"])))

    uex = sol_fn(mesh_ops.xyz[0], mesh_ops.xyz[1])

    err = uh - uex
    err = err.flatten(order="F")
    err_l2 = np.sqrt(np.dot(err, problem.mass_mat @ err))
    err_h1 = np.sqrt(np.dot(err, problem.stiff_mat @ err))
    print(f"Error (LU): {err_l2:.3e} {err_h1:.3e}")

    err = uh_v2.x.reshape(mesh_ops.Np, mesh_ops.K, order="F") - uex
    err = err.flatten(order="F")
    err_l2 = np.sqrt(np.dot(err, problem.mass_mat @ err))
    err_h1 = np.sqrt(np.dot(err, problem.stiff_mat @ err))
    print(f"Error (CG): {err_l2:.3e} {err_h1:.3e}")

    err = uh_mg - uex
    err = err.flatten(order="F")
    err_l2 = np.sqrt(np.dot(err, problem.mass_mat @ err))
    err_h1 = np.sqrt(np.dot(err, problem.stiff_mat @ err))
    print(f"Error (MG): {err_l2:.3e} {err_h1:.3e}")

    plot_2d(mesh_ops, uh)
    plot_2d(mesh_ops, uh_mg)
