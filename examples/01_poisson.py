from pyndg.mesh import read_mesh
from pyndg.mg.smoother import jacobi, chebyshev, chebyshev_v2, bjacobi
from pyndg.mg.cg import pcg, pcg_np
from pyndg.mg.mg import build_prolongator_restrictor, mg_iter
from pyndg.ops.meshops import MeshOps
from pyndg.mesh.bc import BC
from pyndg.mesh.mesh import Mesh
from pyndg.physics.poisson import (
    Poisson,
    block_assemble_kernel,
    block_assemble_kernel_v2,
)
from pyndg.utils.plot import plot_2d

from pathlib import Path
import scipy.io
import numpy as np
import time
import jax
import jax.numpy as jnp

PATH = "/home/matteo/Documents/nodal-dg/Codes1.1/"

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)


freq = 1.0


def sol_fn(xyz):
    return np.prod(np.sin(freq * np.pi * xyz), axis=0)


def dir_fn(xyz):
    return sol_fn(xyz)


def neu_fn(xyz):
    k = freq * np.pi
    u = sol_fn(xyz)
    return tuple(
        k * u * (np.cos(k * xyz[i]) / np.sin(k * xyz[i])) for i in range(len(xyz))
    )


def rhs_fn(xyz):
    return xyz.shape[0] * freq * freq * np.pi * np.pi * sol_fn(xyz)


_A_CALLS = 0
_P_CALLS = 0


def reset_counters():
    global _A_CALLS, _P_CALLS
    _A_CALLS = 0
    _P_CALLS = 0


def triangular_mesh(l):
    x = np.linspace(0.0, 1.0, l + 1)
    y = np.linspace(0.0, 1.0, l + 1)

    xx, yy = np.meshgrid(x, y, indexing="ij")
    vxyz = np.column_stack([xx.ravel(), yy.ravel()])
    e2v = []

    def vid(i, j):
        return i * (l + 1) + j

    for i in range(l):
        for j in range(l):

            # square corners
            v0 = vid(i, j)
            v1 = vid(i + 1, j)
            v2 = vid(i + 1, j + 1)
            v3 = vid(i, j + 1)

            # split square into two triangles
            e2v.append([v0, v1, v2])
            e2v.append([v0, v2, v3])

    e2v = np.asarray(e2v, dtype=int)

    return vxyz, e2v


def main():
    N = 1

    home_path = Path(__file__).resolve().parent.parent
    mesh_path = home_path / "mesh" / "gambit" / "cubeK86.neu"

    mesh = read_mesh(mesh_path)
    mesh.face_tag = np.where(mesh.face_tag == 15, 1, mesh.face_tag)
    mesh_ops = MeshOps(mesh, N)
    params = {"penalty": 20.0, "bc_tags": {1: BC.Dirichlet}}
    problem = Poisson(params, mesh_ops)

    data = scipy.io.loadmat(PATH + f"Poisson{mesh.dim}D_N{N}.mat")

    start_time = time.perf_counter()
    problem.assemble()
    rhs = problem.assemble_rhs(rhs_fn, dir_fn, neu_fn)
    end_time = time.perf_counter()
    print(f"Assemble time: {end_time - start_time:.3f} seconds")

    print(f"Mass error: {np.max(np.abs(problem.mass_mat - data['M'])):.3e}")
    print(f"Stif error: {np.max(np.abs(problem.stiff_mat - data['A'])):.3e}")
    data["rhs"] = data["rhs"].flatten(order="C" if data["rhs"].shape[1] == 1 else "F")
    print(f"RHS error: {np.max(np.abs(rhs.flatten(order='F') - data["rhs"])):.3e}")

    print(f"Solving {rhs.size}...")
    uex = sol_fn(mesh_ops.xyz)
    uh = scipy.sparse.linalg.spsolve(problem.stiff_mat, rhs.flatten(order="F"))
    uh = uh.reshape(mesh_ops.Np, mesh_ops.K, order="F")

    err = uh - uex
    err = err.flatten(order="F")
    err_l2 = np.sqrt(np.dot(err, problem.mass_mat @ err))
    err_h1 = np.sqrt(np.dot(err, problem.stiff_mat @ err))
    print(f"Error (LU): {err_l2:.3e} {err_h1:.3e}")


def main_2d():
    N = 7

    home_path = Path(__file__).resolve().parent.parent
    mesh_path = home_path / "mesh" / "gambit" / "circA01.neu"

    mesh = read_mesh(mesh_path)

    mesh = Mesh(*triangular_mesh(400), None, None, None)
    mesh.face_tag = np.where(mesh.face_tag == 15, 1, mesh.face_tag)
    mesh_ops = MeshOps(mesh, N)
    params = {"penalty": 20.0, "bc_tags": {1: BC.Dirichlet}}
    problem = Poisson(params, mesh_ops)

    start_time = time.perf_counter()
    problem.assemble()
    end_time = time.perf_counter()
    print(f"Assemble time: {end_time - start_time:.3f} seconds")

    mesh_dims, mesh_data = mesh_ops.build_mesh_data()

    time_start = time.perf_counter()
    block_assemble_kernel(mesh_data, params["penalty"], mesh_dims)[
        1
    ].block_until_ready()
    time_end = time.perf_counter()
    print(f"JIT compile time: {time_end - time_start:.3f} seconds")
    time_start = time.perf_counter()
    reps = 3
    for _ in range(reps):
        block_assemble_kernel(mesh_data, params["penalty"], mesh_dims)[
            1
        ].block_until_ready()
    time_end = time.perf_counter()
    print(f"Block assemble time: {(time_end - time_start) / reps:.3f} seconds")
    jmass, jstiff = block_assemble_kernel(mesh_data, params["penalty"], mesh_dims)
    print(f"Mass error: {np.max(np.abs(np.array(jmass) - problem.mass)):.3e}")
    print(f"Stif error: {np.max(np.abs(np.array(jstiff) - problem.stiff)):.3e}")

    # time_start = time.perf_counter()
    # block_assemble_kernel_v2(mesh_data, params["penalty"], mesh_dims)[1].block_until_ready()
    # time_end = time.perf_counter()
    # print(f"JIT Block assemble v2 time: {time_end - time_start:.3f} seconds")

    # time_start = time.perf_counter()
    # reps = 3
    # for _ in range(reps):
    #     block_assemble_kernel_v2(mesh_data, params["penalty"], mesh_dims)[1].block_until_ready()
    # time_end = time.perf_counter()
    # print(f"Block assemble v2 time: {(time_end - time_start) / reps:.3f} seconds")

    # jmass2, jstiff2 = block_assemble_kernel_v2(mesh_data, params["penalty"], mesh_dims)

    # print(f"Mass error (v2): {np.max(np.abs(np.array(jmass2) - problem.mass)):.3e}")
    # print(f"Stif error (v2): {np.max(np.abs(np.array(jstiff2) - problem.stiff)):.3e}")

    return

    data = scipy.io.loadmat(PATH + f"Poisson2D_N{N}.mat")
    print(f"Mass error: {np.max(np.abs(problem.mass_mat - data['M'])):.3e}")
    print(f"Stif error: {np.max(np.abs(problem.stiff_mat - data['A'])):.3e}")

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
    print("Diff with ref [Matlab]", np.max(np.abs(uh - data["u"])))

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


if __name__ == "__main__":
    main()
