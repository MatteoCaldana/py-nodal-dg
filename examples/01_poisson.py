import jax
import jax.numpy as jnp

from pyndg.mesh import read_mesh
from pyndg.mg.cg import pcg
from pyndg.mg.mg import mg_iter
from pyndg.ops.meshops import MeshOps
from pyndg.mesh.bc import BC
from pyndg.physics.poisson import Poisson
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


if __name__ == "__main__":
    N = 5

    home_path = Path(__file__).resolve().parent.parent
    mesh_path = home_path / "mesh" / "gambit" / "circA01.neu"

    mesh = read_mesh(mesh_path)
    mesh.build_bc({15: BC.Dirichlet})
    mesh_ops = MeshOps(mesh, N)
    problem = Poisson({"penalty": 20.0}, mesh_ops)
    problem.assemble()

    data = scipy.io.loadmat(PATH + f"Poisson2D_N{N}.mat")
    print(np.max(np.abs(problem.mass_mat - data["M"])))
    print(np.max(np.abs(problem.stiff_mat - data["A"])))

    rhs = problem.assemble_rhs(rhs_fn, dir_fn, neu_fn)
    print(np.max(np.abs(rhs - data["rhs"])))

    uh = scipy.sparse.linalg.spsolve(problem.stiff_mat, rhs.flatten(order="F"))

    tmp = jnp.array(problem.stiff_mat.todense())
    uh_v2 = pcg(lambda x: tmp @ x, rhs.flatten(order="F"), tol=1e-8, max_iter=10000)
    print("CG converged:", uh_v2.converged)
    print("CG iterations:", uh_v2.iterations)

    diag = jnp.diag(tmp)
    M_fn = lambda x: x / diag
    uh_v2 = pcg(lambda x: tmp @ x, rhs.flatten(order="F"), M=M_fn, tol=1e-8)
    print("CG converged:", uh_v2.converged)
    print("CG iterations:", uh_v2.iterations)

    bdiag_inv = np.linalg.inv(problem.stiff[: mesh.K])
    x_shape = (bdiag_inv.shape[0], bdiag_inv.shape[2], 1)
    M_fn = lambda x: (bdiag_inv @ x.reshape(x_shape)).reshape(-1)
    uh_v2 = pcg(lambda x: tmp @ x, rhs.flatten(order="F"), M=M_fn, tol=1e-8)
    print("CG converged:", uh_v2.converged)
    print("CG iterations:", uh_v2.iterations)

    print("Diff with ref (PCG)", np.max(np.abs(uh_v2.x - uh)))

    ##########################################################################
    # multigrid

    mesh_ops_coarse = MeshOps(mesh, N - 1)
    problem_coarse = Poisson({"penalty": 20.0}, mesh_ops_coarse)
    problem_coarse.assemble()

    Icf = mesh_ops_coarse.ref_elem_ops.build_interp_mat(mesh_ops.ref_elem_ops.rst)
    Ifc = mesh_ops.ref_elem_ops.build_interp_mat(mesh_ops_coarse.ref_elem_ops.rst)

    Np_f = mesh_ops.Np
    K = mesh_ops.K
    Np_c = mesh_ops_coarse.Np

    P = Icf
    R = (
        mesh_ops_coarse.ref_elem_ops.int_phiphi_inv
        @ Icf.T
        @ mesh_ops.ref_elem_ops.int_phiphi
    )

    uh = uh.reshape(mesh_ops.Np, mesh_ops.K, order="F")

    A = problem.stiff_mat.todense()
    Ac = np.zeros((Np_c * K, Np_c * K))
    for i in range(K):
        for j in range(K):
            A_block = A[i * Np_f : (i + 1) * Np_f, j * Np_f : (j + 1) * Np_f]
            Ac_block = R @ (A_block @ P)
            Ac[i * Np_c : (i + 1) * Np_c, j * Np_c : (j + 1) * Np_c] = Ac_block

    uh_mg = mg_iter(
        problem.stiff_mat, rhs.flatten(order="F"), np.zeros(uh.size), None, P, R, Ac
    ).reshape(mesh_ops.Np, mesh_ops.K, order="F")

    plot_2d(mesh_ops, uh)
    plot_2d(mesh_ops, uh_mg)

    # - plot projection to make sure it makes sense
    # - mg iteration
    # - mg solver
    # - mg as preconditioner for CG

    ##########################################################################

    uh = uh.reshape(mesh_ops.Np, mesh_ops.K, order="F")
    print("Diff with ref", np.max(np.abs(uh - data["u"])))

    uex = sol_fn(mesh_ops.xyz[0], mesh_ops.xyz[1])
    err = uh - uex

    ref_mass = mesh_ops.ref_elem_ops.int_phiphi
    err_l2 = np.sqrt(np.sum((err.T @ ref_mass) * mesh_ops.J[:, None] * err.T))

    err = err.flatten(order="F")
    err_l2 = np.sqrt(np.dot(err, problem.mass_mat @ err))
    err_h1 = np.sqrt(np.dot(err, problem.stiff_mat @ err))
    print(f"Error: {err_l2:.3e} {err_h1:.3e}")

    # plot_2d(mesh_ops, uh)

    # TODO:
    # - multigrid solver
    # - jax sparse utils
    # - convergence test (p and h)
    # - 3D, correctness check + benchmarks
    # - jax.scan assembly loop
    # - h-refinement grid
    # - local h-refinement ?
