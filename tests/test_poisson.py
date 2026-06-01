from pathlib import Path
import scipy.io
import numpy as np

from pyndg.mesh import read_mesh
from pyndg.mesh.bc import BC
from pyndg.ops.meshops import MeshOps
from pyndg.physics.poisson import Poisson

freq = 1.0


def sol_fn(xyz):
    return np.prod(np.sin(freq * np.pi * xyz), axis=0)


def bc_fn(xyz, nxyz, maps):
    bc_eval = np.zeros(xyz.shape[1:])
    bc_eval.flat[maps[1]] = sol_fn(xyz.reshape(xyz.shape[0], -1)[:, maps[1]])
    return bc_eval


def rhs_fn(xyz):
    return xyz.shape[0] * freq * freq * np.pi * np.pi * sol_fn(xyz)


def _test_allclose_sparse(A, B, atol=1e-12):
    if A.shape != B.shape:
        raise ValueError(f"Shapes do not match: {A.shape} vs {B.shape}")
    diff = A - B
    max_diff = np.max(np.abs(diff.data))
    if max_diff > atol:
        raise AssertionError(
            f"Max difference {max_diff:.3e} exceeds tolerance {atol:.3e}"
        )


def test_poisson_2d():
    for N in [1, 2, 3]:
        home_path = Path(__file__).resolve().parent.parent
        mesh_path = home_path / "mesh" / "gambit" / "circA01.neu"

        mesh = read_mesh(mesh_path)
        mesh.face_tag = np.where(mesh.face_tag == 15, 1, mesh.face_tag)
        mesh_ops = MeshOps(mesh, N)
        params = {"penalty": 20.0, "bc_tags": {1: BC.Dirichlet}}
        problem = Poisson(mesh_ops, params)

        ref_path = home_path / "tests" / "data" / f"Poisson{mesh.dim}D_N{N}.mat"
        data = scipy.io.loadmat(ref_path)

        problem.assemble()
        rhs = problem.assemble_rhs(rhs_fn, bc_fn)

        _test_allclose_sparse(problem.mass_mat, data["M"], atol=1e-13)
        _test_allclose_sparse(problem.stiff_mat, data["A"], atol=1e-11)
        np.testing.assert_allclose(rhs, data["rhs"], atol=1e-13)


def test_poisson_3d():
    for N in [1, 2, 3]:
        home_path = Path(__file__).resolve().parent.parent
        mesh_path = home_path / "mesh" / "gambit" / "cubeK86.neu"

        mesh = read_mesh(mesh_path)
        mesh_ops = MeshOps(mesh, N)
        params = {"penalty": 20.0, "bc_tags": {1: BC.Dirichlet}}
        problem = Poisson(mesh_ops, params)

        ref_path = home_path / "tests" / "data" / f"Poisson{mesh.dim}D_N{N}.mat"
        data = scipy.io.loadmat(ref_path)

        problem.assemble()
        rhs = problem.assemble_rhs(rhs_fn, bc_fn)

        _test_allclose_sparse(problem.mass_mat, data["M"], atol=1e-13)
        _test_allclose_sparse(problem.stiff_mat, data["A"], atol=1e-11)
        np.testing.assert_allclose(rhs, data["rhs"], atol=1e-13)
