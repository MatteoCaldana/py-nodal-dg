from pyndg.mesh import read_mesh
from pyndg.ops.meshops import MeshOps
from pyndg.mesh.bc import BC
from pyndg.physics.poisson import Poisson
from pyndg.utils.plot import plot_2d

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse

freq = 1.0


def sol_fn(xyz):
    return np.prod(np.sin(freq * np.pi * xyz), axis=0)


def rhs_fn(xyz):
    return xyz.shape[0] * freq * freq * np.pi * np.pi * sol_fn(xyz)


def bc_fn(xyz, nxyz, maps):
    bc_eval = np.zeros(xyz.shape[1:])
    bc_eval.flat[maps[1]] = sol_fn(xyz.reshape(xyz.shape[0], -1)[:, maps[1]])
    return bc_eval


def main():
    N = 5

    home_path = Path(__file__).resolve().parent.parent
    mesh_path = home_path / "mesh" / "gambit" / "circA01.neu"

    mesh = read_mesh(mesh_path)
    mesh.face_tag = np.where(mesh.face_tag == 15, 1, mesh.face_tag)
    mesh_ops = MeshOps(mesh, N)
    params = {"penalty": 20.0, "bc_tags": {1: BC.Dirichlet}}
    problem = Poisson(mesh_ops, params)

    problem.assemble()
    rhs = problem.assemble_rhs(rhs_fn, bc_fn)
    uh = scipy.sparse.linalg.spsolve(problem.stiff_mat, rhs.flatten(order="F"))
    uh = uh.reshape(mesh_ops.Np, mesh_ops.K, order="F")

    plot_2d(mesh_ops, uh)
    plt.show()


if __name__ == "__main__":
    main()
