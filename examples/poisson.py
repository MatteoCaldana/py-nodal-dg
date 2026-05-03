from pyndg.mesh import read_mesh
from pyndg.ops.mesh import MeshOps
from pyndg.physics.poisson import Poisson

from pathlib import Path
import scipy.io
import numpy as np

PATH = "/home/matteo/Documents/nodal-dg/Codes1.1/"

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

if __name__ == "__main__":
    N = 10
    freq = 2

    home_path = Path(__file__).resolve().parent.parent
    mesh_path = home_path / "mesh" / "gambit" / "circA01.neu"

    mesh = read_mesh(mesh_path)
    mesh_ops = MeshOps(mesh, N)

    data = scipy.io.loadmat(PATH + f"Poisson2D_N{N}.mat")

    problem = Poisson({"penalty": 20.0}, mesh_ops)
    problem.assemble()

    print(np.max(np.abs(problem.mass_mat - data["M"])))
    print(np.max(np.abs(problem.stiff_mat - data["A"])))
