import pyndg.mesh
import pyndg.ops.mesh

from pathlib import Path
import scipy.io


def test_2d_mesh_gambit_reader():
    current_dir = Path(__file__).resolve().parent
    mesh_dir = current_dir.parent / "mesh"
    mesh_name = "circA01"
    mesh_path = mesh_dir / "gambit" / f"{mesh_name}.neu"

    mesh = pyndg.mesh.read_mesh(mesh_path)

    ref_path = current_dir / "data" / f"mesh_{mesh_name}_N3.mat"
    ref_data = scipy.io.loadmat(ref_path)

    mesh_ops = pyndg.ops.mesh.MeshOps(mesh, N=3)


def test_3d_mesh_gambit_reader():
    pass
