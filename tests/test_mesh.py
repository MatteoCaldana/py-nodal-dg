import pyndg.mesh

from pathlib import Path
import scipy.io
import numpy as np


def test_2d_mesh_gambit_reader():
    current_dir = Path(__file__).resolve().parent
    mesh_dir = current_dir.parent / "mesh"
    mesh_name = "circA01"
    mesh_path = mesh_dir / "gambit" / f"{mesh_name}.neu"

    mesh = pyndg.mesh.read_mesh(mesh_path)

    ref_path = current_dir / "data" / f"mesh_{mesh_name}_N3.mat"
    ref_data = scipy.io.loadmat(ref_path)

    np.testing.assert_allclose(mesh.vxyz[:, 0], ref_data["VX"].ravel(), atol=1e-15)
    np.testing.assert_allclose(mesh.vxyz[:, 1], ref_data["VY"].ravel(), atol=1e-15)
    np.testing.assert_equal(mesh.e2v, ref_data["EToV"].astype(int) - 1)
    np.testing.assert_equal(mesh.e2e, ref_data["EToE"].astype(int) - 1)
    np.testing.assert_equal(mesh.e2f, ref_data["EToF"].astype(int) - 1)


def test_3d_mesh_gambit_reader():
    current_dir = Path(__file__).resolve().parent
    mesh_dir = current_dir.parent / "mesh"
    mesh_name = "cubeK86"
    mesh_path = mesh_dir / "gambit" / f"{mesh_name}.neu"

    mesh = pyndg.mesh.read_mesh(mesh_path)

    ref_path = current_dir / "data" / f"mesh_{mesh_name}_N3.mat"
    ref_data = scipy.io.loadmat(ref_path)

    np.testing.assert_allclose(mesh.vxyz[:, 0], ref_data["VX"].ravel(), atol=1e-15)
    np.testing.assert_allclose(mesh.vxyz[:, 1], ref_data["VY"].ravel(), atol=1e-15)
    np.testing.assert_allclose(mesh.vxyz[:, 2], ref_data["VZ"].ravel(), atol=1e-15)
    np.testing.assert_equal(mesh.e2v, ref_data["EToV"].astype(int) - 1)
    np.testing.assert_equal(mesh.e2e, ref_data["EToE"].astype(int) - 1)
    np.testing.assert_equal(mesh.e2f, ref_data["EToF"].astype(int) - 1)


def test_2d_mesh_periodic():
    pass


def test_1d_mesh():
    pass
