import pyndg.mesh
import pyndg.ops.mesh

from pathlib import Path
import scipy.io
import numpy as np

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)


def test_2d_mesh_gambit_reader():
    current_dir = Path(__file__).resolve().parent
    mesh_dir = current_dir.parent / "mesh"
    mesh_name = "circA01"
    mesh_path = mesh_dir / "gambit" / f"{mesh_name}.neu"

    mesh = pyndg.mesh.read_mesh(mesh_path)

    ref_path = current_dir / "data" / f"mesh_{mesh_name}_N3.mat"
    ref_data = scipy.io.loadmat(ref_path)

    mesh_ops = pyndg.ops.mesh.MeshOps(mesh, N=3)

    np.testing.assert_allclose(mesh_ops.xyz[0], ref_data["x"], atol=1e-15)
    np.testing.assert_allclose(mesh_ops.xyz[1], ref_data["y"], atol=1e-15)

    np.testing.assert_allclose(mesh_ops.J, ref_data["J"][0], atol=1e-15)
    np.testing.assert_allclose(mesh_ops.J_rst_xyz[0, 0], ref_data["rx"][0], atol=1e-15)
    np.testing.assert_allclose(mesh_ops.J_rst_xyz[0, 1], ref_data["ry"][0], atol=1e-15)
    np.testing.assert_allclose(mesh_ops.J_rst_xyz[1, 0], ref_data["sx"][0], atol=1e-15)
    np.testing.assert_allclose(mesh_ops.J_rst_xyz[1, 1], ref_data["sy"][0], atol=1e-15)

    # some permutations
    Npf = mesh_ops.Nfp
    for i, c in enumerate(["nx", "ny"]):
        np.testing.assert_allclose(mesh_ops.nxyz[i], ref_data[c], atol=1e-15)

    nids_c2f = (
        np.arange(mesh_ops.xyz[0].size)
        .reshape(mesh_ops.xyz.shape[1:], order="F")
        .flatten()
    )
    nids_f2c = (
        np.arange(mesh_ops.xyz[0].size)
        .reshape(mesh_ops.xyz.shape[1:])
        .flatten(order="F")
    )

    vmap_m = mesh_ops.vmap_m.reshape((Npf * 3, -1))

    vmapM = ref_data["vmapM"].astype(int) - 1
    vmapM = vmapM.reshape((Npf * 3, -1), order="F")

    np.testing.assert_equal(nids_c2f[vmap_m], vmapM)
    np.testing.assert_equal(vmap_m, nids_f2c[vmapM])

    vmap_p = mesh_ops.vmap_p.reshape((Npf * 3, -1))
    vmapP = ref_data["vmapP"].astype(int) - 1
    vmapP = vmapP.reshape((Npf * 3, -1), order="F")

    np.testing.assert_equal(nids_c2f[vmap_p], vmapP)
    np.testing.assert_equal(vmap_p, nids_f2c[vmapP])


def test_3d_mesh_gambit_reader():
    current_dir = Path(__file__).resolve().parent
    mesh_dir = current_dir.parent / "mesh"
    mesh_name = "cubeK86"
    mesh_path = mesh_dir / "gambit" / f"{mesh_name}.neu"

    mesh = pyndg.mesh.read_mesh(mesh_path)

    ref_path = current_dir / "data" / f"mesh_{mesh_name}_N3.mat"
    ref_data = scipy.io.loadmat(ref_path)

    mesh_ops = pyndg.ops.mesh.MeshOps(mesh, N=3)

    np.testing.assert_allclose(mesh_ops.xyz[0], ref_data["x"], atol=1e-15)
    np.testing.assert_allclose(mesh_ops.xyz[1], ref_data["y"], atol=1e-15)
    np.testing.assert_allclose(mesh_ops.xyz[2], ref_data["z"], atol=1e-15)

    np.testing.assert_allclose(mesh_ops.J, ref_data["J"][0], atol=6e-14)
    np.testing.assert_allclose(mesh_ops.J_rst_xyz[0, 0], ref_data["rx"][0], atol=6e-14)
    np.testing.assert_allclose(mesh_ops.J_rst_xyz[0, 1], ref_data["ry"][0], atol=6e-14)
    np.testing.assert_allclose(mesh_ops.J_rst_xyz[0, 2], ref_data["rz"][0], atol=6e-14)
    np.testing.assert_allclose(mesh_ops.J_rst_xyz[1, 0], ref_data["sx"][0], atol=6e-14)
    np.testing.assert_allclose(mesh_ops.J_rst_xyz[1, 1], ref_data["sy"][0], atol=6e-14)
    np.testing.assert_allclose(mesh_ops.J_rst_xyz[1, 2], ref_data["sz"][0], atol=6e-14)
    np.testing.assert_allclose(mesh_ops.J_rst_xyz[2, 0], ref_data["tx"][0], atol=6e-14)
    np.testing.assert_allclose(mesh_ops.J_rst_xyz[2, 1], ref_data["ty"][0], atol=6e-14)
    np.testing.assert_allclose(mesh_ops.J_rst_xyz[2, 2], ref_data["tz"][0], atol=6e-14)

    Npf = mesh_ops.Nfp
    for i, c in enumerate(["nx", "ny", "nz"]):
        np.testing.assert_allclose(mesh_ops.nxyz[i], ref_data[c], atol=6e-14)

    nids_c2f = (
        np.arange(mesh_ops.xyz[0].size)
        .reshape(mesh_ops.xyz.shape[1:], order="F")
        .flatten()
    )
    nids_f2c = (
        np.arange(mesh_ops.xyz[0].size)
        .reshape(mesh_ops.xyz.shape[1:])
        .flatten(order="F")
    )

    vmap_m = mesh_ops.vmap_m.reshape((Npf * 4, -1))

    vmapM = ref_data["vmapM"].astype(int) - 1
    vmapM = vmapM.reshape((Npf * 4, -1), order="F")

    np.testing.assert_equal(nids_c2f[vmap_m], vmapM)
    np.testing.assert_equal(vmap_m, nids_f2c[vmapM])

    vmap_p = mesh_ops.vmap_p.reshape((Npf * 4, -1))
    vmapP = ref_data["vmapP"].astype(int) - 1
    vmapP = vmapP.reshape((Npf * 4, -1), order="F")

    np.testing.assert_equal(nids_c2f[vmap_p], vmapP)
    np.testing.assert_equal(vmap_p, nids_f2c[vmapP])
