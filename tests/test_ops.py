import pyndg.operators.ref_elem_ops as ref_ops

import numpy as np
import scipy.io
from pathlib import Path

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)


def test_reference_matlab_1d():
    current_dir = Path(__file__).resolve().parent

    for N in range(1, 10):
        file_path = current_dir / "data" / f"reference_ops_N{N}_1d.mat"
        data = scipy.io.loadmat(file_path)

        ops = ref_ops.ReferenceElementOps(dim=1, N=N)

        np.testing.assert_allclose(ops.rst.flatten(), data["r"].flatten(), atol=1e-15)
        np.testing.assert_allclose(ops.V, data["V"], atol=1e-14)
        np.testing.assert_allclose(ops.invV, data["invV"], atol=1e-14)
        np.testing.assert_allclose(ops.Dphi[0], data["Dr"], atol=1e-13)
        np.testing.assert_allclose(ops.int_phiphi, data["M"], atol=1e-15)
        np.testing.assert_allclose(ops.int_phiDphi[0], data["S"], atol=1e-14)


def test_reference_matlab_2d():
    current_dir = Path(__file__).resolve().parent

    for N in range(1, 10):
        file_path = current_dir / "data" / f"reference_ops_N{N}_2d.mat"
        data = scipy.io.loadmat(file_path)

        ops = ref_ops.ReferenceElementOps(dim=2, N=N)

        np.testing.assert_allclose(ops.rst[0], data["r"].flatten(), atol=1e-15)
        np.testing.assert_allclose(ops.rst[1], data["s"].flatten(), atol=1e-15)
        np.testing.assert_allclose(ops.V, data["V"], atol=1e-14)
        np.testing.assert_allclose(ops.invV, data["invV"], atol=1e-15)
        np.testing.assert_allclose(ops.int_phiphi, data["MassMatrix"], atol=1e-15)
        np.testing.assert_allclose(ops.Dphi[0], data["Dr"], atol=2e-13)
        np.testing.assert_allclose(ops.Dphi[1], data["Ds"], atol=2e-13)
        np.testing.assert_allclose(ops.Dphi_weak[0], data["Drw"], atol=4e-13)
        np.testing.assert_allclose(ops.Dphi_weak[1], data["Dsw"], atol=4e-13)

        np.testing.assert_equal(ops.fmasks[0], data["Fmask"][:, 1].astype(int) - 1)
        np.testing.assert_equal(ops.fmasks[1], data["Fmask"][:, 2].astype(int) - 1)
        np.testing.assert_equal(ops.fmasks[2], data["Fmask"][:, 0].astype(int) - 1)

        Npf = ops.fmasks.shape[1]
        lift = np.hstack([ops.lift[:, Npf * d : Npf * (d + 1)] for d in [2, 0, 1]])
        np.testing.assert_allclose(lift, data["LIFT"], atol=1e-14)


def test_reference_matlab_3d():
    current_dir = Path(__file__).resolve().parent

    for N in range(1, 10):
        file_path = current_dir / "data" / f"reference_ops_N{N}_3d.mat"
        data = scipy.io.loadmat(file_path)

        ops = ref_ops.ReferenceElementOps(dim=3, N=N)

        np.testing.assert_allclose(ops.rst[0], data["r"].flatten(), atol=1e-15)
        np.testing.assert_allclose(ops.rst[1], data["s"].flatten(), atol=1e-15)
        np.testing.assert_allclose(ops.rst[2], data["t"].flatten(), atol=1e-15)
        np.testing.assert_allclose(ops.V, data["V"], atol=1e-12)
        np.testing.assert_allclose(ops.invV, data["invV"], atol=1e-15)
        np.testing.assert_allclose(ops.int_phiphi, data["MassMatrix"], atol=1e-15)
        np.testing.assert_allclose(ops.Dphi[0], data["Dr"], atol=1e-12)
        np.testing.assert_allclose(ops.Dphi[1], data["Ds"], atol=1e-12)
        np.testing.assert_allclose(ops.Dphi[2], data["Dt"], atol=1e-12)
        np.testing.assert_allclose(ops.Dphi_weak[0], data["Drw"], atol=1e-10)
        np.testing.assert_allclose(ops.Dphi_weak[1], data["Dsw"], atol=1e-10)
        np.testing.assert_allclose(ops.Dphi_weak[2], data["Dtw"], atol=1e-10)

        np.testing.assert_equal(ops.fmasks[0], data["Fmask"][:, 2].astype(int) - 1)
        np.testing.assert_equal(ops.fmasks[1], data["Fmask"][:, 3].astype(int) - 1)
        np.testing.assert_equal(ops.fmasks[2], data["Fmask"][:, 1].astype(int) - 1)
        np.testing.assert_equal(ops.fmasks[3], data["Fmask"][:, 0].astype(int) - 1)

        Npf = ops.fmasks.shape[1]
        lift = np.hstack([ops.lift[:, Npf * d : Npf * (d + 1)] for d in [3, 2, 0, 1]])
        np.testing.assert_allclose(lift, data["LIFT"], atol=1e-11)