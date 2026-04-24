from pyndg.core.basis import pnleg, jacobi_gq, jacobi_gl, vandermonde_1d, dmatrix_1d
from pyndg.core.basis import nodes_2d, xy_to_rs, vandermonde_2d, dmatrices_2d
from pyndg.core.basis import nodes_3d, xyz_to_rst, vandermonde_3d, dmatrices_3d
from pyndg.core.la import solve_lu
import pyndg.backend as bkd

import pytest
import numpy as np
import scipy.io
from pathlib import Path

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)


class TestPnLegExtended:
    @pytest.fixture
    def sample_x(self):
        # Using a range of values to ensure robustness
        return np.linspace(-1, 1, 20, dtype=bkd.np_prec)

    def test_p0_to_p5(self, sample_x):
        x = sample_x

        # P0(x) = 1
        np.testing.assert_allclose(pnleg(x, 0), np.ones_like(x))

        # P1(x) = x
        np.testing.assert_allclose(pnleg(x, 1), x)

        # P2(x) = 1/2 * (3x^2 - 1)
        np.testing.assert_allclose(pnleg(x, 2), 0.5 * (3 * x**2 - 1))

        # P3(x) = 1/2 * (5x^3 - 3x)
        np.testing.assert_allclose(pnleg(x, 3), 0.5 * (5 * x**3 - 3 * x))

        # P4(x) = 1/8 * (35x^4 - 30x^2 + 3)
        np.testing.assert_allclose(pnleg(x, 4), (1 / 8) * (35 * x**4 - 30 * x**2 + 3))

        # P5(x) = 1/8 * (63x^5 - 70x^3 + 15x)
        np.testing.assert_allclose(
            pnleg(x, 5), (1 / 8) * (63 * x**5 - 70 * x**3 + 15 * x)
        )


class TestJacobiGQ:

    def test_n1(self):
        # Case n=1 (2 nodes)
        nodes, weights = jacobi_gq(0, 0, 1)

        sqrt3 = np.sqrt(3, dtype=bkd.np_prec)
        expected_nodes = np.array([-1 / sqrt3, 1 / sqrt3])
        expected_weights = np.array([1.0, 1.0])

        np.testing.assert_allclose(nodes, expected_nodes)
        np.testing.assert_allclose(weights, expected_weights)
        print("TestJacobiGQ::test_n1, error:", np.max(np.abs(nodes - expected_nodes)))

    def test_n2(self):
        # Case n=2 (3 nodes)
        nodes, weights = jacobi_gq(0, 0, 2)

        sqrt06 = np.sqrt(0.6, dtype=bkd.np_prec)
        expected_nodes = np.array([-sqrt06, 0.0, sqrt06])
        expected_weights = np.array([5, 8, 5], dtype=bkd.np_prec) / 9

        np.testing.assert_allclose(
            nodes, expected_nodes, atol=np.finfo(bkd.np_prec).eps * 2
        )
        np.testing.assert_allclose(weights, expected_weights)
        print("TestJacobiGQ::test_n2, error:", np.max(np.abs(nodes - expected_nodes)))

    def test_n3(self):
        # Case n=3 (4 nodes)
        nodes, weights = jacobi_gq(0, 0, 3)

        # Analytical values for Gauss-Legendre n=3
        val1 = np.sqrt((3 - 2 * np.sqrt(6 / 5, dtype=bkd.np_prec)) / 7)
        val2 = np.sqrt((3 + 2 * np.sqrt(6 / 5, dtype=bkd.np_prec)) / 7)

        expected_nodes = np.array([-val2, -val1, val1, val2])

        # Analytical weights
        w1 = (18 + np.sqrt(30, dtype=bkd.np_prec)) / 36
        w2 = (18 - np.sqrt(30, dtype=bkd.np_prec)) / 36
        expected_weights = np.array([w2, w1, w1, w2], dtype=bkd.np_prec)

        np.testing.assert_allclose(nodes, expected_nodes)
        np.testing.assert_allclose(weights, expected_weights)
        print("TestJacobiGQ::test_n3, error:", np.max(np.abs(nodes - expected_nodes)))


def test_reference_matlab_1d():
    current_dir = Path(__file__).resolve().parent

    for N in range(1, 10):
        file_path = current_dir / "data" / f"reference_ops_N{N}_1d.mat"
        data = scipy.io.loadmat(file_path)

        r = jacobi_gl(0, 0, N)
        V = vandermonde_1d(N, r)
        invV = solve_lu(V, np.eye(*V.shape, dtype=bkd.np_prec))
        Dr = dmatrix_1d(N, r, V)
        invVT = solve_lu(V.T, np.eye(*V.shape, dtype=bkd.np_prec))
        M = solve_lu(V.T, invVT.T).T
        invM = solve_lu(M, np.eye(*M.shape, dtype=bkd.np_prec))
        S = np.matmul(M, Dr)

        np.testing.assert_allclose(r, data["r"].flatten(), atol=1e-15)
        np.testing.assert_allclose(V, data["V"], atol=1e-14)
        np.testing.assert_allclose(invV, data["invV"], atol=1e-14)
        np.testing.assert_allclose(Dr, data["Dr"], atol=1e-12)
        np.testing.assert_allclose(M, data["M"], atol=1e-15)
        np.testing.assert_allclose(invM, data["invM"], atol=1e-15)
        np.testing.assert_allclose(S, data["S"], atol=1e-14)


def test_reference_matlab_2d():
    current_dir = Path(__file__).resolve().parent

    for N in range(1, 10):
        file_path = current_dir / "data" / f"reference_ops_N{N}_2d.mat"
        data = scipy.io.loadmat(file_path)

        x, y = nodes_2d(N)
        r, s = xy_to_rs(x, y)
        V = vandermonde_2d(N, r, s)
        invV = solve_lu(V, np.eye(*V.shape, dtype=bkd.np_prec))
        MassMatrix = invV.T @ invV
        Dr, Ds = dmatrices_2d(N, r, s, V)

        np.testing.assert_allclose(r, data["r"].flatten(), atol=1e-15)
        np.testing.assert_allclose(s, data["s"].flatten(), atol=1e-15)
        np.testing.assert_allclose(V, data["V"], atol=1e-13)
        np.testing.assert_allclose(invV, data["invV"], atol=1e-15)
        np.testing.assert_allclose(MassMatrix, data["MassMatrix"], atol=1e-15)
        np.testing.assert_allclose(Dr, data["Dr"], atol=1e-12)
        np.testing.assert_allclose(Ds, data["Ds"], atol=1e-12)


def test_reference_matlab_3d():
    current_dir = Path(__file__).resolve().parent

    for N in range(1, 10):
        file_path = current_dir / "data" / f"reference_ops_N{N}_3d.mat"
        data = scipy.io.loadmat(file_path)

        print(f"Testing N={N}...")

        x, y, z = nodes_3d(N)
        r, s, t = xyz_to_rst(x, y, z)
        V = vandermonde_3d(N, r, s, t)
        invV = solve_lu(V, np.eye(*V.shape, dtype=bkd.np_prec))
        MassMatrix = invV.T @ invV
        Dr, Ds, Dt = dmatrices_3d(N, r, s, t, V)

        np.testing.assert_allclose(r, data["r"].flatten(), atol=1e-15)
        np.testing.assert_allclose(s, data["s"].flatten(), atol=1e-15)
        np.testing.assert_allclose(t, data["t"].flatten(), atol=1e-15)
        np.testing.assert_allclose(V, data["V"], atol=1e-12)
        np.testing.assert_allclose(invV, data["invV"], atol=1e-15)
        np.testing.assert_allclose(MassMatrix, data["MassMatrix"], atol=1e-15)
        np.testing.assert_allclose(Dr, data["Dr"], atol=1e-12)
        np.testing.assert_allclose(Ds, data["Ds"], atol=1e-12)
        np.testing.assert_allclose(Dt, data["Dt"], atol=1e-12)
