import matplotlib

from pyndg.mesh import read_mesh
from pyndg.ops.mesh import MeshOps
from pyndg.mesh.bc import BC
from pyndg.physics.poisson import Poisson

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


def plot_2d(ops, u, npts=None):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import numpy as np

    x = ops.xyz[0]
    y = ops.xyz[1]

    if u.shape != x.shape:
        u = u.reshape(x.shape, order="F")

    if npts is None:
        npts = ops.N * 2

    # 1. Build equispaced grid and interpolation matrix
    ref_rst = ops.ref_elem_ops.build_equi_grid(2, npts)
    interp = ops.ref_elem_ops.build_interp_mat(ref_rst)

    # Interpolate to plotting nodes
    u_interp = interp @ u
    x_interp = interp @ x
    y_interp = interp @ y

    n_pts_per_elem, K = x_interp.shape

    # 2. Compute reference triangulation ONCE
    ref_tri = mtri.Triangulation(ref_rst[0], ref_rst[1])
    ref_triangles = ref_tri.triangles  # Shape: (N_tri_ref, 3)

    # 3. Build global connectivity (triangles)
    # Create an offset array for each element: [0, n_pts, 2*n_pts, ..., (K-1)*n_pts]
    offsets = np.arange(K) * n_pts_per_elem

    # Broadcast offsets to reference triangles to create global indices
    # Resulting shape: (K, N_tri_ref, 3), then flatten to (-1, 3)
    global_triangles = (
        ref_triangles[np.newaxis, :, :] + offsets[:, np.newaxis, np.newaxis]
    ).reshape(-1, 3)

    # 4. Flatten coordinate and solution arrays
    # Using order='F' (Fortran) is crucial because your arrays are shaped (n_pts, K)
    # This guarantees we flatten element-by-element (column-by-column)
    x_flat = x_interp.flatten(order="F")
    y_flat = y_interp.flatten(order="F")
    u_flat = u_interp.flatten(order="F")

    # 5. Create the global triangulation and plot in one call
    global_tri = mtri.Triangulation(x_flat, y_flat, global_triangles)

    plt.figure()
    plt.tricontourf(global_tri, u_flat, levels=30)
    plt.show()


if __name__ == "__main__":
    N = 6

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

    plot_2d(mesh_ops, uh)

    # TODO:
    # - convergence test (p and h)
    # - square grid
    # - multigrid solver
    # - h-refinement grid
    # - 3D
