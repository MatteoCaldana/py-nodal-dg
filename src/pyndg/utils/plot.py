import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def plot_2d(ops, u, npts=None):
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
