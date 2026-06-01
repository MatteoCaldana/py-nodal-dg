import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

# Global cache dictionary to store global_triangles connectivity matrices.
# Key format: (id(ops), npts)
_TRIANGULATION_CACHE = {}


def plot_2d(ops, u, npts=None, ax=None):
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

    # We use id(ops) as a safe way to identify the specific object instance
    # without attaching attributes to it directly.
    cache_key = (id(ops), npts)

    if cache_key in _TRIANGULATION_CACHE:
        global_triangles = _TRIANGULATION_CACHE[cache_key]
    else:
        # 2. Compute reference triangulation ONCE
        ref_tri = mtri.Triangulation(ref_rst[0], ref_rst[1])
        ref_triangles = ref_tri.triangles  # Shape: (N_tri_ref, 3)

        # 3. Build global connectivity (triangles)
        offsets = np.arange(K) * n_pts_per_elem

        global_triangles = (
            ref_triangles[np.newaxis, :, :] + offsets[:, np.newaxis, np.newaxis]
        ).reshape(-1, 3)

        # Store in the external global cache
        _TRIANGULATION_CACHE[cache_key] = global_triangles

    # 4. Flatten coordinate and solution arrays
    x_flat = x_interp.flatten(order="F")
    y_flat = y_interp.flatten(order="F")
    u_flat = u_interp.flatten(order="F")

    # 5. Create the global triangulation and plot in one call
    global_tri = mtri.Triangulation(x_flat, y_flat, global_triangles)

    if ax is None:
        plt.figure()
        ax = plt.gca()

    im = ax.tricontourf(global_tri, u_flat, levels=20)
    # plt.colorbar(im, ax=ax)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
