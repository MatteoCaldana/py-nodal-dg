import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from pathlib import Path

from pyndg.mesh.reader import read_gmsh_file_2d, mesh_reader_gambit

LOCAL_FACE_TO_VERTEX = {
    2: np.array(
        [
            [0, 1],
            [1, 2],
            [2, 0],
        ]
    ),  # Triangles
    3: np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [0, 2, 3],
        ]
    ),  # Tetrahedra
}


def build_e2e_e2f(e2v):
    """
    Computes both Element-to-Element (EToE) and Element-to-Face (EToF) maps.

    Parameters:
    e2v : ndarray (K, Np) - Element-to-Vertex map.

    Returns:
    EToE : ndarray (K, Nf) - Neighboring element index for each face.
    EToF : ndarray (K, Nf) - Neighbor's local face index for each interface.
    FToE : dict - Maps a facet (tuple of vtxs) to the list of (element, local_face) pairs that share it.

    Convention:
    If EToE[k, f] == k, then face f of element k is a boundary face.
    """
    K = e2v.shape[0]  # Number of elements
    Np = e2v.shape[1]  # Number of vertices per element (3 for Tri, 4 for Tet)
    Nf = Np  # Number of faces per element (3 for Tri, 4 for Tet)
    dim = Np - 1  # Dimension of the element (2 for Tri, 3 for Tet)

    faces_idx = LOCAL_FACE_TO_VERTEX[dim]  # (Nf, dim)
    total_faces = K * Nf

    # We store: [sorted_vertex_indices, element_id, local_face_id]
    # To make sorting efficient, we sort vertex indices per face row-wise first
    all_faces = np.zeros((total_faces, dim), dtype=int)
    for f in range(Nf):
        all_faces[f::Nf, :] = np.sort(e2v[:, faces_idx[f]], axis=1)

    # Create a unique ID for each face for sorting
    # We use lexsort on the vertex columns to find matches
    # This brings faces with the same vertices next to each other
    indices = np.lexsort(all_faces.T)
    sorted_faces = all_faces[indices]

    # Initialize as self-referential
    EToE = np.arange(K).repeat(Nf).reshape(K, Nf)
    EToF = np.tile(np.arange(Nf), (K, 1))

    # Find matches in the sorted list
    # Match occurs if row i and row i+1 are identical
    matches = np.all(sorted_faces[1:] == sorted_faces[:-1], axis=1)

    # Identify indices of the matching face pairs
    idx1 = indices[:-1][matches]
    idx2 = indices[1:][matches]

    # Map back to element and local face IDs
    k1, f1 = idx1 // Nf, idx1 % Nf
    k2, f2 = idx2 // Nf, idx2 % Nf

    # Face f1 of element k1 touches face f2 of element k2
    # Face f2 of element k2 touches face f1 of element k1
    EToE[k1, f1] = k2
    EToE[k2, f2] = k1
    EToF[k1, f1] = f2
    EToF[k2, f2] = f1

    # map from face (defined by its vertices) to (element, local_face) pairs
    FToE = {}
    for cell_id in range(K):
        for local_face_id in range(Nf):
            face_vertices = tuple(sorted(e2v[cell_id, faces_idx[local_face_id]]))
            if face_vertices not in FToE:
                FToE[face_vertices] = []
            FToE[face_vertices].append((cell_id, local_face_id))

    return EToE, EToF, FToE


def list_connectivity_edges(EToE):
    """
    Parameters:
    EToE : ndarray (K, Nf) - Element-to-Element connectivity map

    Returns:
    cell_id12 : ndarray (M, 2) - Each row is a unique edge

    Description:
    Lists all unique edges in the connectivity graph defined by EToE.
    Each edge is represented as a pair of element indices (cell_id1, cell_id2).
    The order of the pair is sorted (cell_id1 < cell_id2) to avoid duplicates.
    """
    # number of elements
    K = EToE.shape[0]
    cell_id1 = np.repeat(np.arange(K), EToE.shape[1])
    cell_id12 = np.stack([cell_id1, EToE.flat], axis=1)
    # sort elements within each row
    cell_id12 = np.sort(cell_id12, axis=1)
    # sort rows lexicographycally
    cell_id12 = cell_id12[np.lexsort(cell_id12.T[::-1])]
    # remove connection of element with self
    cell_id12 = cell_id12[~np.all(cell_id12 == cell_id12[:, [0]], axis=1)]
    # remove duplicates
    mask = np.any(cell_id12[1:] != cell_id12[:-1], axis=1)
    cell_id12 = np.vstack([cell_id12[0], cell_id12[1:][mask]])
    return cell_id12.astype(np.int32)


def get_connectivity_edge_id(K, connectivity_edges, cell_id1, cell_id2):
    """
    Parameters:
    K : int - Number of elements
    connectivity_edges : ndarray (M, 2) - List of unique edges (cell_id1, cell_id2)
    cell_id1, cell_id2 : int - IDs of the two cells to find the coupling ID for

    Returns:
    int - The edge ID corresponding to the edge between cell_id1 and cell_id2.

    Description:
    Given two cell IDs, this function returns the corresponding edge ID from the connectivity_edges list.
    The function assumes that cell_id1 and cell_id2 are connected (i.e., they share a face).
    """
    assert cell_id1 != cell_id2
    if cell_id1 < cell_id2:
        rev = False
        a, b = cell_id1, cell_id2
    else:
        rev = True
        a, b = cell_id2, cell_id1

    keys = connectivity_edges[:, 0] * K + connectivity_edges[:, 1]
    target = a * K + b

    idx = int(np.searchsorted(keys, target))

    n_couples = connectivity_edges.shape[0]
    if idx < n_couples and keys[idx] == target:
        return idx + n_couples if rev else idx
    else:
        raise ValueError(f"These cells are not connected {cell_id1} {cell_id2}")


def reorder_cells(connectivity_edges, K):
    """
    Parameters:
    connectivity_edges : ndarray (M, 2) - List of unique edges (cell_id1, cell_id2)
    K : int - Number of elements

    Returns:
    ndarray (K,) - The new ordering of the original indices

    Description:
    This function computes a new ordering of the elements (cells) to minimize the bandwidth of the connectivity graph defined by the edges.
    It uses the Reverse Cuthill-McKee algorithm.
    """
    # Create a sparse symmetric matrix
    rows, cols = connectivity_edges[:, 0], connectivity_edges[:, 1]
    data = np.ones(len(rows))
    adj = csr_matrix((data, (rows, cols)), shape=(K, K))
    # Ensure it is symmetric (undirected graph)
    adj = adj + adj.T

    def get_bandwidth(matrix):
        # Find indices of non-zero elements
        r, c = matrix.nonzero()
        if len(r) == 0:
            return 0
        return np.max(np.abs(r - c))

    # Compute Original Bandwidth
    old_bw = get_bandwidth(adj)

    # Compute Reverse Cuthill-McKee Reordering
    # perm is the new ordering of the original indices
    perm = reverse_cuthill_mckee(adj)

    # Apply Reordering to get New Bandwidth
    # We reorder rows and columns: adj[perm, :][:, perm]
    new_adj = adj[perm, :][:, perm]
    new_bw = get_bandwidth(new_adj)

    print("Old bandwidth:", old_bw)
    print("New bandwidth:", new_bw)
    return perm


def read_mesh(filename: Path):
    readers = {".msh": read_gmsh_file_2d, ".neu": mesh_reader_gambit}
    fields = readers[filename.suffix](filename)
    return Mesh(*fields)


def check_mesh_orientation_2d(VXY, e2v):
    """
    Checks orientation of a 2D triangular mesh.

    Parameters
    ----------
    VXY  : ndarray (N, 2)   Vertex coordinates.
    e2v : ndarray (K, 3)   Element-to-vertex connectivity.

    Returns
    -------
    inverted   : ndarray    Indices of CW-oriented (inverted) triangles.
    degenerate : ndarray    Indices of flat/degenerate triangles (area ≈ 0).
    """
    p1, p2, p3 = VXY[e2v[:, 0]], VXY[e2v[:, 1]], VXY[e2v[:, 2]]

    # 2D cross product = twice the signed area (positive = CCW)
    cr1 = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1])
    cr2 = (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1])
    val = cr1 - cr2

    tol = 1e-12 * (np.max(np.abs(val)) or 1.0)
    inverted = np.where(val < -tol)[0]
    degenerate = np.where(np.abs(val) <= tol)[0]

    if len(inverted) == 0 and len(degenerate) == 0:
        print("All triangles are properly oriented (CCW).")
    else:
        print(
            f"WARNING: {len(inverted)} inverted, {len(degenerate)} degenerate triangles."
        )

    return inverted, degenerate


def check_mesh_orientation_3d(vxyz, e2v):
    """
    Checks orientation of a 3D tetrahedral mesh.

    Parameters
    ----------
    vxyz : ndarray (N, 3)   Vertex coordinates.
    e2v : ndarray (K, 4)   Element-to-vertex connectivity.

    Returns
    -------
    inverted   : ndarray    Indices of inverted tetrahedra (vol < 0).
    degenerate : ndarray    Indices of flat/degenerate tetrahedra (vol ≈ 0).
    """
    p1, p2, p3, p4 = (
        vxyz[e2v[:, 0]],
        vxyz[e2v[:, 1]],
        vxyz[e2v[:, 2]],
        vxyz[e2v[:, 3]],
    )

    a, b, c = p2 - p1, p3 - p1, p4 - p1

    # Scalar triple product = 6 * signed volume
    val = (
        a[:, 0] * (b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1])
        - a[:, 1] * (b[:, 0] * c[:, 2] - b[:, 2] * c[:, 0])
        + a[:, 2] * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
    )

    tol = 1e-12 * (np.max(np.abs(val)) or 1.0)
    inverted = np.where(val < -tol)[0]
    degenerate = np.where(np.abs(val) <= tol)[0]

    if len(inverted) == 0 and len(degenerate) == 0:
        print("All tetrahedra are properly oriented.")
    else:
        print(
            f"WARNING: {len(inverted)} inverted, {len(degenerate)} degenerate tetrahedra."
        )

    return inverted, degenerate


def check_mesh_orientation_1d(VX, e2v):
    """
    Checks orientation of a 1D edge mesh.

    Parameters
    ----------
    VX   : ndarray (N,)    Vertex coordinates.
    e2v : ndarray (K, 2)  Element-to-vertex connectivity.

    Returns
    -------
    inverted   : ndarray   Indices of inverted edges (v1 < v0).
    degenerate : ndarray   Indices of degenerate edges (v0 == v1).
    """
    # Signed length: positive means correctly oriented (v0 → v1)
    val = VX[e2v[:, 1]] - VX[e2v[:, 0]]

    tol = 1e-12 * (np.max(np.abs(val)) or 1.0)
    inverted = np.where(val < -tol)[0]
    degenerate = np.where(np.abs(val) <= tol)[0]

    if len(inverted) == 0 and len(degenerate) == 0:
        print("All edges are properly oriented.")
    else:
        print(f"WARNING: {len(inverted)} inverted, {len(degenerate)} degenerate edges.")

    return inverted, degenerate


def check_mesh_orientation(V, e2v):
    check_mesh_orientation_fns = [
        check_mesh_orientation_1d,
        check_mesh_orientation_2d,
        check_mesh_orientation_3d,
    ]
    return check_mesh_orientation_fns[V.shape[1] - 1](V, e2v)


class Mesh:
    def __init__(self, vxyz, K, Nv, e2v, b_faces, per_b2b, per_bf2f):
        self.vxyz = vxyz
        self.K = K
        self.Nv = Nv
        self.dim = vxyz.shape[1]
        self.e2v = e2v
        self.b_faces = b_faces
        self.per_b2b = per_b2b
        self.per_bf2f = per_bf2f

        check_mesh_orientation(vxyz, e2v)

        self.e2e, self.e2f, self.f2e = build_e2e_e2f(e2v)

        self.connectivity_edges = list_connectivity_edges(self.e2e)
        reorder_cells(self.connectivity_edges, self.K)

    def plot(self, show_elem_id=True, show_vtx_id=True):
        if self.dim == 2:
            self._plot_2d(show_elem_id, show_vtx_id)
        else:
            print("Plotting not implemented for dim > 2.")

    def _plot_2d(self, show_elem_id=True, show_vtx_id=True):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        import matplotlib.colors as mcolors

        _, ax = plt.subplots(figsize=(10, 8))

        # Extract all triangle edges from e2v
        all_edges = []
        for tri in self.e2v:
            # tri contains indices of the 3 vertices
            pts = self.vxyz[tri]
            all_edges.append((pts[0], pts[1]))
            all_edges.append((pts[1], pts[2]))
            all_edges.append((pts[2], pts[0]))

        # Plot mesh in light gray to make BCs stand out
        mesh_lc = LineCollection(
            all_edges, colors="lightgray", linewidths=0.5, alpha=1.0
        )
        ax.add_collection(mesh_lc)

        # Plot Boundary Edges by Tag
        color_cycle = list(mcolors.TABLEAU_COLORS.values())
        for i, (tag, faces) in enumerate(self.b_faces.items()):
            bc_edges = []
            for node_pair in faces:
                p1 = self.vxyz[node_pair[0]]
                p2 = self.vxyz[node_pair[1]]
                bc_edges.append((p1, p2))

            color = color_cycle[i % len(color_cycle)]
            bc_lc = LineCollection(
                bc_edges,
                colors=color,
                linewidths=2.5,
                label=f"Tag ID={tag}, type={tag}",
            )
            ax.add_collection(bc_lc)

        if show_elem_id:
            for eid, tri in enumerate(self.e2v):
                pts = self.vxyz[tri].mean(axis=0)
                plt.text(pts[0], pts[1], f"{eid}")

        if show_vtx_id:
            for i in range(self.vxyz.shape[0]):
                plt.text(self.vxyz[i, 0], self.vxyz[i, 1], f"{i}", color="r")

        ax.set_aspect("equal")
        ax.autoscale()
        plt.title(f"2D Mesh: K={self.K}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
        plt.tight_layout()
        plt.show()
