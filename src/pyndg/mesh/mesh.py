import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee


LOCAL_FACE_TO_VERTEX = {
    2: np.array([[0, 1], [1, 2], [2, 0]]),  # Triangles
    3: np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]),  # Tetrahedra
}


def build_E2E_E2F(EToV):
    """
    Computes both Element-to-Element (EToE) and Element-to-Face (EToF) maps.

    Parameters:
    EToV : ndarray (K, Np) - Element-to-Vertex map.

    Returns:
    EToE : ndarray (K, Nf) - Neighboring element index for each face.
    EToF : ndarray (K, Nf) - Neighbor's local face index for each interface.
    FToE : dict - Maps a facet (tuple of vtxs) to the list of (element, local_face) pairs that share it.

    Convention:
    If EToE[k, f] == k, then face f of element k is a boundary face.
    """
    K = EToV.shape[0]  # Number of elements
    Np = EToV.shape[1]  # Number of vertices per element (3 for Tri, 4 for Tet)
    Nf = Np  # Number of faces per element (3 for Tri, 4 for Tet)
    dim = Np - 1  # Dimension of the element (2 for Tri, 3 for Tet)

    faces_idx = LOCAL_FACE_TO_VERTEX[dim]  # (Nf, dim)
    total_faces = K * Nf

    # We store: [sorted_vertex_indices, element_id, local_face_id]
    # To make sorting efficient, we sort vertex indices per face row-wise first
    all_faces = np.zeros((total_faces, Np), dtype=int)
    for f in range(Nf):
        all_faces[f::Nf, :] = np.sort(EToV[:, faces_idx[f]], axis=1)

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
            face_vertices = tuple(sorted(EToV[cell_id, faces_idx[local_face_id]]))
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
