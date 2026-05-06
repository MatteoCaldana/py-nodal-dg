import gmsh
import numpy as np

from pyndg.mesh import build_e2e_e2f


def _f2t_to_face_mat(e2v, f2t):
    """
    Parameters
    ----------
    e2v : np.ndarray
        Element-to-vertex connectivity matrix of shape (K, 3).
    f2t : dict
        (Canonical) face-to-tag mapping.

    Returns
    -------
    face_mat: np.ndarray
        Matrix of shape (K, 3), tag for each local face of each element`
    """

    _, _, f2e = build_e2e_e2f(e2v)
    K = e2v.shape[0]
    face_mat = np.zeros((K, 3), dtype=int)
    for canonical_face, tag in f2t.items():
        elems = f2e[canonical_face]
        for elem_id, local_face_id in elems:
            face_mat[elem_id, local_face_id] = tag
    return face_mat


def _gmsh_extractor_callback(surface):
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    vertex_matrix = coords.reshape((-1, 3))[:, :2]

    # Tag-to-Index Lookup Array
    max_node_tag = np.max(node_tags)
    tag_to_idx = np.zeros(max_node_tag + 1, dtype=int)
    tag_to_idx[node_tags] = np.arange(node_tags.size, dtype=int)

    # get the elements of the surface
    # the function returns a list of elements, but in our case is one
    _, _, node_connec_list = gmsh.model.mesh.getElements(2, surface)
    node_connec = node_connec_list[0].astype(int)

    # Map all tags to indices at once using the lookup array
    ev_matrix = tag_to_idx[node_connec].reshape((-1, 3))

    # Boundary Tag Data
    boundary_data = {}
    physical_groups = gmsh.model.getPhysicalGroups(1)

    for dim, p_tag in physical_groups:
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, p_tag)
        for entity in entities:
            # Again, take the 0th index of the returned list
            _, _, line_nodes_list = gmsh.model.mesh.getElements(1, entity)
            if len(line_nodes_list) == 0:
                continue

            line_nodes = line_nodes_list[0].reshape((-1, 2))
            for nodes in line_nodes:
                # Map tags to indices
                v1, v2 = tag_to_idx[nodes[0]], tag_to_idx[nodes[1]]
                face_key = tuple(sorted((int(v1), int(v2))))
                boundary_data[face_key] = p_tag

    return ev_matrix, vertex_matrix, boundary_data


def generate_rectangular_mesh(
    length: float, height: float, n: int, structured: bool = False
):
    gmsh.initialize()
    gmsh.model.add("Rectangle")

    L, H, n = length, height, n
    # Characteristic length for unstructured mode
    lc = (L**2 + H**2) ** 0.5 / (n - 1) / 2**0.5

    # --- 1. Geometry definition ---
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(L, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(L, H, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, H, 0, lc)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s1 = gmsh.model.geo.addPlaneSurface([cl])

    # --- 2. Apply Mesh Strategy ---
    if structured:
        # Force a regular grid split into triangles
        for line in [l1, l2, l3, l4]:
            gmsh.model.geo.mesh.setTransfiniteCurve(line, n)
        gmsh.model.geo.mesh.setTransfiniteSurface(s1, "Left")

    gmsh.model.geo.synchronize()

    # Physical Groups
    gmsh.model.addPhysicalGroup(2, [s1], 100000)
    gmsh.model.addPhysicalGroup(1, [l4], 100001)  # Left
    gmsh.model.addPhysicalGroup(1, [l2], 100002)  # Right
    gmsh.model.addPhysicalGroup(1, [l1], 100003)  # Bottom
    gmsh.model.addPhysicalGroup(1, [l3], 100004)  # Top

    gmsh.model.mesh.generate(2)
    e2v, vxy, boundary_data = _gmsh_extractor_callback(s1)
    gmsh.finalize()
    return e2v, vxy, _f2t_to_face_mat(e2v, boundary_data)


def generate_mesh_1d(xmin, xmax, K, mesh_per):
    Nv = K + 1
    h = (xmax - xmin) / K
    pert_scale = mesh_per * h
    VX = np.linspace(xmin, xmax, Nv)

    if mesh_per > 0:
        pert = (np.random.rand(Nv - 2) - 0.5) * pert_scale
        VX[1:-1] += pert
    hK = VX[1:] - VX[:-1]
    return Nv, VX, hK
