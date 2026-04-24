import numpy as np


def read_gmsh_file(filename):
    with open(filename, "r") as fp:
        data = fp.read().split("\n")
    assert data[0] == "$MeshFormat"
    assert data[1][0] == "2", "Support only version 2"
    assert data[2] == "$EndMeshFormat"
    assert data[3] == "$Nodes"
    Nv = int(data[4])
    VXY = np.empty((Nv, 2))
    for i in range(Nv):
        line_tokens = data[i + 5].split(" ")
        VXY[i, 0] = float(line_tokens[1])
        VXY[i, 1] = float(line_tokens[2])
    assert data[Nv + 5] == "$EndNodes"
    assert data[Nv + 6] == "$Elements"
    Ne = int(data[Nv + 7])
    Nb = 0
    line_tokens = data[Nv + 8].split(" ")
    b_faces = {}
    g_to_utag = {}
    while line_tokens[1] == "1":
        Nb += 1
        gftag = int(line_tokens[4])
        uftag = int(line_tokens[3])
        if gftag in g_to_utag:
            assert g_to_utag[gftag] == uftag
        else:
            g_to_utag[gftag] = uftag
        if uftag not in b_faces:
            b_faces[uftag] = []
        b_faces[uftag].append(int(line_tokens[-2]))
        b_faces[uftag].append(int(line_tokens[-1]))
        line_tokens = data[Nv + 8 + Nb].split(" ")
    for key in b_faces:
        b_faces[key] = np.array(b_faces[key]).reshape((-1, 2))
    K = Ne - Nb
    EToV = np.empty((K, 3), dtype=np.int32)
    for i in range(K - 1):
        EToV[i, 0] = int(line_tokens[-3])
        EToV[i, 1] = int(line_tokens[-2])
        EToV[i, 2] = int(line_tokens[-1])
        line_tokens = data[Nv + 8 + Nb + i + 1].split(" ")
    EToV[-1, 0] = int(line_tokens[-3])
    EToV[-1, 1] = int(line_tokens[-2])
    EToV[-1, 2] = int(line_tokens[-1])
    assert data[Nv + 8 + Nb + K] == "$EndElements"
    PerBToB, PerBFToF = {}, {}
    if data[Nv + 8 + Nb + K + 1] == "$Periodic":
        Np = int(data[Nv + 8 + Nb + K + 2])
        idx = 0
        for i in range(Np):
            line_tokens, idx = data[Nv + 8 + Nb + K + 3 + idx].split(" "), idx + 1
            n_ent, idx = int(data[Nv + 8 + Nb + K + 3 + idx]), idx + 1
            if line_tokens[0] == "1":
                bf1 = g_to_utag[int(line_tokens[1])]
                bf2 = g_to_utag[int(line_tokens[2])]
                nface = b_faces[bf1].shape[0]
                PerBToB[bf1] = bf2
                f2f = np.zeros((nface, nface))
                bf1_vlist = b_faces[bf1]
                bf2_vlist = b_faces[bf2]
                p2p = {}
                for j in range(n_ent):
                    line_tokens, idx = (
                        data[Nv + 8 + Nb + K + 3 + idx].split(" "),
                        idx + 1,
                    )
                    p1, p2 = int(line_tokens[0]), int(line_tokens[1])
                    p2p[p1] = p2
                    ind1 = np.where((bf1_vlist[:, 0] == p1) | (bf1_vlist[:, 1] == p1))
                    ind2 = np.where((bf2_vlist[:, 0] == p2) | (bf2_vlist[:, 1] == p2))
                    for i1 in ind1[0]:
                        for i2 in ind2[0]:
                            f2f[i1, i2] = f2f[i1, i2] + 1
                f2flist = np.empty((nface, 1), dtype=np.int32)
                for f1 in range(nface):
                    f2 = np.where(f2f[f1, :] == 2)[0]
                    f2flist[f1] = f2 * (
                        (p2p[bf1_vlist[f1, 0]] == bf1_vlist[f2, 0]) * 2 - 1
                    )
                PerBFToF[bf1] = f2flist
            else:
                idx += n_ent
    EToV -= 1
    ax, ay = VXY[:, 0][EToV[:, 0]], VXY[:, 1][EToV[:, 0]]
    bx, by = VXY[:, 0][EToV[:, 1]], VXY[:, 1][EToV[:, 1]]
    cx, cy = VXY[:, 0][EToV[:, 2]], VXY[:, 1][EToV[:, 2]]
    D = (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)
    i = np.where(D < 0)[0]
    if i.size:
        EToV = EToV[i, [0, 2, 1]]
    for k in b_faces:
        b_faces[k] -= 1
    for k in PerBFToF:
        PerBFToF[k] -= 1
    return VXY, K, Nv, EToV, b_faces, PerBToB, PerBFToF


def _compute_bfaces_from_bctag(BCTag, EToV):
    BFaces = {}
    face_to_v = np.array([[0, 1], [1, 2], [0, 2]])
    unique_tags = np.unique(BCTag)
    unique_tags = unique_tags[unique_tags != 0]
    for tag in unique_tags:
        elements, local_faces = np.where(BCTag == tag)
        v_indices = face_to_v[local_faces]
        BFaces[int(tag)] = EToV[elements[:, None], v_indices]
    return BFaces


_GAMBIT_BC_MAP = {
    "In": 11,
    "Out": 12,
    "Wall": 13,
    "Cyl": 14,
    "Dirichlet": 15,
    "Neuman": 16,
    "Slip": 17,
}


def mesh_reader_gambit(file_name):

    with open(file_name, "r") as fid:
        lines = fid.readlines()

    # 1. Parse Dimensions (Node count and Element count)
    # Typically found on line 7 (index 6)
    dims = np.fromstring(lines[6], sep=" ", dtype=int)
    Nv, K, NGRPS, NBSETS, NDFCD, NDFVL = dims[0:6]
    print("Gambit Mesh info:")
    print(f"  # of vertices: {Nv}")
    print(f"  # of elements: {K}")
    print(f"  # of element groups: {NGRPS}")
    print(f"  # of boundary sets: {NBSETS}")
    print(f"  # of coordinate dimensions: {NDFCD}")
    print(f"  # of velocity components: {NDFVL}")

    # 2. Read Node Coordinates
    VXY = np.zeros((Nv, NDFCD), dtype=np.float64)
    for i in range(Nv):
        # Coordinates start after the header (line 10 / index 9)
        data = np.fromstring(lines[9 + i], sep=" ")
        assert data[0] == i + 1, "Node IDs should be sequential and start from 1"
        VXY[i] = data[1 : 1 + NDFCD]

    # 3. Read Element Connectivity
    # Locate the start of the ELEMENTS/CELLS section
    start_line = 9 + Nv + 2
    EToV = np.zeros((K, NDFCD + 1), dtype=int)
    for k in range(K):
        data = np.fromstring(lines[start_line + k], sep=" ", dtype=int)
        # data[3:6] contains the 3 vertex IDs
        # Adjust 1-based Gambit indexing to 0-based Python indexing
        assert data[0] == k + 1, "Element IDs should be sequential and start from 1"
        EToV[k, :] = data[3 : 3 + NDFCD + 1] - 1

    if NBSETS == 0:
        # No boundary conditions specified, return early
        return VXY, K, Nv, EToV, None, None, None

    # Fill Boundary Condition array
    bc_type = np.zeros((K, NDFCD + 1), dtype=int)
    for idx, line in enumerate(lines):
        bc_flag = 0

        for key, flag in _GAMBIT_BC_MAP.items():
            if key in line:
                bc_flag = flag
                break

        if bc_flag != 0:
            # Skip the header lines of the BC section
            curr = idx + 1
            while "ENDOFSECTION" not in lines[curr]:
                tmp_id = np.fromstring(lines[curr], sep=" ", dtype=int)

                # tmp_id[0]: Element ID (1-based)
                # tmp_id[2]: Local Face ID (1-based)
                elem_idx = tmp_id[0] - 1
                face_idx = tmp_id[2] - 1

                bc_type[elem_idx, face_idx] = bc_flag
                curr += 1

    return VXY, K, Nv, EToV, _compute_bfaces_from_bctag(bc_type, EToV), None, None
