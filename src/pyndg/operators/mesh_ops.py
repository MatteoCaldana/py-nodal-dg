from pyndg.operators.ref_elem_ops import ReferenceElementOps

import numpy as np


class MeshOperators:
    def __init__(self, mesh, N):
        self.mesh = mesh
        self.N = N
        self.dim = mesh.dim

        self.ref_elem_ops = ReferenceElementOps(mesh.dim, N)
        self._build()

    def _build(self):
        self._compute_barycentric_coordinates()
        self._compute_nodes_coordiantes()
        self._compute_geometric_factors()
        self._compute_normals()
        self._compute_nodal_maps()

    def _compute_nodes_coordiantes(self):
        bcc = self.ref_elem_ops.bary_coord  # (dim + 1, Np)
        fvxyz = self.mesh.vxyz[self.mesh.EToV]  # (K, Np, dim)
        self.xyz = 0.5 * np.einsum("ir,kid->drk", bcc, fvxyz)  # (dim, Np, K)

    def _compute_face_coordinates(self):
        """
        Compute the coordinates of the face nodes for each element.

        self.fxyz has shape (dim, Nfp * Nfaces, K)
        """
        self.fxyz = self.xyz[self.ref_elem_ops.fmasks.flat, :]

    def _compute_geometric_factors(self):
        # derivative xyz to rst (dim, dim, Np, K)
        # each row is the transpose of the gradient of one coord wrt rst
        self.J_xyz_rst = np.empty((self.dim, *self.xyz.shape), dtype=np.float64)
        for d1 in range(self.dim):
            for d2 in range(self.dim):
                self.J_xyz_rst[d1, d2] = self.ref_elem_ops.Dphi[d2] @ self.xyz[d1]

        J_mat = self.J_xyz_rst.transpose(2, 3, 0, 1)  # (Np, K, dim, dim)
        self.J = np.linalg.det(J_mat)  # (Np, K)
        J_inv = np.linalg.inv(J_mat)  # (Np, K, dim, dim)

        # derivative rst to xyz (dim, dim, Np, K)
        # each row is the transpose of the gradient of one coord wrt xyz
        self.J_rst_xyz = J_inv.transpose(2, 3, 0, 1)  # (dim, dim, Np, K)

        # TODO: the jacobian should be constant element-wise for affine mappings
        # check if it is constant and average it to enhance quality

    def _compute_normals(self):
        """
        Compute the outward normal vectors at the face nodes for each element
        applying the chain rule to transform the reference normals into physical space.

        self.nxyz has shape (dim, Nfp * Nfaces, K)
        self.surf_J has shape (Nfp * Nfaces, K)
        """
        Nfaces = self.dim + 1
        fmasks = self.ref_elem_ops.fmasks.flat
        self.sJ_rst_xyz = self.J_rst_xyz[:, :, fmasks.flat, :]
        sJ_rst_xyz = self.sJ_rst_xyz.reshape(
            self.dim, self.dim, Nfaces, self.Nfp, self.mesh.K
        )

        self.nxyz = np.empty((self.dim, Nfaces, self.Nfp, self.K))
        for d in range(self.dim):
            self.nxyz[:, d + 1, :, :] = -sJ_rst_xyz[d, :, d + 1, :, :]
        self.nxyz[:, 0, :, :] = sJ_rst_xyz[:, :, 0, :, :].sum(axis=1)

        n_norm = np.linalg.norm(self.nxyz, axis=0)
        self.nxyz = self.nxyz / n_norm
        self.surf_J = n_norm * self.J[fmasks.flat, :]

    def _compute_nodal_maps(self):
        """
        Compute the minus and plus node maps for each face of each element.
        The minus is the current element, the plus is the neighboring element across the face.

        Note: The Npf x dim row size of the map is to contract during the application of the 'lift' operator.
        """
        K = self.mesh.K
        Np = self.mesh.Np
        nodeids = np.arange(Np * K).reshape(Np, K)
        fmasks = self.ref_elem_ops.fmasks

        self.vmap_m = np.empty((self.dim, self.Nfp, K), dtype=int)
        for elem_id in range(self.K):
            self.vmap_m[:, :, elem_id] = nodeids[fmasks.flat, elem_id]

        # node map for the plus side, initialized as self-referential
        # usual convention that equality means boundary condition
        self.vid_p = self.vmap_m.copy()

        # reference element length, should be the shortest edge of each element,
        # we set for an approximation assuming elements are not too distorted
        bbox = np.max(self.mesh.VXYZ, axis=0) - np.min(self.mesh.VXYZ, axis=0)
        refd = (np.prod(bbox) / self.K) ** (1 / self.dim)
        print("Reference element length scale (approx):", refd)

        for cid in range(self.K):
            for lfid in range(self.dim + 1):
                ncid, nlfid = self.EToE[cid, lfid], self.EToF[cid, lfid]
                # node ids for the current face
                vid_m = self.vmap_m[lfid, :, cid]
                # node ids for the neighboring face
                vid_p = self.vmap_m[nlfid, :, ncid]
                # to find out if there is any permutation, we compare coordinates
                xyz_m = self.xyz.reshape(self.dim, -1)[:, vid_m]
                xyz_p = self.xyz.reshape(self.dim, -1)[:, vid_p]
                d2 = sum(
                    [
                        np.subtract.outer(xyz_m[d], xyz_p[d]) ** 2
                        for d in range(self.dim)
                    ]
                )
                id_m, id_p = np.where(np.sqrt(d2) < 1e-8 * refd)
                self.vmap_p[lfid, id_m, cid] = vid_p[id_p]

        # TODO: compute vmap_p, taking into account periodicity and boundary conditions
        print(
            "WARNING: vmap_p periodicity and boundary conditions not implemented yet."
        )
