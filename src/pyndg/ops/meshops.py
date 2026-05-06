from pyndg.mesh.bc import BC
from pyndg.ops.refelem import REF_NORMALS, ReferenceElementOps

import numpy as np


class MeshOps:
    def __init__(self, mesh, N):
        self.mesh = mesh
        self.N = N
        self.dim = mesh.dim
        self.K = mesh.K
        self.Nf = self.dim + 1

        self.ref_elem_ops = ReferenceElementOps(mesh.dim, N)
        self.Np = self.ref_elem_ops.Np
        self.Nfp = self.ref_elem_ops.Nfp

        self._build()

    def _build(self):
        self._compute_nodes_coordiantes()
        self._compute_face_coordinates()
        self._compute_geometric_factors()
        self._compute_normals()
        self._compute_nodal_maps()
        self._compute_bc_nodal_maps()

    def _compute_nodes_coordiantes(self):
        bcc = self.ref_elem_ops.bary_coord  # (dim + 1, Np)
        fvxyz = self.mesh.vxyz[self.mesh.e2v]  # (K, Np, dim)
        self.xyz = 0.5 * np.einsum("ir,kid->drk", bcc, fvxyz)  # (dim, Np, K)

    def _compute_face_coordinates(self):
        """
        Compute the coordinates of the face nodes for each element.

        self.fxyz has shape (dim, Nfp * Nfaces, K)
        """
        self.fxyz = self.xyz[:, self.ref_elem_ops.fmasks.flat, :]

    def _compute_geometric_factors(self):
        # derivative xyz to rst (dim, dim, Np, K)
        # each row is the transpose of the gradient of one coord wrt rst
        J_xyz_rst = np.empty((self.dim, self.dim, self.K), dtype=np.float64)
        for d1 in range(self.dim):
            for d2 in range(self.dim):
                tmp = self.ref_elem_ops.Dphi[d2] @ self.xyz[d1]
                diff = np.max(np.max(tmp, axis=0) - np.min(tmp, axis=0))
                assert diff < 1e-12, f"Jacobian is not constant within elements: {diff}"
                J_xyz_rst[d1, d2] = np.mean(tmp, axis=0)

        J_mat = J_xyz_rst.transpose(2, 0, 1)  # (K, dim, dim)
        # determinant of jacobian for each element
        self.J = np.linalg.det(J_mat)  # (K)

        # derivative rst to xyz (dim, dim, K)
        # each row is the transpose of the gradient of one coord wrt xyz
        self.J_rst_xyz = np.linalg.inv(J_mat).transpose(1, 2, 0)

    def _compute_normals(self):
        """
        Compute the outward normal vectors at the face nodes for each element
        applying the chain rule to transform the reference normals into physical space.

        self.nxyz has shape (dim, Nfp * Nfaces, K)
        self.surf_J has shape (Nfp * Nfaces, K)
        """

        nxyz = np.empty((self.dim, self.Nf, self.K))
        ref_normals = REF_NORMALS[self.dim - 1]
        # We multiply the inverse transposed jacobian with the reference normal
        # k is the index of the element
        # f is the face index
        # i is the jacobian row index
        # d is the jacobian column index, spatial dimension index
        nxyz = np.einsum("idk,fi->dfk", self.J_rst_xyz, ref_normals)
        self.fscale = np.linalg.norm(nxyz, axis=0)
        nxyz = nxyz / self.fscale

        self.sJ = self.fscale * self.J[None, :]

        # due to the fact that face information is retained in matrices of
        # shape (Nf * Nfp, K) to apply the lift operator, we cannot use
        # easily the broadcasting to apply nxyz of shape (dim, Nf, K),
        # so we reshape it to (dim, Nf * Nfp, K)
        self.nxyz = np.repeat(nxyz, self.Nfp, axis=1)

    def _compute_nodal_maps(self):
        """
        Compute the minus and plus node maps for each face of each element.
        The minus is the current element, the plus is the neighboring element across the face.

        Note: The Npf x dim row size of the map is to contract during the application of the 'lift' operator.
        """
        nodeids = np.arange(self.Np * self.K).reshape(self.Np, self.K)
        fmasks = self.ref_elem_ops.fmasks

        self.vmap_m = np.empty((self.Nf, self.Nfp, self.K), dtype=int)
        for elem_id in range(self.K):
            self.vmap_m[:, :, elem_id].flat = nodeids[fmasks.flat, elem_id]

        # node map for the plus side, initialized as self-referential
        # usual convention that equality means boundary condition
        self.vmap_p = self.vmap_m.copy()

        # reference element length, should be the shortest edge of each element,
        # we set for an approximation assuming elements are not too distorted
        bbox = np.max(self.mesh.vxyz, axis=0) - np.min(self.mesh.vxyz, axis=0)
        refd = (np.prod(bbox) / self.K) ** (1 / self.dim)
        for cid in range(self.K):
            for lfid in range(self.dim + 1):
                ncid = self.mesh.e2e[cid, lfid]
                nlfid = self.mesh.e2f[cid, lfid]
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
                assert id_m.size == self.Nfp and id_p.size == self.Nfp
                self.vmap_p[lfid, id_m, cid] = vid_p[id_p]

        # TODO: compute vmap_p, taking into account periodicity and boundary conditions
        print(
            "WARNING: vmap_p periodicity and boundary conditions not implemented yet."
        )

    def _compute_bc_nodal_maps(self):
        self.bc_nodes_maps = {}
        for tag in range(1, len(BC)):
            map = np.zeros_like(self.vmap_m, dtype=bool)
            cell_ids, local_face_ids = np.where(self.mesh.face_tag == tag)
            map[local_face_ids, :, cell_ids] = True
            self.bc_nodes_maps[tag] = map.reshape(-1, self.K)
