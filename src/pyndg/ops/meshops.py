from pyndg.mesh.mesh import LOCAL_FACE_TO_VERTEX
from pyndg.ops.refelem import REF_NORMALS, ReferenceElementOps
from pyndg.mesh.bc import BC
import pyndg.backend as bkd

from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
import operator
from functools import reduce


class MeshDims(NamedTuple):
    N: int
    Np: int
    Nfp: int
    dim: int
    Nf: int
    K: int


class MeshData(NamedTuple):
    # mesh
    e2e: jax.Array
    e2f: jax.Array
    face_tag: jax.Array
    connectivity_edges: jax.Array
    eid2ef: jax.Array
    # reference element data
    rst: jax.Array
    V: jax.Array
    invV: jax.Array
    int_phiphi: jax.Array
    avg_phi: jax.Array
    int_phiphi_inv: jax.Array
    Dphi: jax.Array
    int_phiDphi: jax.Array
    Dphi_weak: jax.Array
    int_DphiDphi: jax.Array
    bary_coord: jax.Array
    fmasks: jax.Array
    face_int_phiphi: jax.Array
    lift: jax.Array
    # mesh ops data
    xyz: jax.Array
    fxyz: jax.Array
    J: jax.Array
    J_rst_xyz: jax.Array
    nxyz: jax.Array
    sJ: jax.Array
    fscale: jax.Array
    vmap_m: jax.Array
    vmap_p: jax.Array
    bc_maps: dict[int, jax.Array]


def apply_bc_maps(ops, bc_tags_map):
    assert (0 not in bc_tags_map) or (
        bc_tags_map[0] == BC.NONE
    ), "Tag 0 is reserved for internal faces and must be mapped to BC.NONE"
    # map mesh tag to BC type
    bc_tags_map[0] = BC.NONE
    # map BC type to mesh tag
    bc_tags_map_rev = {}
    for tag, bc in bc_tags_map.items():
        if bc not in bc_tags_map_rev:
            bc_tags_map_rev[bc] = []
        bc_tags_map_rev[bc].append(tag)

    empty_bc = np.zeros((ops.Nfp * ops.Nf, ops.K), dtype=bool)
    bc_type_map = {}
    for bc_type in BC:
        if bc_type == BC.NONE:
            continue

        n_tags = bc_tags_map_rev.get(bc_type, [])
        bc_type_map[bc_type] = reduce(
            operator.or_, [ops.bc_maps[tag] for tag in n_tags], empty_bc
        )

    return bc_type_map


def find_permutation(a, b):
    pa = np.argsort(a, kind="stable")
    pb = np.argsort(b, kind="stable")

    p = np.empty_like(pb)
    p[pb] = pa

    return p


@jax.jit
def div(J_rst_xyz, Dphi, u):
    """
    Dphi      : (dim, Np, Np)         stacked reference derivative matrices
    u         : (dim, Np, K)          vector field  (rank-1)
              | (dim, dim, Np, K)     tensor field  (rank-2), divergence taken on axis 0
    J_rst_xyz : (dim, dim, K)         J[i,j,k] = d(r_i)/d(x_j) at element k

    Returns
    -------
    div_u : (Np, K)          rank-1 input → scalar field
          | (dim, Np, K)     rank-2 input → vector field  (one div per row)
    """
    Np, K = u.shape[-2], u.shape[-1]
    u4 = u.reshape(u.shape[0], -1, Np, K)  # (j, l, b, k),  l=1 for vectors
    out = jnp.einsum("ijk, ipb, jlbk -> lpk", J_rst_xyz, Dphi, u4)  # (l, p, k)
    return out.reshape(u.shape[1:])  # (Np,K) or (dim,Np,K)


@jax.jit
def grad(J_rst_xyz, Dphi, u):
    """
    Dphi      : (dim, Np, Np)      stacked reference derivative matrices
    u         : (Np, K)            scalar field
              | (dim, Np, K)       vector field
    J_rst_xyz : (dim, dim, K)      J[i,j,k] = d(r_i)/d(x_j) at element k

    Returns
    -------
    scalar → u_xyz : (dim, Np, K)          u_xyz[j]  = ∂u/∂x_j
    vector → u_xyz : (dim, dim, Np, K) u_xyz[i, j]   = ∂u_i/∂x_j  (Jacobian)
    """
    if u.ndim == 2:
        # scalar field
        return jnp.einsum("ijk, ipb, bk -> jpk", J_rst_xyz, Dphi, u)

    if u.ndim == 3:
        # vector field: vmap scalar grad over components (axis 0)
        return jax.vmap(lambda ui: grad(J_rst_xyz, Dphi, ui), in_axes=0)(u)

    raise ValueError(f"Unsupported u shape: {u.shape}")


@jax.jit
def de(J_rst_xyz, Dphi, u, axis):
    """
    J_rst_xyz : (dim, dim, K)
    Dphi      : (dim, Np, Np)
    u         : (Np, K)
    Returns   : (Np, K)   ∂u/∂x_{axis}
    """
    # Fix j=axis, contract over i and b only
    J_col = J_rst_xyz[:, axis, :]  # (dim, K)
    return jnp.einsum("ik, ipb, bk -> pk", J_col, Dphi, u)


@jax.jit
def curl(J_rst_xyz, Dphi, u):
    if u.ndim == 2:
        # 2D curl: curl(u) = du_y/dx - du_x/dy
        du = grad(J_rst_xyz, Dphi, u)
        return jnp.stack([du[1], -du[0]], axis=0)
    if u.ndim == 3:
        if u.shape[0] == 3:
            # 3D curl: curl(u) = (du_z/dy - du_y/dz, du_x/dz - du_z/dx, du_y/dx - du_x/dy)
            du = grad(J_rst_xyz, Dphi, u)
            return jnp.stack(
                [
                    du[2, 1] - du[1, 2],
                    du[0, 2] - du[2, 0],
                    du[1, 0] - du[0, 1],
                ],
                axis=0,
            )
        elif u.shape[0] == 2:
            # curl of a 2D vector field du_y/dx - du_x/dy
            duydx = de(J_rst_xyz, Dphi, u[1], 0)
            duxdy = de(J_rst_xyz, Dphi, u[0], 1)
            return duydx - duxdy

    raise ValueError(f"Unsupported u shape: {u.shape}")


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
        print("MeshOps initialized")
        print(f"  Total nodes: {self.Np * self.K}")
        print(f"  Face nodes:  {self.Nfp * self.Nf * self.K}")

    def _build(self):
        print("   Building mesh nodes...")
        self._compute_nodes_coordiantes()
        print("   Building face nodes...")
        self._compute_face_coordinates()
        print("   Building geometric factors...")
        self._compute_geometric_factors()
        print("   Building normals...")
        self._compute_normals()
        print("   Building nodal maps...")
        self._compute_nodal_maps()
        print("   Building BC nodal maps...")
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

        new_shape = (self.Nf, self.Nfp, self.K)
        self.fscale = np.repeat(self.fscale, self.Nfp, axis=0).reshape(new_shape)
        self.sJ = np.repeat(self.sJ, self.Nfp, axis=0).reshape(new_shape)

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
        permutations_cache = {}
        lf2v = LOCAL_FACE_TO_VERTEX[self.dim]
        for cid in range(self.K):
            for lfid in range(self.dim + 1):
                ncid = self.mesh.e2e[cid, lfid]
                nlfid = self.mesh.e2f[cid, lfid]
                # node ids for the current face
                vid_m = self.vmap_m[lfid, :, cid]
                # node ids for the neighboring face
                vid_p = self.vmap_m[nlfid, :, ncid]

                vf = self.mesh.e2v[cid, lf2v[lfid]]
                nvf = self.mesh.e2v[ncid, lf2v[nlfid]]
                key = (lfid, nlfid, *find_permutation(vf, nvf))

                if key in permutations_cache:
                    # if we already computed the permutation for this pair of faces, we can reuse it
                    id_p = permutations_cache[key]
                else:
                    # to find out if there is any permutation, we compare coordinates
                    xyz_m = self.xyz.reshape(self.dim, -1)[:, vid_m]
                    xyz_p = self.xyz.reshape(self.dim, -1)[:, vid_p]
                    d2 = sum(
                        [
                            np.subtract.outer(xyz_m[d], xyz_p[d]) ** 2
                            for d in range(self.dim)
                        ]
                    )
                    id_m, id_p = np.where(np.sqrt(d2) < 1e-8 * np.sqrt(d2.max()))
                    assert id_m.size == self.Nfp and id_p.size == self.Nfp
                    assert (id_m == np.arange(self.Nfp)).all()
                    permutations_cache[key] = id_p

                self.vmap_p[lfid, :, cid] = vid_p[id_p]

        # TODO: compute vmap_p, taking into account periodicity and boundary conditions
        print("WARNING: periodic boundary conditions not implemented yet.")

    def _compute_bc_nodal_maps(self):
        self.bc_maps = {}

        if self.mesh.face_tag is None:
            print("WARNING: mesh has no face tags, no BC will be applied.")
            return

        tags = np.sort(np.unique(self.mesh.face_tag))
        assert tags[0] == 0, "No tags. Tag 0 is reserved for untagged faces."
        for tag in tags[1:]:
            map = np.zeros_like(self.vmap_m, dtype=bool)
            cell_ids, local_face_ids = np.where(self.mesh.face_tag == tag)
            map[local_face_ids, :, cell_ids] = True
            self.bc_maps[tag] = map.reshape(-1, self.K)

    def build_mesh_data(self):
        return MeshDims(
            N=self.N,
            Np=self.Np,
            Nfp=self.Nfp,
            dim=self.dim,
            Nf=self.Nf,
            K=self.K,
        ), MeshData(
            e2e=jnp.array(self.mesh.e2e, dtype=jnp.int32),
            e2f=jnp.array(self.mesh.e2f, dtype=jnp.int32),
            eid2ef=jnp.array(self.mesh.eid2ef, dtype=jnp.int32),
            face_tag=jnp.array(self.mesh.face_tag, dtype=jnp.int32),
            connectivity_edges=jnp.array(self.mesh.connectivity_edges, dtype=jnp.int32),
            rst=jnp.array(self.ref_elem_ops.rst, dtype=bkd.jnp_prec),
            V=jnp.array(self.ref_elem_ops.V, dtype=bkd.jnp_prec),
            invV=jnp.array(self.ref_elem_ops.invV, dtype=bkd.jnp_prec),
            int_phiphi=jnp.array(self.ref_elem_ops.int_phiphi, dtype=bkd.jnp_prec),
            avg_phi=jnp.array(self.ref_elem_ops.avg_phi, dtype=bkd.jnp_prec),
            int_phiphi_inv=jnp.array(
                self.ref_elem_ops.int_phiphi_inv, dtype=bkd.jnp_prec
            ),
            Dphi=jnp.array(self.ref_elem_ops.Dphi, dtype=bkd.jnp_prec),
            int_phiDphi=jnp.array(self.ref_elem_ops.int_phiDphi, dtype=bkd.jnp_prec),
            Dphi_weak=jnp.array(self.ref_elem_ops.Dphi_weak, dtype=bkd.jnp_prec),
            int_DphiDphi=jnp.array(self.ref_elem_ops.int_DphiDphi, dtype=bkd.jnp_prec),
            bary_coord=jnp.array(self.ref_elem_ops.bary_coord, dtype=bkd.jnp_prec),
            fmasks=jnp.array(self.ref_elem_ops.fmasks, dtype=jnp.int32),
            face_int_phiphi=jnp.array(
                self.ref_elem_ops.face_int_phiphi, dtype=bkd.jnp_prec
            ),
            lift=jnp.array(self.ref_elem_ops.lift, dtype=bkd.jnp_prec),
            xyz=jnp.array(self.xyz, dtype=bkd.jnp_prec),
            fxyz=jnp.array(self.fxyz, dtype=bkd.jnp_prec),
            J=jnp.array(self.J, dtype=bkd.jnp_prec),
            J_rst_xyz=jnp.array(self.J_rst_xyz, dtype=bkd.jnp_prec),
            nxyz=jnp.array(self.nxyz, dtype=bkd.jnp_prec),
            sJ=jnp.array(self.sJ, dtype=bkd.jnp_prec),
            fscale=jnp.array(self.fscale.reshape(-1, self.K), dtype=bkd.jnp_prec),
            vmap_m=jnp.array(self.vmap_m, dtype=jnp.int32),
            vmap_p=jnp.array(self.vmap_p, dtype=jnp.int32),
            bc_maps={
                tag: jnp.where(map.flatten())[0] for tag, map in self.bc_maps.items()
            },
        )
