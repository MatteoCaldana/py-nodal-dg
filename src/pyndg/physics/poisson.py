from typing import NamedTuple
from xml.etree.ElementPath import ops

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse
import operator
from functools import reduce

from pyndg.mesh.bc import BC
from pyndg.ops.meshops import MeshData


class PoissonData(NamedTuple):
    penalty: float
    stiff: jax.Array
    stiff_coupling: jax.Array
    mass: jax.Array


@jax.jit(static_argnames=["dims"])
def _block_assemble_kernel(mesh_data, penalty, dims):
    class CarryData(NamedTuple):
        mdata: MeshData
        pdata: PoissonData

    data = CarryData(
        mdata=mesh_data,
        pdata=PoissonData(
            penalty=penalty,
            stiff=jnp.zeros((dims.K, dims.Np, dims.Np)),
            stiff_coupling=jnp.zeros((dims.K, dims.Nf, dims.Np, dims.Np)),
            mass=jnp.zeros((dims.K, dims.Np, dims.Np)),
        ),
    )

    Np = dims.Np

    def _dirichlet(mesh_data, lnx, lny, ncid, gtau, mmE, Dn1, Fm1, Fm2):
        return gtau * mmE - mmE @ Dn1 - Dn1.T @ mmE, jnp.zeros((Np, Np))

    def _neumann(mesh_data, lnx, lny, ncid, gtau, mmE, Dn1, Fm1, Fm2):
        return jnp.zeros((Np, Np)), jnp.zeros((Np, Np))

    def _interior(mesh_data, lnx, lny, ncid, gtau, mmE, Dn1, Fm1, Fm2):
        Jmat = mesh_data.J_rst_xyz[:, :, ncid]
        Dx2 = Jmat[0, 0] * mesh_data.Dphi[0] + Jmat[1, 0] * mesh_data.Dphi[1]
        Dy2 = Jmat[0, 1] * mesh_data.Dphi[0] + Jmat[1, 1] * mesh_data.Dphi[1]
        Dn2 = lnx * Dx2 + lny * Dy2

        loc_stiff = jnp.zeros((Np, Np))
        loc_stiff = loc_stiff.at[:, Fm2].add(-0.5 * gtau * mmE[:, Fm1])
        loc_stiff = loc_stiff.at[Fm1, :].add(
            -0.5 * mmE[jnp.ix_(Fm1, Fm1)] @ Dn2[Fm2, :]
        )
        loc_stiff = loc_stiff.at[:, Fm2].add(0.5 * (Dn1.T @ mmE[:, Fm1]))

        return 0.5 * (gtau * mmE - mmE @ Dn1 - Dn1.T @ mmE), loc_stiff

    branches = [_interior, _dirichlet, _neumann]

    def _assemble_elem_kernel(data, cid):
        mesh_data = data.mdata
        data.pdata.mass.at[cid].set(mesh_data.J[cid] * mesh_data.int_phiphi)

        # global stiff
        Jmat = mesh_data.J_rst_xyz[:, :, cid]
        Dx = Jmat[0, 0] * mesh_data.Dphi[0] + Jmat[1, 0] * mesh_data.Dphi[1]
        Dy = Jmat[0, 1] * mesh_data.Dphi[0] + Jmat[1, 1] * mesh_data.Dphi[1]

        data.pdata.stiff.at[cid].set(
            mesh_data.J[cid]
            * (Dx.T @ mesh_data.int_phiphi @ Dx + Dy.T @ mesh_data.int_phiphi @ Dy)
        )

        # face loop
        for lfid in range(dims.dim):  # lfid = local face id
            ncid = mesh_data.e2e[cid, lfid]  # neigh cell id
            nlfid = mesh_data.e2f[cid, lfid]  # neigh local face id

            Fm1 = mesh_data.vmap_m[lfid, :, cid] // dims.K
            Fm2 = mesh_data.vmap_p[lfid, :, cid] // dims.K

            lnx, lny = mesh_data.nxyz[:, lfid * dims.Nfp, cid]
            lsJ = mesh_data.sJ[lfid, 0, cid]

            hinv = jnp.max(mesh_data.fscale[[lfid, nlfid], 0, cid])

            # Penalty parameter
            gtau = data.pdata.penalty * (dims.N + 1) * (dims.N + 1) * hinv
            # Scaled face mass matrix
            mmE = jnp.zeros_like(Dx)
            mmE.at[jnp.ix_(mesh_data.fmasks[lfid], mesh_data.fmasks[lfid])].set(
                lsJ * mesh_data.face_int_phiphi[lfid]
            )
            # Derivative operators
            Dn1 = lnx * Dx + lny * Dy

            tag = mesh_data.face_tag[cid, lfid]

            dstiff, cstiff = jax.lax.switch(
                tag, branches, mesh_data, lnx, lny, ncid, gtau, mmE, Dn1, Fm1, Fm2
            )

            data.pdata.stiff.at[cid].add(dstiff)
            data.pdata.stiff_coupling.at[cid, lfid].add(cstiff)

        return data, None

    data, _ = jax.lax.scan(_assemble_elem_kernel, data, jnp.arange(dims.K))
    return data.pdata


class Poisson:
    def __init__(self, params, mesh_ops):
        self.params = params
        self.mesh_ops = mesh_ops

        self.is_block_assembled = False
        self.is_assembled_rhs = False
        self.is_assembled = False

        self.tau = params["penalty"]
        self.bc_tags_map = params["bc_tags"]
        assert (0 not in self.bc_tags_map) or (
            self.bc_tags_map[0] == BC.NONE
        ), "Tag 0 is reserved for internal faces and must be mapped to BC.NONE"
        # map mesh tag to BC type
        self.bc_tags_map[0] = BC.NONE
        # map BC type to mesh tag
        self.bc_tags_map_rev = {}
        for tag, bc in self.bc_tags_map.items():
            if bc not in self.bc_tags_map_rev:
                self.bc_tags_map_rev[bc] = []
            self.bc_tags_map_rev[bc].append(tag)

    def _block_assemble(self):
        if self.is_block_assembled:
            return

        N = self.mesh_ops.N
        Np = self.mesh_ops.Np
        K = self.mesh_ops.K
        Nfp = self.mesh_ops.Nfp
        dim = self.mesh_ops.dim
        n_couples = self.mesh_ops.mesh.connectivity_edges.shape[0]

        ops = self.mesh_ops
        ref_ops = ops.ref_elem_ops
        mesh = ops.mesh

        self.stiff = np.zeros((K + n_couples * 2, Np, Np))
        self.mass = np.zeros((K, Np, Np))

        for cid in range(K):  # cid = cell id
            # global mass
            self.mass[cid] = ops.J[cid] * ref_ops.int_phiphi

            # global stiff
            Jmat = ops.J_rst_xyz[:, :, cid]
            Dx = Jmat[0, 0] * ref_ops.Dphi[0] + Jmat[1, 0] * ref_ops.Dphi[1]
            Dy = Jmat[0, 1] * ref_ops.Dphi[0] + Jmat[1, 1] * ref_ops.Dphi[1]

            self.stiff[cid] = ops.J[cid] * (
                Dx.T @ ref_ops.int_phiphi @ Dx + Dy.T @ ref_ops.int_phiphi @ Dy
            )

            # face loop
            for lfid in range(dim + 1):  # lfid = local face id
                ncid = mesh.e2e[cid, lfid]  # neigh cell id
                nlfid = mesh.e2f[cid, lfid]  # neigh local face id

                Fm1 = ops.vmap_m[lfid, :, cid] // K
                Fm2 = ops.vmap_p[lfid, :, cid] // K

                lnx, lny = ops.nxyz[:, lfid * Nfp, cid]
                lsJ = ops.sJ[lfid, 0, cid]
                hinv = max(ops.fscale[lfid, 0, cid], ops.fscale[nlfid, 0, ncid])

                # Penalty parameter
                gtau = self.tau * (N + 1) * (N + 1) * hinv
                # Scaled face mass matrix
                mmE = np.zeros_like(Dx)
                mmE[np.ix_(ref_ops.fmasks[lfid], ref_ops.fmasks[lfid])] = (
                    lsJ * ref_ops.face_int_phiphi[lfid]
                )
                # Derivative operators
                Dn1 = lnx * Dx + lny * Dy

                bc_type = self.bc_tags_map[mesh.face_tag[cid, lfid]]

                match bc_type:
                    case BC.Dirichlet:
                        # Dirichlet: full penalty on diagonal block
                        self.stiff[cid] += gtau * mmE - mmE @ Dn1 - Dn1.T @ mmE
                    case BC.Neumann:
                        # no flux
                        pass
                    case BC.NONE:
                        # Interior face: half contributions to diagonal block
                        self.stiff[cid] += 0.5 * (gtau * mmE - mmE @ Dn1 - Dn1.T @ mmE)

                        Jmat = ops.J_rst_xyz[:, :, ncid]
                        Dx2 = (
                            Jmat[0, 0] * ref_ops.Dphi[0] + Jmat[1, 0] * ref_ops.Dphi[1]
                        )
                        Dy2 = (
                            Jmat[0, 1] * ref_ops.Dphi[0] + Jmat[1, 1] * ref_ops.Dphi[1]
                        )
                        Dn2 = lnx * Dx2 + lny * Dy2

                        # coupling term
                        loc_stiff = np.zeros((Np, Np))
                        loc_stiff[:, Fm2] += -0.5 * gtau * mmE[:, Fm1]
                        loc_stiff[Fm1, :] += -0.5 * mmE[np.ix_(Fm1, Fm1)] @ Dn2[Fm2, :]
                        loc_stiff[:, Fm2] += 0.5 * (Dn1.T @ mmE[:, Fm1])

                        couple_id = mesh.get_cell_couple_id(cid, ncid)
                        self.stiff[couple_id + K] = loc_stiff
                    case _:
                        raise NotImplementedError(f"Cannot handle BC {bc_type}")

        connectivity_edges = self.mesh_ops.mesh.connectivity_edges
        self.ij = np.vstack(
            [
                np.stack([np.arange(K)] * 2, axis=1),
                connectivity_edges,
                connectivity_edges[:, ::-1],
            ]
        )
        self.is_block_assembled = True

    @staticmethod
    def _get_block_idxs(Np, cell_id1, cell_id2=None):
        if cell_id2 is None:
            cell_id2 = cell_id1

        row_indices = np.arange(cell_id1 * Np, (cell_id1 + 1) * Np, dtype=np.int32)
        col_indices = np.arange(cell_id2 * Np, (cell_id2 + 1) * Np, dtype=np.int32)
        rows_grid, cols_grid = np.meshgrid(row_indices, col_indices, indexing="ij")
        return rows_grid.flatten(), cols_grid.flatten()

    @staticmethod
    def assemble_stiff_from_blocks(mesh, stiff_blocks):
        K = mesh.K
        Np = stiff_blocks[0].shape[0]
        n = K * Np

        # stiffness
        connectivity_edges = mesh.connectivity_edges
        ii12, jj12 = zip(
            *(Poisson._get_block_idxs(Np, k1, k2) for k1, k2 in connectivity_edges)
        )
        ii21, jj21 = zip(
            *(Poisson._get_block_idxs(Np, k2, k1) for k1, k2 in connectivity_edges)
        )
        ii, jj = zip(*(Poisson._get_block_idxs(Np, k) for k in range(K)))
        ii = np.concatenate([np.concatenate(ii), *ii12, *ii21])
        jj = np.concatenate([np.concatenate(jj), *jj12, *jj21])
        stiff_mat = scipy.sparse.coo_matrix(
            (stiff_blocks.flat, (ii, jj)), shape=(n, n)
        ).tocsr()
        return stiff_mat

    def assemble(self):
        self._block_assemble()
        if self.is_assembled:
            return

        K = self.mesh_ops.K
        Np = self.mesh_ops.Np
        # mass
        ii, jj = zip(*(self._get_block_idxs(Np, k) for k in range(K)))
        ii = np.concatenate(ii)
        jj = np.concatenate(jj)
        n = K * self.mesh_ops.Np
        self.mass_mat = scipy.sparse.coo_matrix(
            (self.mass.flat, (ii, jj)), shape=(n, n)
        )

        # stiffness
        self.stiff_mat = self.assemble_stiff_from_blocks(self.mesh_ops.mesh, self.stiff)

        self.is_assembled = True

    def assemble_rhs(self, rhs_fn, dir_fn, neu_fn):
        if self.is_assembled_rhs:
            return self.rhs

        ops = self.mesh_ops
        ref_ops = ops.ref_elem_ops
        mesh = ops.mesh

        Np = ops.Np
        Nfp = ops.Nfp
        K = mesh.K
        N = ops.N

        self.rhs = np.zeros((Np, K))

        # TODO: generalize to 3D
        Fx = ops.fxyz[0]
        Fy = ops.fxyz[1]

        empty_bc = np.zeros_like(Fx, dtype=bool)

        d_tags = self.bc_tags_map_rev.get(BC.Dirichlet, [])
        map_d = reduce(operator.or_, [ops.bc_maps[tag] for tag in d_tags], empty_bc)
        self.u_dir = np.zeros_like(Fx)
        self.u_dir[map_d] = dir_fn(Fx[map_d], Fy[map_d])
        self.u_dir = self.u_dir.reshape(ops.Nf, Nfp, K)

        n_tags = self.bc_tags_map_rev.get(BC.Neumann, [])
        map_n = reduce(operator.or_, [ops.bc_maps[tag] for tag in n_tags], empty_bc)
        un_x, un_y = neu_fn(Fx[map_n], Fy[map_n])
        self.u_neu = np.zeros_like(Fx)
        self.u_neu[map_n] = ops.nxyz[0, map_n] * un_x + ops.nxyz[1, map_n] * un_y
        self.u_neu = self.u_neu.reshape(ops.Nf, Nfp, K)

        for cid in range(K):
            Jmat = ops.J_rst_xyz[:, :, cid]
            Dx = Jmat[0, 0] * ref_ops.Dphi[0] + Jmat[1, 0] * ref_ops.Dphi[1]
            Dy = Jmat[0, 1] * ref_ops.Dphi[0] + Jmat[1, 1] * ref_ops.Dphi[1]
            for lfid in range(3):
                Fm1 = ops.vmap_m[lfid, :, cid] // K
                lnx, lny = ops.nxyz[:, lfid * Nfp, cid]
                lsJ = ops.sJ[lfid, 0, cid]
                hinv = ops.fscale[lfid, 0, cid]

                # Penalty parameter
                gtau = self.tau * (N + 1) * (N + 1) * hinv
                # Scaled face mass matrix
                mmE = np.zeros_like(Dx)
                mmE[np.ix_(ref_ops.fmasks[lfid], ref_ops.fmasks[lfid])] = (
                    lsJ * ref_ops.face_int_phiphi[lfid]
                )
                # Derivative operators
                Dn1 = lnx * Dx + lny * Dy

                bc_type = self.bc_tags_map[mesh.face_tag[cid, lfid]]
                match bc_type:
                    case BC.Dirichlet:
                        self.rhs[:, cid] += (
                            gtau * mmE[:, Fm1] - Dn1.T @ mmE[:, Fm1]
                        ) @ self.u_dir[lfid, :, cid]
                    case BC.Neumann:
                        self.rhs[:, cid] += mmE[:, Fm1] * self.u_neu[lfid, :, cid]
                    case BC.NONE:
                        pass
                    case _:
                        raise NotImplementedError(f"Cannot handle BC {bc_type}")

        rhs_vol = ref_ops.int_phiphi @ (rhs_fn(ops.xyz[0], ops.xyz[1]) * ops.J[None, :])
        self.rhs += rhs_vol

        self.is_assembled_rhs = True
        return self.rhs

    def matvec(self, x):
        self._block_assemble()
        assert self.ij.shape[0] == self.stiff.shape[0]
        Np = self.mesh.Np
        y = np.zeros_like(x)
        for k in range(self.stiff.shape[0]):
            i, j = self.ij[k, :]
            y[i * Np : (i + 1) * Np] += self.stiff[i] @ x[j * Np : (j + 1) * Np]
        return y


def matvec_vmap(x, ij, stiff, Np):
    row_indices, col_indices = ij[:, 0], ij[:, 1]
    x_reshaped = x.reshape(-1, Np)
    contributions = jax.vmap(lambda m, v: m @ v)(stiff, x_reshaped[col_indices])
    y_reshaped = jnp.zeros_like(x_reshaped).at[row_indices].add(contributions)
    return y_reshaped.flatten()


def matvec_segment(x, ij, stiff, Np):
    row_indices, col_indices = ij[:, 0], ij[:, 1]
    x_reshaped = x.reshape(-1, Np)
    contributions = jax.vmap(lambda m, v: m @ v)(stiff, x_reshaped[col_indices])
    y_reshaped = jax.ops.segment_sum(
        contributions, row_indices, num_segments=x_reshaped.shape[0]
    )
    return y_reshaped.flatten()


def matvec_fori(x, ij, stiff, Np):
    x_reshaped = x.reshape(-1, Np)
    y_init = jnp.zeros_like(x_reshaped)

    def body_fun(k, y_acc):
        i, j = ij[k, 0], ij[k, 1]
        return y_acc.at[i].add(stiff[k] @ x_reshaped[j])

    y_final = lax.fori_loop(0, stiff.shape[0], body_fun, y_init)
    return y_final.flatten()
