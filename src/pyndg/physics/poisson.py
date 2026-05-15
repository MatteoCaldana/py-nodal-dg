import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse
import operator
from functools import reduce
from functools import partial

from pyndg.mesh.bc import BC
import pyndg.backend as bkd


@jax.jit
def _mass_kernel(mesh_data):
    def compute_element(j_val):
        return j_val * mesh_data.int_phiphi

    return jax.vmap(compute_element)(mesh_data.J)


@jax.jit
def _Dxyz_kernel(mesh_data):
    return jnp.einsum("rdk,rij->dkij", mesh_data.J_rst_xyz, mesh_data.Dphi)


@jax.jit
def _stiff_self_kernel(mesh_data, Dxyz):
    return jnp.einsum(
        "k,dkji,jm,dkml->kil", mesh_data.J, Dxyz, mesh_data.int_phiphi, Dxyz
    )


@jax.jit
def _hinv_kernel(mesh_data):
    fscale_self = mesh_data.fscale[:, 0, :].T  # (K, Nf)
    fscale_neigh = mesh_data.fscale[mesh_data.e2f, 0, mesh_data.e2e]  # (K, Nf)
    return jnp.maximum(fscale_self, fscale_neigh)  # (K, Nf)


@partial(jax.jit, static_argnames=["Nfp"])
def _Dnormal_kernel(mesh_data, Dxyz, Nfp):
    # nxyz : (ndim, Nf*Nfp, K) — take first node of each face as the normal
    lnxyz = mesh_data.nxyz[:, ::Nfp, :]  # (ndim, Nf, K)
    return jnp.einsum("dfk,dkij->kfij", lnxyz, Dxyz)  # (K, Nf, Np, Np)


_COEFS = jnp.asarray(
    [
        # loc_ff  loc_row  loc_col  neigh_ff neigh_row neigh_col
        [0.5, -0.5, -0.5, -0.5, -0.5, 0.5],  # interior
        [1.0, -1.0, -1.0, +0.0, +0.0, 0.0],  # dirichlet
        [0.0, +0.0, +0.0, +0.0, +0.0, 0.0],  # neumann
    ],
    dtype=bkd.jnp_prec,
)


@partial(jax.jit, static_argnames=["Nf", "Np"])
def _coupling_assemble_kernel(mesh_data, Dn, stiff_self, gtau, cids, K, Nf, Np):
    def _assemble_elem_kernel(cid):
        stiff_loc = stiff_self[cid]
        stiff_neigh = jnp.zeros((Nf, Np, Np), dtype=bkd.jnp_prec)
        for lfid in range(Nf):
            ncid = mesh_data.e2e[cid, lfid]
            nlfid = mesh_data.e2f[cid, lfid]

            Fm1 = mesh_data.vmap_m[lfid, :, cid] // K
            Fm2 = mesh_data.vmap_p[lfid, :, cid] // K

            gamma = gtau[cid, lfid]

            mmE = mesh_data.sJ[lfid, 0, cid] * mesh_data.face_int_phiphi[lfid]
            Dn1_f = Dn[cid, lfid][Fm1, :]  # (Nfp, Np)
            Dn2_f = -Dn[ncid, nlfid][Fm2, :]  # (Nfp, Np)

            tag = mesh_data.face_tag[cid, lfid]
            coef = _COEFS[tag]

            Adn1 = mmE @ Dn1_f
            Dn1tA = Dn1_f.T @ mmE
            Adn2 = mmE @ Dn2_f

            stiff_loc = stiff_loc.at[Fm1[:, None], Fm1].add(coef[0] * gamma * mmE)
            stiff_loc = stiff_loc.at[Fm1, :].add(coef[1] * Adn1)
            stiff_loc = stiff_loc.at[:, Fm1].add(coef[2] * Dn1tA)

            stiff_neigh = stiff_neigh.at[lfid, Fm1[:, None], Fm2].add(
                coef[3] * gamma * mmE
            )
            stiff_neigh = stiff_neigh.at[lfid, Fm1, :].add(coef[4] * Adn2)
            # Does not preserve axis order under scatter/gather lowering.
            # Advanced-index dimensions get moved to the front.
            stiff_neigh = stiff_neigh.at[lfid, :, Fm2].add(coef[5] * Dn1tA.T)

        return stiff_loc, stiff_neigh

    assemble_fn = jax.jit(jax.vmap(_assemble_elem_kernel))
    stiff, stiff_coupling = assemble_fn(cids)
    return stiff, stiff_coupling


@jax.jit
def _flatten_blocks(mesh_data, stiff, stiff_coupling):
    idxs = mesh_data.eid2ef
    stiff = jnp.concatenate([stiff, stiff_coupling[idxs[0], idxs[1]]])
    return stiff


@partial(jax.jit, static_argnames=["dim", "Nf", "Np"])
def _stiff_asseble_step_kernel(
    J,
    Dxyz,
    mass,
    gtau,
    sJ,
    face_int_phiphi,
    Dn1,
    Dn2,
    Fm1s,
    Fm2s,
    face_tag,
    dim,
    Nf,
    Np,
):
    print("Assembling stiffness for one element...")
    stiff = J * sum([Dxyz[d].T @ mass @ Dxyz[d] for d in range(dim)])
    stiff_neigh = jnp.zeros((Nf, Np, Np), dtype=bkd.jnp_prec)
    for lfid in range(Nf):
        gamma = gtau[lfid]
        Fm1 = Fm1s[lfid]
        Fm2 = Fm2s[lfid]

        mmE = sJ[lfid] * face_int_phiphi[lfid]
        Dn1_f = Dn1[lfid][Fm1, :]
        Dn2_f = -Dn2[lfid][Fm2, :]

        tag = face_tag[lfid]
        coef = _COEFS[tag]

        Adn1 = mmE @ Dn1_f
        Dn1tA = Dn1_f.T @ mmE
        Adn2 = mmE @ Dn2_f

        stiff = stiff.at[Fm1[:, None], Fm1].add(coef[0] * gamma * mmE)
        stiff = stiff.at[Fm1, :].add(coef[1] * Adn1)
        stiff = stiff.at[:, Fm1].add(coef[2] * Dn1tA)

        stiff_neigh = stiff_neigh.at[lfid, Fm1[:, None], Fm2].add(coef[3] * gamma * mmE)
        stiff_neigh = stiff_neigh.at[lfid, Fm1, :].add(coef[4] * Adn2)
        # Does not preserve axis order under scatter/gather lowering.
        # Advanced-index dimensions get moved to the front.
        stiff_neigh = stiff_neigh.at[lfid, :, Fm2].add(coef[5] * Dn1tA.T)

    return stiff, stiff_neigh


def block_assemble_kernel(mesh_data, penalty, dims):
    mass = _mass_kernel(mesh_data)
    Dxyz = _Dxyz_kernel(mesh_data)
    Dn = _Dnormal_kernel(mesh_data, Dxyz, dims.Nfp)
    hinv = _hinv_kernel(mesh_data)
    gtau = penalty * (dims.N + 1) * (dims.N + 1) * hinv
    stiff_self = _stiff_self_kernel(mesh_data, Dxyz)
    stiff, stiff_coupling = _coupling_assemble_kernel(
        mesh_data, Dn, stiff_self, gtau, jnp.arange(dims.K), dims.K, dims.Nf, dims.Np
    )
    stiff = _flatten_blocks(mesh_data, stiff, stiff_coupling)
    return mass, stiff


def block_assemble_kernel_v2(mesh_data, penalty, dims):
    mass = _mass_kernel(mesh_data)
    Dxyz = _Dxyz_kernel(mesh_data)
    Dn = _Dnormal_kernel(mesh_data, Dxyz, dims.Nfp)
    hinv = _hinv_kernel(mesh_data)
    gtau = penalty * (dims.N + 1) * (dims.N + 1) * hinv

    stiff, stiff_coupling = [], []
    for cid in range(dims.K):
        ncid = mesh_data.e2e[cid, 0]
        nlfid = mesh_data.e2f[cid, :]
        tmp1, tmp2 = _stiff_asseble_step_kernel(
            mesh_data.J[cid],
            Dxyz[:, cid],
            mesh_data.int_phiphi,
            gtau[cid],
            mesh_data.sJ[:, 0, cid],
            mesh_data.face_int_phiphi,
            Dn[cid],
            Dn[ncid, nlfid],
            mesh_data.vmap_m[:, :, cid] // dims.K,
            mesh_data.vmap_p[:, :, cid] // dims.K,
            mesh_data.face_tag[:, cid],
            dims.dim,
            dims.Nf,
            dims.Np,
        )
        stiff.append(tmp1)
        stiff_coupling.append(tmp2)

    stiff = _flatten_blocks(mesh_data, jnp.stack(stiff), jnp.stack(stiff_coupling))
    return mass, mass


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
            Dxyz = [0] * dim
            for d1 in range(dim):
                for d2 in range(dim):
                    Dxyz[d1] += Jmat[d2, d1] * ref_ops.Dphi[d2]

            self.stiff[cid] = ops.J[cid] * sum(
                [Dxyz[d].T @ ref_ops.int_phiphi @ Dxyz[d] for d in range(dim)]
            )

            # face loop
            for lfid in range(dim + 1):  # lfid = local face id
                ncid = mesh.e2e[cid, lfid]  # neigh cell id
                nlfid = mesh.e2f[cid, lfid]  # neigh local face id

                Fm1 = ops.vmap_m[lfid, :, cid] // K
                Fm2 = ops.vmap_p[lfid, :, cid] // K

                ln = ops.nxyz[:, lfid * Nfp, cid]
                lsJ = ops.sJ[lfid, 0, cid]

                hinv = ops.fscale[[lfid, nlfid], 0, [cid, ncid]].max()

                # Penalty parameter
                gtau = self.tau * (N + 1) * (N + 1) * hinv
                # Scaled face mass matrix
                mmE = np.zeros((Np, Np))
                idx = np.ix_(ref_ops.fmasks[lfid], ref_ops.fmasks[lfid])
                mmE[idx] = lsJ * ref_ops.face_int_phiphi[lfid]
                # Derivative operators
                Dn1 = sum([ln[d] * Dxyz[d] for d in range(dim)])

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
                        Dxyz2 = [0] * dim
                        for d1 in range(dim):
                            for d2 in range(dim):
                                Dxyz2[d1] += Jmat[d2, d1] * ref_ops.Dphi[d2]
                        Dn2 = sum([ln[d] * Dxyz2[d] for d in range(dim)])

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
        Nf = ops.Nf
        dim = ops.dim

        self.rhs = np.zeros((Np, K))

        # TODO: generalize to 3D
        empty_bc = np.zeros((Nfp * Nf, K), dtype=bool)

        d_tags = self.bc_tags_map_rev.get(BC.Dirichlet, [])
        map_d = reduce(operator.or_, [ops.bc_maps[tag] for tag in d_tags], empty_bc)
        self.u_dir = np.zeros((Nfp * Nf, K), dtype=ops.xyz.dtype)
        self.u_dir[map_d] = dir_fn(ops.fxyz[:, map_d])
        self.u_dir = self.u_dir.reshape(ops.Nf, Nfp, K)

        n_tags = self.bc_tags_map_rev.get(BC.Neumann, [])
        map_n = reduce(operator.or_, [ops.bc_maps[tag] for tag in n_tags], empty_bc)
        dudxyz = neu_fn(ops.fxyz[:, map_n])
        self.u_neu = np.zeros((Nfp * Nf, K), dtype=ops.xyz.dtype)
        self.u_neu[map_n] = sum([ops.nxyz[d, map_n] * dudxyz[d] for d in range(dim)])
        self.u_neu = self.u_neu.reshape(ops.Nf, Nfp, K)

        for cid in range(K):
            Jmat = ops.J_rst_xyz[:, :, cid]
            Dxyz = [0] * dim
            for d1 in range(dim):
                for d2 in range(dim):
                    Dxyz[d1] += Jmat[d2, d1] * ref_ops.Dphi[d2]
            for lfid in range(Nf):
                Fm1 = ops.vmap_m[lfid, :, cid] // K
                ln = ops.nxyz[:, lfid * Nfp, cid]
                lsJ = ops.sJ[lfid, 0, cid]
                hinv = ops.fscale[lfid, 0, cid]

                # Penalty parameter
                gtau = self.tau * (N + 1) * (N + 1) * hinv
                # Scaled face mass matrix
                mmE = np.zeros((Np, Np))
                idx = np.ix_(ref_ops.fmasks[lfid], ref_ops.fmasks[lfid])
                mmE[idx] = lsJ * ref_ops.face_int_phiphi[lfid]
                # Derivative operators
                Dn1 = sum([ln[d] * Dxyz[d] for d in range(dim)])

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

        rhs_vol = ref_ops.int_phiphi @ (rhs_fn(ops.xyz) * ops.J[None, :])
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
