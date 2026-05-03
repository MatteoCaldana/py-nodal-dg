import numpy as np
import scipy.sparse

from pyndg.mesh.bc import BC


class Poisson:
    def __init__(self, params, mesh_ops):
        self.params = params
        self.mesh_ops = mesh_ops

        self.is_block_assembled = False
        self.is_assembled = False

        self.tau = params["penalty"]

    def _block_assemble(self):
        if self.is_block_assembled:
            return

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
                lsJ = ops.sJ[lfid, cid]
                hinv = max(ops.fscale[lfid, cid], ops.fscale[nlfid, ncid])

                # Penalty parameter
                gtau = self.tau * Nfp * Nfp * hinv
                # Scaled face mass matrix
                mmE = np.zeros_like(Dx)
                mmE[np.ix_(ref_ops.fmasks[lfid], ref_ops.fmasks[lfid])] = (
                    lsJ * ref_ops.face_int_phiphi[lfid]
                )
                # Derivative operators
                Dn1 = lnx * Dx + lny * Dy

                # TODO: fix
                bc_type = BC.Dirichlet if ncid == cid else BC.NONE
                # bc_type = mesh.bc[mesh.BCTag[cid, lfid]]

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

    def _get_block_idxs(self, cell_id1, cell_id2=None):
        if cell_id2 is None:
            cell_id2 = cell_id1

        Np = self.mesh_ops.Np
        row_indices = np.arange(cell_id1 * Np, (cell_id1 + 1) * Np, dtype=np.int32)
        col_indices = np.arange(cell_id2 * Np, (cell_id2 + 1) * Np, dtype=np.int32)
        rows_grid, cols_grid = np.meshgrid(row_indices, col_indices, indexing="ij")
        return rows_grid.flatten(), cols_grid.flatten()

    def assemble(self):
        self._block_assemble()
        if self.is_assembled:
            return

        K = self.mesh_ops.K
        # mass
        ii, jj = zip(*(self._get_block_idxs(k) for k in range(K)))
        ii = np.concatenate(ii)
        jj = np.concatenate(jj)
        n = K * self.mesh_ops.Np
        self.mass_mat = scipy.sparse.coo_matrix(
            (self.mass.flat, (ii, jj)), shape=(n, n)
        )

        # stiffness
        connectivity_edges = self.mesh_ops.mesh.connectivity_edges
        ii12, jj12 = zip(
            *(self._get_block_idxs(k1, k2) for k1, k2 in connectivity_edges)
        )
        ii21, jj21 = zip(
            *(self._get_block_idxs(k2, k1) for k1, k2 in connectivity_edges)
        )
        ii = np.concatenate([ii, *ii12, *ii21])
        jj = np.concatenate([jj, *jj12, *jj21])
        self.stiff_mat = scipy.sparse.coo_matrix(
            (self.stiff.flat, (ii, jj)), shape=(n, n)
        )

        self.is_assembled = True

    def assemble_rhs(self):
        if self.is_assembled_rhs:
            return self.rhs

        Np = self.mesh.Np
        Nfp = self.mesh.Nfp
        K = self.mesh.K
        mesh = self.mesh

        self.rhs = np.zeros((Np, K))

        self.uD = np.zeros_like(self.mesh.Fx)
        self.uD[self.mesh.mapD] = self.params.uD(
            self.mesh.Fx(self.mesh.mapD), self.mesh.Fy(self.mesh.mapD)
        )

        self.uN = np.zeros_like(self.mesh.Fx)
        uNdx, uNdy = self.params.uN(
            self.mesh.Fx(self.mesh.mapN), self.mesh.Fy(self.mesh.mapN)
        )
        self.uN[self.mesh.mapN] = (
            self.mesh.nx(self.mesh.mapN) * uNdx + self.mesh.ny(self.mesh.mapN) * uNdy
        )

        for cid in range(K):
            Dx = mesh.rx_avg[cid] * mesh.Dr + mesh.sx_avg[cid] * mesh.Ds
            Dy = mesh.ry_avg[cid] * mesh.Dr + mesh.sy_avg[cid] * mesh.Ds
            for lfid in range(3):

                fslice = slice(lfid * Nfp, (lfid + 1) * Nfp)
                Fm1 = mesh.vmapM[fslice, cid] % Np

                lnx = mesh.nx[lfid * Nfp, cid]
                lny = mesh.ny[lfid * Nfp, cid]
                lsJ = mesh.sJ[lfid * Nfp, cid]
                hinv = mesh.Fscale[lfid * Nfp, cid]

                # Penalty parameter
                gtau = self.tau * mesh.Nfp * mesh.Nfp * hinv
                # Scaled face mass matrix
                mmE = lsJ * mesh.mass_edge[:, :, lfid]
                # Derivative operators
                Dn1 = lnx * Dx + lny * Dy

                bc_type = mesh.bc[mesh.BCTag[cid, lfid]]
                match bc_type:
                    case BC.Dirichlet:
                        self.rhs[:, cid] += (
                            gtau * mmE[:, Fm1] - Dn1.T * mmE[:, Fm1]
                        ) * self.uD[fslice, cid]
                    case BC.Neumann:
                        self.rhs[:, cid] += mmE[:, Fm1] * self.uN[fslice, cid]
                    case BC.NONE:
                        pass
                    case _:
                        raise NotImplementedError(f"Cannot handle BC {bc_type}")

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
