import scipy.io
import scipy.sparse
import os

os.environ["PYNDG_BACKEND"] = "numpy"

from pyndg.scalar_param_2d import ScalarParam2D
from pyndg.mesh_2d import Mesh2D

from pyndg.time_integrator import LS54
from pyndg.physic_model import Advection2D
from pyndg.bc_2d import BC
from pyndg.limiters import Limiter
from pyndg.viscosity_model import NoViscosity

import numpy as np
import time

import jax
import jax.numpy as jnp
from jax import lax

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)


PATH = "/home/matteo/Documents/nodal-dg/Codes1.1/"


class Poisson2D:
    def __init__(self, params, mesh=None):
        self.params = params
        self.mesh = Mesh2D(params) if mesh is None else mesh

        if not self.mesh.is_init:
            self.mesh.initialize()

        self.is_block_assembled = False
        self.is_assembled = False

        self.tau = 20  # penalty

    def _block_assemble(self):
        if self.is_block_assembled:
            return

        Np = self.mesh.Np
        K = self.mesh.K
        Nfp = self.mesh.Nfp
        n_couples = self.mesh.cell_couplings.shape[0]

        mesh = self.mesh

        self.stiff = np.empty((K + n_couples * 2, Np, Np))
        self.mass = np.empty((K, Np, Np))

        for cid in range(K):  # cid = cell id
            # global mass
            self.mass[cid] = mesh.J_avg[cid] * mesh.mass

            # global stiff
            Dx = mesh.rx_avg[cid] * mesh.Dr + mesh.sx_avg[cid] * mesh.Ds
            Dy = mesh.ry_avg[cid] * mesh.Dr + mesh.sy_avg[cid] * mesh.Ds

            self.stiff[cid] = mesh.J_avg[cid] * (
                Dx.T @ mesh.mass @ Dx + Dy.T @ mesh.mass @ Dy
            )

            # face loop
            for lfid in range(3):  # lfid = local face id
                ncid = mesh.EToE[cid, lfid]  # neigh cell id
                nlfid = mesh.EToF[cid, lfid]  # neigh local face id

                fslice = slice(lfid * Nfp, (lfid + 1) * Nfp)
                Fm1 = mesh.vmapM[fslice, cid] % Np
                Fm2 = mesh.vmapP[fslice, cid] % Np

                lnx = mesh.nx[lfid * Nfp, cid]
                lny = mesh.ny[lfid * Nfp, cid]
                lsJ = mesh.sJ[lfid * Nfp, cid]
                hinv = max(mesh.Fscale[lfid * Nfp, cid], mesh.Fscale[nlfid * Nfp, ncid])

                # Penalty parameter
                gtau = self.tau * mesh.Nfp * mesh.Nfp * hinv
                # Scaled face mass matrix
                mmE = lsJ * mesh.mass_edge[:, :, lfid]
                # Derivative operators
                Dn1 = lnx * Dx + lny * Dy

                bc_type = mesh.bc[mesh.BCTag[cid, lfid]]

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

                        Dx2 = mesh.rx_avg[ncid] * mesh.Dr + mesh.sx_avg[ncid] * mesh.Ds
                        Dy2 = mesh.ry_avg[ncid] * mesh.Dr + mesh.sy_avg[ncid] * mesh.Ds
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

        self.is_block_assembled = True

        self.ij = np.vstack(
            [
                np.stack([np.arange(K)] * 2, axis=1),
                self.mesh.cell_couplings,
                self.mesh.cell_couplings[:, ::-1],
            ]
        )

    def _get_block_idxs(self, cell_id1, cell_id2=None):
        if cell_id2 is None:
            cell_id2 = cell_id1

        Np = self.mesh.Np
        row_indices = np.arange(cell_id1 * Np, (cell_id1 + 1) * Np, dtype=np.int32)
        col_indices = np.arange(cell_id2 * Np, (cell_id2 + 1) * Np, dtype=np.int32)
        rows_grid, cols_grid = np.meshgrid(row_indices, col_indices, indexing="ij")
        return rows_grid.flatten(), cols_grid.flatten()

    def assemble(self):
        self._block_assemble()
        if self.is_assembled:
            return

        # mass
        ii, jj = zip(*(self._get_block_idxs(k) for k in range(self.mesh.K)))
        ii = np.concatenate(ii)
        jj = np.concatenate(jj)
        n = self.mesh.K * self.mesh.Np
        self.mass_mat = scipy.sparse.coo_matrix(
            (self.mass.flat, (ii, jj)), shape=(n, n)
        )

        # stiffness
        ii12, jj12 = zip(
            *(self._get_block_idxs(k1, k2) for k1, k2 in self.mesh.cell_couplings)
        )
        ii21, jj21 = zip(
            *(self._get_block_idxs(k2, k1) for k1, k2 in self.mesh.cell_couplings)
        )
        ii = np.concatenate([ii, *ii12, *ii21])
        jj = np.concatenate([jj, *jj12, *jj21])
        self.stiff_mat = scipy.sparse.coo_matrix(
            (self.stiff.flat, (ii, jj)), shape=(n, n)
        )

        self.is_assembled = True

    def matvec(self, x):
        self._block_assemble()
        assert self.ij.shape[0] == self.stiff.shape[0]
        Np = self.mesh.Np
        y = np.zeros_like(x)
        for k in range(self.stiff.shape[0]):
            i, j = self.ij[k, :]
            y[i * Np : (i + 1) * Np] += self.stiff[i] @ x[j * Np : (j + 1) * Np]
        return y


@jax.jit(static_argnums=(3,))
def matvec_vmap(x, ij, stiff, Np):
    row_indices, col_indices = ij[:, 0], ij[:, 1]
    x_reshaped = x.reshape(-1, Np)
    contributions = jax.vmap(lambda m, v: m @ v)(stiff, x_reshaped[col_indices])
    y_reshaped = jnp.zeros_like(x_reshaped).at[row_indices].add(contributions)
    return y_reshaped.flatten()

@jax.jit(static_argnums=(3,))
def matvec_segment(x, ij, stiff, Np):
    row_indices, col_indices = ij[:, 0], ij[:, 1]
    x_reshaped = x.reshape(-1, Np)
    contributions = jax.vmap(lambda m, v: m @ v)(stiff, x_reshaped[col_indices])
    y_reshaped = jax.ops.segment_sum(contributions, row_indices, num_segments=x_reshaped.shape[0])
    return y_reshaped.flatten()

@jax.jit(static_argnums=(3,))
def matvec_fori(x, ij, stiff, Np):
    x_reshaped = x.reshape(-1, Np)
    y_init = jnp.zeros_like(x_reshaped)
    
    def body_fun(k, y_acc):
        i, j = ij[k, 0], ij[k, 1]
        return y_acc.at[i].add(stiff[k] @ x_reshaped[j])

    y_final = lax.fori_loop(0, stiff.shape[0], body_fun, y_init)
    return y_final.flatten()


def run_benchmark(name, func, args, iters=100):
    # Warmup (compilation happens here)
    _ = func(*args).block_until_ready()
    
    t0 = time.time()
    for _ in range(iters):
        res = func(*args).block_until_ready() # Crucial for JAX timing!
    t1 = time.time()
    
    print(f"{name:15} | Elapsed: {(t1 - t0) * 1000:8.3f} ms/iter")

# TODO:
# - check convergence error
# - implement matvec (check against assemble)
# - check cpu vs gpu cost of CRS vs matvec (BCOO, mine block, real block)
# - possible cuthill mckee reordering of elements for matvec
# - pGMG

if __name__ == "__main__":
    N = 5

    data = scipy.io.loadmat(PATH + f"Poisson2D_N{N}.mat")

    mesh_name = "Grid/Other/circA01.neu"

    params = ScalarParam2D(
        model=Advection2D(),
        name="Test",
        N=N,
        mesh_file=PATH + mesh_name,
        u_IC=lambda x: x,
        bc={},  # {101: BC.Out, 102: BC.Slip, 103: BC.In, 104: BC.Slip},
        final_time=1.0,
        cfl=1.0,
        time_integrator=LS54(),
        viscosity_model=NoViscosity(),
        limiter=Limiter(),
    )

    mesh = Mesh2D(params)
    mesh.initialize()
    # mesh.plot(show_elem_id=False, show_vtx_id=False)

    print(np.max(np.abs(mesh.VX - data["VX"].squeeze())))
    print(np.max(np.abs(mesh.VY - data["VY"].squeeze())))

    print(np.max(np.abs(mesh.Fmask - (data["Fmask"] - 1))))
    print(np.max(np.abs(mesh.r - data["r"].squeeze())))
    print(np.max(np.abs(mesh.s - data["s"].squeeze())))

    print(np.max(np.abs(mesh.V - data["V"])))
    print(np.max(np.abs(mesh.invV - data["invV"])))

    print(np.max(np.abs(mesh.rx - data["rx"])))
    print(np.max(np.abs(mesh.sx - data["sx"])))
    print(np.max(np.abs(mesh.Dr - data["Dr"])))
    print(np.max(np.abs(mesh.Ds - data["Ds"])))
    print(np.max(np.abs(mesh.J - data["J"])))
    print(np.max(np.abs(mesh.Fscale - data["Fscale"])))

    problem = Poisson2D(params, mesh)
    problem.assemble()

    print(np.max(np.abs(problem.mass_mat - data["M"])))
    print(np.max(np.abs(problem.stiff_mat - data["A"])))

    print(problem.stiff_mat.shape)

    x = np.ones(problem.stiff_mat.shape[0])
    t0 = time.time()
    for _ in range(100):
        problem.stiff_mat @ x
    t1 = time.time()
    print(f"Scipy COO matvec     {(t1 - t0) * 1000:.1f}")

    mat = problem.stiff_mat.tocsr()
    t0 = time.time()
    for _ in range(100):
        mat @ x
    t1 = time.time()
    print(f"Scipy CSR matvec     {(t1 - t0) * 1000:.1f}")

    t0 = time.time()
    for _ in range(100):
        problem.matvec(x)
    t1 = time.time()
    print(f"Block numpy matvec   {(t1 - t0) * 1000:.1f}")


    run_benchmark("JAX Vmap", matvec_vmap, (x, problem.ij, problem.stiff, problem.mesh.Np))
    run_benchmark("JAX Segment Sum",   matvec_segment, (x, problem.ij, problem.stiff, problem.mesh.Np))
    run_benchmark("JAX Fori Loop",     matvec_fori, (x, problem.ij, problem.stiff, problem.mesh.Np))