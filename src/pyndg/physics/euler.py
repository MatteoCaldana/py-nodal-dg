import jax
import jax.numpy as jnp
from typing import NamedTuple

from pyndg.ops.meshops import div


class EulerVarPrim(NamedTuple):
    rho: jnp.ndarray  # density (n, K)
    u: jnp.ndarray  # velocity (d, n, K)
    p: jnp.ndarray  # pressure (n, K)


class EulerVarCons(NamedTuple):
    rho: jnp.ndarray  # density (n, K) / mass flux (d, n, K)
    rhou: jnp.ndarray  # momentum (d, n, K) / momentum flux (d, d, n, K)
    E: jnp.ndarray  # total energy (n, K) / energy flux (d, n, K)


@jax.jit
def _p2c(var: EulerVarPrim, gamma: float) -> EulerVarCons:
    rhou = var.rho * var.u
    E = var.p / (gamma - 1.0) + 0.5 * var.rho * jnp.sum(var.u * var.u, axis=0)
    return EulerVarCons(var.rho, rhou, E)


@jax.jit
def _c2p(var: EulerVarCons, gamma: float) -> EulerVarPrim:
    u = var.rhou / var.rho
    p = (gamma - 1.0) * (var.E - 0.5 * jnp.sum(var.rhou * u, axis=0))
    return EulerVarPrim(var.rho, u, p)


@jax.jit
def _c2flux(cvar: EulerVarCons, pvar: EulerVarPrim) -> EulerVarCons:
    dim = cvar.rhou.shape[0]
    ii = jnp.arange(dim)
    return EulerVarCons(
        cvar.rhou,
        (cvar.rhou[:, None] * pvar.u[None, :]).at[ii, ii].add(pvar.p),
        pvar.u * (cvar.E + pvar.p)[None, ...],
    )


@jax.jit
def _wave_speed(var: EulerVarPrim, gamma: float) -> jnp.ndarray:
    c_sound = jnp.sqrt(jnp.abs(gamma * var.p / var.rho))
    return c_sound + jnp.linalg.norm(var.u, axis=0)


@jax.jit
def _dt(
    wspeed: jnp.ndarray, min_hk: float, N: int, visc: float | jnp.ndarray = 0.0
) -> jnp.ndarray:
    N2 = N * N
    N4 = N2 * N2
    dt_inv_adv = jnp.max(wspeed) * N2 / min_hk / 2
    dt_ind_vis = visc * N4 / min_hk**2 / 4
    return 1.0 / (dt_inv_adv + dt_ind_vis)


def _tree_div(J_rst_xyz, Dphi_weak, var):
    return jax.tree.map(lambda x: div(J_rst_xyz, Dphi_weak, x), var)


def _lax_friedrichs_flux(mesh_data, dims, q_plus, q_minus, gamma):
    p_plus = _c2p(q_plus, gamma)
    p_minus = _c2p(q_minus, gamma)
    flux_plus = _c2flux(q_plus, p_plus)
    flux_minus = _c2flux(q_minus, p_minus)
    wspeed = jnp.maximum(_wave_speed(p_plus, gamma), _wave_speed(p_minus, gamma))
    max_wspeed = jnp.max(wspeed, axis=1, keepdims=True)

    nxyz = mesh_data.nxyz.reshape((dims.dim, dims.Nf, dims.Nfp, dims.K))

    def _lf_flux(qm, qp, fm, fp):
        # nxyz can be broadcasted, the -4 is to sum on the first non-broadcasted axis
        return 0.5 * (jnp.sum(nxyz * (fp + fm), axis=-4) + max_wspeed * (qm - qp))

    flux = jax.tree.map(
        _lf_flux,
        q_minus,
        q_plus,
        flux_minus,
        flux_plus,
    )
    return flux


def _hll_flux(mesh_data, dims, q_plus, q_minus, gamma):
    # 1. Compute primitive variables and standard Cartesian fluxes
    p_plus = _c2p(q_plus, gamma)
    p_minus = _c2p(q_minus, gamma)
    flux_plus = _c2flux(q_plus, p_plus)
    flux_minus = _c2flux(q_minus, p_minus)

    # Normal vector reshaped to (dim, Nf, Nfp, K)
    nxyz = mesh_data.nxyz.reshape((dims.dim, dims.Nf, dims.Nfp, dims.K))

    # 2. Compute normal velocities (u dot n)
    # axis=-4 corresponds to the 'dim' axis in our spatial variables
    un_minus = jnp.sum(p_minus.u * nxyz, axis=-4)
    un_plus = jnp.sum(p_plus.u * nxyz, axis=-4)

    # 3. Compute Enthalpy and Sound Speed for L/R states
    H_minus = (q_minus.E + p_minus.p) / p_minus.rho
    H_plus = (q_plus.E + p_plus.p) / p_plus.rho
    c_minus = jnp.sqrt(gamma * p_minus.p / p_minus.rho)
    c_plus = jnp.sqrt(gamma * p_plus.p / p_plus.rho)

    # 4. Compute Roe Averages
    sqrt_rho_minus = jnp.sqrt(p_minus.rho)
    sqrt_rho_plus = jnp.sqrt(p_plus.rho)
    sqrt_rho_sum = sqrt_rho_minus + sqrt_rho_plus

    # Vector u broadcasts perfectly across the scalar density values
    u_roe = (sqrt_rho_minus * p_minus.u + sqrt_rho_plus * p_plus.u) / sqrt_rho_sum
    H_roe = (sqrt_rho_minus * H_minus + sqrt_rho_plus * H_plus) / sqrt_rho_sum

    # Normal velocity and sound speed of the Roe average
    un_roe = jnp.sum(u_roe * nxyz, axis=-4)
    u_roe_sq = jnp.sum(u_roe * u_roe, axis=-4)
    c_roe = jnp.sqrt((gamma - 1.0) * (H_roe - 0.5 * u_roe_sq))

    # 5. Estimate Wave Speeds
    SL = jnp.minimum(un_minus - c_minus, un_roe - c_roe)
    SR = jnp.maximum(un_plus + c_plus, un_roe + c_roe)

    # 6. Compute HLL blending coefficients
    # Add a tiny epsilon to the denominator to prevent division by zero in perfect vacuum/uniform flow
    denom = jnp.maximum(SR - SL, 1e-12)

    t1 = (jnp.minimum(SR, 0.0) - jnp.minimum(0.0, SL)) / denom
    t2 = 1.0 - t1
    t3 = (SR * jnp.abs(SL) - SL * jnp.abs(SR)) / (2.0 * denom)

    # 7. Map the HLL calculation across all conserved variable fields
    def _hll_calc(qm, qp, fm, fp):
        # Project the standard Cartesian fluxes onto the face normal vector
        fn_m = jnp.sum(nxyz * fm, axis=-4)
        fn_p = jnp.sum(nxyz * fp, axis=-4)

        # Apply the HLL averaging formula
        # JAX will automatically right-align and broadcast t1, t2, t3 onto vector/tensor fields
        return t1 * fn_p + t2 * fn_m - t3 * (qp - qm)

    flux = jax.tree.map(
        _hll_calc,
        q_minus,
        q_plus,
        flux_minus,
        flux_plus,
    )

    return flux


_BD_FLUX_FN = {"LF": _lax_friedrichs_flux, "HLL": _hll_flux}


def _apply_vmap(var, vmap):
    return jax.tree.map(lambda x: x.reshape(*x.shape[:-2], -1)[..., vmap], var)


def _add_bd_flux(rhs, bd_flux, ops, dims):
    def _reshape(fx):
        # compact the Nf and Nfp dimensions into one, to be able to apply the flux to the rhs
        return fx.reshape(*fx.shape[:-3], dims.Nf * dims.Nfp, dims.K)

    return jax.tree.map(
        lambda r, fx: r - ops.lift @ (ops.fscale * _reshape(fx)),
        rhs,
        bd_flux,
    )


def _rhs(ops, dims, q, time, bd_flux_type, gamma, bc_fn):
    p = _c2p(q, gamma)
    flux = _c2flux(q, p)
    rhs = _tree_div(ops.J_rst_xyz, ops.Dphi_weak, flux)
    q_plus_tmp = _apply_vmap(q, ops.vmap_p)
    q_minus = _apply_vmap(q, ops.vmap_m)
    q_plus = bc_fn(q_plus_tmp, time)
    bd_flux = _BD_FLUX_FN[bd_flux_type](ops, dims, q_plus, q_minus, gamma)
    rhs_new = _add_bd_flux(rhs, bd_flux, ops, dims)
    return rhs, q_plus_tmp, q_plus, q_minus, bd_flux, rhs_new


def _step():
    pass


class Euler:
    def __init__(self, mesh_ops, params):
        self.mesh_ops = mesh_ops
        self.params = params
