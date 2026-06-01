from collections import defaultdict
from functools import partial

import jax
import jax.numpy as jnp
from typing import NamedTuple

from pyndg.ops.meshops import div
from pyndg.time.integrator import get_timestepper


class EulerVarPrim(NamedTuple):
    rho: jnp.ndarray  # density (n, K)
    u: jnp.ndarray  # velocity (d, n, K)
    p: jnp.ndarray  # pressure (n, K)


class EulerVarCons(NamedTuple):
    rho: jnp.ndarray  # density (n, K) / mass flux (d, n, K)
    rhou: jnp.ndarray  # momentum (d, n, K) / momentum flux (d, d, n, K)
    E: jnp.ndarray  # total energy (n, K) / energy flux (d, n, K)

    def __rmul__(self, scalar):
        return jax.tree.map(lambda x: scalar * x, self)

    def __mul__(self, scalar):
        return jax.tree.map(lambda x: scalar * x, self)

    def __add__(self, other):
        return jax.tree.map(lambda x, y: x + y, self, other)


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


@jax.jit
def _tree_div(J_rst_xyz, Dphi_weak, var):
    return jax.tree.map(lambda x: div(J_rst_xyz, Dphi_weak, x), var)


@partial(jax.jit, static_argnames=["dims"])
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


@partial(jax.jit, static_argnames=["dims"])
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


@partial(jax.jit, static_argnames=["dims"])
def _roe_flux(mesh_data, dims, q_plus, q_minus, gamma):
    # 1. Compute primitive variables and standard Cartesian fluxes
    p_plus = _c2p(q_plus, gamma)
    p_minus = _c2p(q_minus, gamma)
    flux_plus = _c2flux(q_plus, p_plus)
    flux_minus = _c2flux(q_minus, p_minus)

    # Normal vector reshaped to (dim, Nf, Nfp, K)
    nxyz = mesh_data.nxyz.reshape((dims.dim, dims.Nf, dims.Nfp, dims.K))

    # 2. Compute Enthalpy for L/R states
    H_minus = (q_minus.E + p_minus.p) / p_minus.rho
    H_plus = (q_plus.E + p_plus.p) / p_plus.rho

    # 3. Compute Roe Averages
    sqrt_rho_minus = jnp.sqrt(p_minus.rho)
    sqrt_rho_plus = jnp.sqrt(p_plus.rho)
    sqrt_rho_sum = sqrt_rho_minus + sqrt_rho_plus

    rho_roe = sqrt_rho_minus * sqrt_rho_plus
    u_roe = (sqrt_rho_minus * p_minus.u + sqrt_rho_plus * p_plus.u) / sqrt_rho_sum
    H_roe = (sqrt_rho_minus * H_minus + sqrt_rho_plus * H_plus) / sqrt_rho_sum

    # Normal velocity, squared magnitude, and sound speed of the Roe average
    un_roe = jnp.sum(u_roe * nxyz, axis=0)
    u_roe_sq = jnp.sum(u_roe * u_roe, axis=0)
    c2_roe = (gamma - 1.0) * (H_roe - 0.5 * u_roe_sq)
    c_roe = jnp.sqrt(c2_roe)

    # 4. Compute State Jumps (Right - Left)
    delta_rho = p_plus.rho - p_minus.rho
    delta_p = p_plus.p - p_minus.p
    delta_u = p_plus.u - p_minus.u
    delta_un = jnp.sum(delta_u * nxyz, axis=0)  # Normal velocity jump (uP - uM)

    # 5. Compute Riemann Invariants / Wave Strengths (dW)
    dW1 = -0.5 * rho_roe * delta_un / c_roe + 0.5 * delta_p / c2_roe
    dW2 = delta_rho - delta_p / c2_roe
    # In vector form, dW3 becomes the transverse velocity jump vector (orthogonal to n)
    dW3_vec = rho_roe * (delta_u - delta_un * nxyz)
    dW4 = 0.5 * rho_roe * delta_un / c_roe + 0.5 * delta_p / c2_roe

    # 6. Apply Absolute Eigenvalues (Wave Speeds)
    # Keeping strict adherence to the Matlab semantics: no Harten entropy fix.
    alpha1 = jnp.abs(un_roe - c_roe) * dW1
    alpha2 = jnp.abs(un_roe) * dW2
    alpha3_vec = jnp.abs(un_roe) * dW3_vec
    alpha4 = jnp.abs(un_roe + c_roe) * dW4

    # 7. Construct the Roe Dissipation Flux Vector (|A| * delta_Q)
    dF_rho = alpha1 + alpha2 + alpha4
    dF_rhou = (
        alpha1 * (u_roe - c_roe * nxyz)
        + alpha2 * u_roe
        + alpha3_vec
        + alpha4 * (u_roe + c_roe * nxyz)
    )
    dF_E = (
        alpha1 * (H_roe - un_roe * c_roe)
        + alpha2 * 0.5 * u_roe_sq
        + jnp.sum(alpha3_vec * u_roe, axis=0)
        + alpha4 * (H_roe + un_roe * c_roe)
    )

    # 8. Project the standard Cartesian fluxes onto the face normal vector
    fn_rho_m = jnp.sum(flux_minus.rho * nxyz, axis=0)
    fn_rho_p = jnp.sum(flux_plus.rho * nxyz, axis=0)

    # For momentum flux tensor (dim_mom, dim_flux, ...), multiply nxyz across the flux direction (axis=1)
    fn_rhou_m = jnp.sum(flux_minus.rhou * nxyz[None, ...], axis=1)
    fn_rhou_p = jnp.sum(flux_plus.rhou * nxyz[None, ...], axis=1)

    fn_E_m = jnp.sum(flux_minus.E * nxyz, axis=0)
    fn_E_p = jnp.sum(flux_plus.E * nxyz, axis=0)

    # 9. Compute the final upwind Roe Flux: 1/2 * (F_R + F_L - |A| * delta_Q)
    flux_rho = 0.5 * (fn_rho_p + fn_rho_m - dF_rho)
    flux_rhou = 0.5 * (fn_rhou_p + fn_rhou_m - dF_rhou)
    flux_E = 0.5 * (fn_E_p + fn_E_m - dF_E)

    return EulerVarCons(flux_rho, flux_rhou, flux_E)


@partial(jax.jit, static_argnames=["dims"])
def _hllc_flux(mesh_data, dims, q_plus, q_minus, gamma):
    # 1. Compute primitive variables and standard Cartesian fluxes
    p_plus = _c2p(q_plus, gamma)
    p_minus = _c2p(q_minus, gamma)
    flux_plus = _c2flux(q_plus, p_plus)
    flux_minus = _c2flux(q_minus, p_minus)

    # Normal vector reshaped to (dim, Nf, Nfp, K)
    nxyz = mesh_data.nxyz.reshape((dims.dim, dims.Nf, dims.Nfp, dims.K))

    # 2. Compute normal velocities (u dot n)
    un_minus = jnp.sum(p_minus.u * nxyz, axis=0)
    un_plus = jnp.sum(p_plus.u * nxyz, axis=0)

    # 3. Compute Roe Averages to estimate outer wave speeds
    H_minus = (q_minus.E + p_minus.p) / p_minus.rho
    H_plus = (q_plus.E + p_plus.p) / p_plus.rho
    c_minus = jnp.sqrt(gamma * p_minus.p / p_minus.rho)
    c_plus = jnp.sqrt(gamma * p_plus.p / p_plus.rho)

    sqrt_rho_minus = jnp.sqrt(p_minus.rho)
    sqrt_rho_plus = jnp.sqrt(p_plus.rho)
    sqrt_rho_sum = sqrt_rho_minus + sqrt_rho_plus

    u_roe = (sqrt_rho_minus * p_minus.u + sqrt_rho_plus * p_plus.u) / sqrt_rho_sum
    H_roe = (sqrt_rho_minus * H_minus + sqrt_rho_plus * H_plus) / sqrt_rho_sum

    un_roe = jnp.sum(u_roe * nxyz, axis=0)
    u_roe_sq = jnp.sum(u_roe * u_roe, axis=0)
    c_roe = jnp.sqrt((gamma - 1.0) * (H_roe - 0.5 * u_roe_sq))

    # Outer wave speeds
    S_L = jnp.minimum(un_minus - c_minus, un_roe - c_roe)
    S_R = jnp.maximum(un_plus + c_plus, un_roe + c_roe)

    # 4. Compute Contact Wave Speed (S_M)
    # A_L and A_R are strictly < 0 and > 0 respectively.
    A_L = p_minus.rho * (S_L - un_minus)
    A_R = p_plus.rho * (S_R - un_plus)

    # denom_SM is strictly positive bounded by density and sound speeds,
    # but we add a safety floor for perfect vacuum conditions
    denom_SM = jnp.maximum(A_R - A_L, 1e-12)
    S_M = (A_R * un_plus - A_L * un_minus + p_minus.p - p_plus.p) / denom_SM

    # 5. Star Region Pressure
    p_star = p_minus.p + A_L * (S_M - un_minus)

    # 6. Project standard Cartesian fluxes onto the normal vector
    fn_rho_m = jnp.sum(flux_minus.rho * nxyz, axis=0)
    fn_rho_p = jnp.sum(flux_plus.rho * nxyz, axis=0)

    # For momentum flux tensor, multiply nxyz across the flux direction (axis=1)
    fn_rhou_m = jnp.sum(flux_minus.rhou * nxyz[None, ...], axis=1)
    fn_rhou_p = jnp.sum(flux_plus.rhou * nxyz[None, ...], axis=1)

    fn_E_m = jnp.sum(flux_minus.E * nxyz, axis=0)
    fn_E_p = jnp.sum(flux_plus.E * nxyz, axis=0)

    # 7. Compute intermediate Star Fluxes (F_L* and F_R*)
    # Safe denominators since S_L <= S_M <= S_R
    denom_L = jnp.minimum(S_L - S_M, -1e-12)
    denom_R = jnp.maximum(S_R - S_M, 1e-12)

    factor_L = S_L / denom_L
    factor_R = S_R / denom_R

    # Density Star Fluxes
    fstar_rho_m = fn_rho_m + factor_L * ((S_M - un_minus) * q_minus.rho)
    fstar_rho_p = fn_rho_p + factor_R * ((S_M - un_plus) * q_plus.rho)

    # Momentum Star Fluxes (Broadcasting applied to mapping onto spatial dimensions)
    fstar_rhou_m = fn_rhou_m + factor_L[None, ...] * (
        (S_M - un_minus)[None, ...] * q_minus.rhou
        + (p_star - p_minus.p)[None, ...] * nxyz
    )
    fstar_rhou_p = fn_rhou_p + factor_R[None, ...] * (
        (S_M - un_plus)[None, ...] * q_plus.rhou + (p_star - p_plus.p)[None, ...] * nxyz
    )

    # Energy Star Fluxes
    fstar_E_m = fn_E_m + factor_L * (
        (S_M - un_minus) * q_minus.E + p_star * S_M - p_minus.p * un_minus
    )
    fstar_E_p = fn_E_p + factor_R * (
        (S_M - un_plus) * q_plus.E + p_star * S_M - p_plus.p * un_plus
    )

    # 8. Branching Logic (Select correct flux based on wave states)
    cond_L = S_L >= 0.0
    cond_star_L = (S_L < 0.0) & (S_M >= 0.0)
    cond_star_R = (S_M < 0.0) & (S_R >= 0.0)

    def _select_flux(fL, fstarL, fstarR, fR, is_vector=False):
        # Expand condition arrays to cover vector dimensions if evaluating momentum
        cL = cond_L[None, ...] if is_vector else cond_L
        csL = cond_star_L[None, ...] if is_vector else cond_star_L
        csR = cond_star_R[None, ...] if is_vector else cond_star_R

        return jnp.where(cL, fL, jnp.where(csL, fstarL, jnp.where(csR, fstarR, fR)))

    flux_rho = _select_flux(
        fn_rho_m, fstar_rho_m, fstar_rho_p, fn_rho_p, is_vector=False
    )
    flux_rhou = _select_flux(
        fn_rhou_m, fstar_rhou_m, fstar_rhou_p, fn_rhou_p, is_vector=True
    )
    flux_E = _select_flux(fn_E_m, fstar_E_m, fstar_E_p, fn_E_p, is_vector=False)

    return EulerVarCons(flux_rho, flux_rhou, flux_E)


_BD_FLUX_FN = {
    "LF": _lax_friedrichs_flux,
    "Roe": _roe_flux,
    "HLL": _hll_flux,
    "HLLC": _hllc_flux,
}


@jax.jit
def _apply_vmap(var, vmap):
    return jax.tree.map(lambda x: x.reshape(*x.shape[:-2], -1)[..., vmap], var)


@partial(jax.jit, static_argnames=["dims"])
def _add_bd_flux(rhs, bd_flux, ops, dims):
    def _reshape(fx):
        # compact the Nf and Nfp dimensions into one, to be able to apply the flux to the rhs
        return fx.reshape(*fx.shape[:-3], dims.Nf * dims.Nfp, dims.K)

    return jax.tree.map(
        lambda r, fx: r - ops.lift @ (ops.fscale * _reshape(fx)),
        rhs,
        bd_flux,
    )


def enforce_positivity(mdata, var, gamma):
    p = _c2p(var, gamma)
    bad_cells = jnp.any((p.rho <= 0) | (p.p <= 0), axis=0)

    def _enforce_positivity_var(x):
        avg_phi = mdata.avg_phi if x.ndim == 2 else mdata.avg_phi[None, ...]
        return jnp.where(bad_cells, avg_phi @ x, x)

    return jax.tree.map(_enforce_positivity_var, var)


_PROFILER = defaultdict(list)

from time import perf_counter


def _rhs(ops, dims, q, time, bd_flux_type, gamma, bc_fn):
    global _PROFILER
    t0 = perf_counter()
    p = _c2p(q, gamma)
    p.rho.block_until_ready()
    t1 = perf_counter()
    _PROFILER["c2p"].append(t1 - t0)

    t0 = perf_counter()
    flux = _c2flux(q, p)
    flux.rho.block_until_ready()
    t1 = perf_counter()
    _PROFILER["c2flux"].append(t1 - t0)

    t0 = perf_counter()
    rhs = _tree_div(ops.J_rst_xyz, ops.Dphi_weak, flux)
    rhs.rho.block_until_ready()
    t1 = perf_counter()
    _PROFILER["div"].append(t1 - t0)

    t0 = perf_counter()
    q_plus = _apply_vmap(q, ops.vmap_p)
    q_plus.rho.block_until_ready()
    t1 = perf_counter()
    _PROFILER["vmap_p"].append(t1 - t0)

    t0 = perf_counter()
    q_minus = _apply_vmap(q, ops.vmap_m)
    q_minus.rho.block_until_ready()
    t1 = perf_counter()
    _PROFILER["vmap_m"].append(t1 - t0)

    t0 = perf_counter()
    q_plus = bc_fn(ops, dims, q_plus, time)
    q_plus.rho.block_until_ready()
    t1 = perf_counter()
    _PROFILER["bc_fn_plus"].append(t1 - t0)

    t0 = perf_counter()
    bd_flux = _BD_FLUX_FN[bd_flux_type](ops, dims, q_plus, q_minus, gamma)
    bd_flux.rho.block_until_ready()
    t1 = perf_counter()
    _PROFILER["bd_flux"].append(t1 - t0)

    t0 = perf_counter()
    rhs = _add_bd_flux(rhs, bd_flux, ops, dims)
    rhs.rho.block_until_ready()
    t1 = perf_counter()
    _PROFILER["add_bd_flux"].append(t1 - t0)
    return rhs


def step(ops, dims, q, time, dt, bd_flux_type, gamma, bc_fn, stepper_type):
    def rhs_fn(q, t):
        return _rhs(
            ops,
            dims,
            q,
            t,
            bd_flux_type,
            gamma,
            bc_fn,
        )
    
    def postproc(q):
        return enforce_positivity(ops, q, gamma)

    return get_timestepper(stepper_type)(q, time, dt, rhs_fn, postproc)


class Euler:
    def __init__(self, mesh_ops, params):
        self.mesh_ops = mesh_ops
        self.params = params
