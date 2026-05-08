"""
Preconditioned Conjugate Gradient (PCG) solver in JAX.

Both the linear operator A and the preconditioner M are passed as callables,
making this matrix-free and compatible with JAX JIT / vmap.

Algorithm (left-preconditioned CG, a.k.a. the standard PCG):
  Given: A x = b, with A s.p.d. and M ≈ A^{-1} s.p.d.
  r₀ = b - A x₀
  z₀ = M r₀
  p₀ = z₀

  For k = 0, 1, …:
    αk  = (rk · zk) / (pk · A pk)
    x_{k+1} = xk + αk pk
    r_{k+1} = rk - αk A pk
    z_{k+1} = M r_{k+1}
    βk  = (r_{k+1} · z_{k+1}) / (rk · zk)
    p_{k+1} = z_{k+1} + βk pk
"""

from typing import Callable, NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class PCGState(NamedTuple):
    """Carries all mutable state across iterations."""

    x: jax.Array  # current solution estimate
    r: jax.Array  # residual  r = b - A x
    z: jax.Array  # preconditioned residual  z = M r
    p: jax.Array  # search direction
    rz: jax.Array  # scalar  rᵀ z  (cached to avoid recomputation)
    iteration: jax.Array
    converged: jax.Array


class PCGResult(NamedTuple):
    x: jax.Array  # solution
    residual_norm: jax.Array
    iterations: jax.Array
    converged: jax.Array


def pcg(
    A: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    M: Callable[[jax.Array], jax.Array] | None = None,
    x0: jax.Array | None = None,
    tol: float = 1e-6,
    atol: float = 0.0,
    max_iter: int | None = None,
) -> PCGResult:
    """
    Preconditioned Conjugate Gradient solver.

    Parameters
    ----------
    A        : Callable (n,) -> (n,)   s.p.d. linear operator
    b        : jax.Array of shape (n,) right-hand side
    M        : Callable (n,) -> (n,)   s.p.d. preconditioner (≈ A⁻¹);
               pass None to use the identity (plain CG).
    x0       : initial guess; defaults to zeros
    tol      : relative residual tolerance  ‖r‖ / ‖b‖ < tol
    atol     : absolute residual tolerance  ‖r‖ < atol
    max_iter : maximum iterations; defaults to len(b)

    Returns
    -------
    PCGResult with fields (x, residual_norm, iterations, converged)
    """
    b = jnp.asarray(b)
    n = b.shape[0]

    if M is None:
        M = lambda v: v  # identity preconditioner → plain CG

    if x0 is None:
        x0 = jnp.zeros_like(b)

    if max_iter is None:
        max_iter = n

    b_norm = jnp.linalg.norm(b)
    tol_eff = jnp.maximum(tol * b_norm, atol)  # combined stopping threshold

    # -----------------------------------------------------------------------
    # Initialise
    # -----------------------------------------------------------------------
    r0 = b - A(x0)
    z0 = M(r0)
    p0 = z0
    rz0 = jnp.dot(r0, z0)

    init_state = PCGState(
        x=x0,
        r=r0,
        z=z0,
        p=p0,
        rz=rz0,
        iteration=jnp.zeros((), dtype=jnp.int32),
        converged=jnp.zeros((), dtype=jnp.bool_),
    )

    # -----------------------------------------------------------------------
    # One CG step
    # -----------------------------------------------------------------------
    def step(state: PCGState) -> PCGState:
        Ap = A(state.p)
        alpha = state.rz / jnp.dot(state.p, Ap)

        x_new = state.x + alpha * state.p
        r_new = state.r - alpha * Ap
        z_new = M(r_new)
        rz_new = jnp.dot(r_new, z_new)

        beta = rz_new / state.rz
        p_new = z_new + beta * state.p

        converged = jnp.linalg.norm(r_new) < tol_eff

        return PCGState(
            x=x_new,
            r=r_new,
            z=z_new,
            p=p_new,
            rz=rz_new,
            iteration=state.iteration + 1,
            converged=converged,
        )

    # -----------------------------------------------------------------------
    # Continue while: not converged AND iteration < max_iter
    # -----------------------------------------------------------------------
    def cond(state: PCGState) -> jax.Array:
        return (~state.converged) & (state.iteration < max_iter)

    final: PCGState = lax.while_loop(cond, step, init_state)

    return PCGResult(
        x=final.x,
        residual_norm=jnp.linalg.norm(final.r),
        iterations=final.iteration,
        converged=final.converged,
    )


# ---------------------------------------------------------------------------
# Convenience: JIT-compiled version
# ---------------------------------------------------------------------------
pcg_jit = jax.jit(pcg, static_argnames=("A", "M", "max_iter"))


# ---------------------------------------------------------------------------
# Numpy
# ---------------------------------------------------------------------------


def pcg_np(
    A,
    b,
    M=None,
    x0=None,
    tol: float = 1e-6,
    atol: float = 0.0,
    max_iter: int | None = None,
):

    n = b.shape[0]

    if M is None:
        M = lambda v: v  # identity preconditioner → plain CG

    if x0 is None:
        x0 = np.zeros_like(b)

    if max_iter is None:
        max_iter = n

    b_norm = np.linalg.norm(b)
    tol_eff = np.maximum(tol * b_norm, atol)  # combined stopping threshold

    # -----------------------------------------------------------------------
    # Initialise
    # -----------------------------------------------------------------------
    r0 = b - A(x0)
    z0 = M(r0)
    p0 = z0
    rz0 = np.dot(r0, z0)
    init_converged = np.linalg.norm(r0) < tol_eff

    init_state = PCGState(
        x=x0,
        r=r0,
        z=z0,
        p=p0,
        rz=rz0,
        iteration=np.zeros((), dtype=np.int32),
        converged=init_converged,
    )

    # -----------------------------------------------------------------------
    # One CG step
    # -----------------------------------------------------------------------
    def step(state: PCGState) -> PCGState:
        Ap = A(state.p)
        alpha = state.rz / np.dot(state.p, Ap)

        x_new = state.x + alpha * state.p
        r_new = state.r - alpha * Ap
        z_new = M(r_new)
        rz_new = np.dot(r_new, z_new)

        beta = rz_new / state.rz
        p_new = z_new + beta * state.p

        converged = np.linalg.norm(r_new) < tol_eff

        return PCGState(
            x=x_new,
            r=r_new,
            z=z_new,
            p=p_new,
            rz=rz_new,
            iteration=state.iteration + 1,
            converged=converged,
        )

    def cond_fun(state: PCGState):
        return (~state.converged) & (state.iteration < max_iter)

    val = init_state
    while cond_fun(val):
        val = step(val)

    return PCGResult(
        x=val.x,
        residual_norm=np.linalg.norm(val.r),
        iterations=val.iteration,
        converged=val.converged,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(0)

    n = 200

    # Build a random s.p.d. matrix
    _F = rng.standard_normal((n, n))
    _A = _F @ _F.T + n * np.eye(n)  # well-conditioned
    _A_jax = jnp.array(_A)

    b = jnp.array(rng.standard_normal(n))

    # ------------------------------------------------------------------
    # Example 1 – plain CG (M = identity)
    # ------------------------------------------------------------------
    A_fn = lambda v: _A_jax @ v

    result_cg = pcg(A_fn, b, M=None, tol=1e-10)
    print("=== Plain CG ===")
    print(f"  converged   : {result_cg.converged}")
    print(f"  iterations  : {result_cg.iterations}")
    print(f"  ‖r‖         : {result_cg.residual_norm:.2e}")
    print(
        f"  ‖x - x*‖   : {jnp.linalg.norm(result_cg.x - jnp.linalg.solve(_A_jax, b)):.2e}"
    )

    # ------------------------------------------------------------------
    # Example 2 – diagonal (Jacobi) preconditioner
    # ------------------------------------------------------------------
    _diag_inv = jnp.array(1.0 / np.diag(_A))
    M_fn = lambda v, _: _diag_inv * v  # diagonal scaling

    result_pcg = pcg(A_fn, b, M=M_fn, tol=1e-10)
    print("\n=== PCG (Jacobi preconditioner) ===")
    print(f"  converged   : {result_pcg.converged}")
    print(f"  iterations  : {result_pcg.iterations}")
    print(f"  ‖r‖         : {result_pcg.residual_norm:.2e}")
    print(
        f"  ‖x - x*‖   : {jnp.linalg.norm(result_pcg.x - jnp.linalg.solve(_A_jax, b)):.2e}"
    )

    # ------------------------------------------------------------------
    # Example 3 – JIT compiled
    # ------------------------------------------------------------------
    result_jit = pcg_jit(A_fn, b, M=M_fn, tol=1e-10)
    print("\n=== PCG JIT (same result) ===")
    print(
        f"  ‖x - x*‖   : {jnp.linalg.norm(result_jit.x - jnp.linalg.solve(_A_jax, b)):.2e}"
    )
