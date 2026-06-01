import jax.numpy as jnp
import numpy as np

import pyndg.backend as bkd
from pyndg.time.integrator import _TIMESTEPPERS


# 1. Autonomous System: du/dt = \lambda * u
def autonomous_f(u, t):
    lambda_val = -0.5
    return lambda_val * u


def autonomous_exact(u0, t):
    lambda_val = -0.5
    return u0 * jnp.exp(lambda_val * t)


# 2. Non-Autonomous System: du/dt = u*cos(t) + sin(2t) - cos(t)*sin(t)
def non_autonomous_f(u, t):
    return u * jnp.cos(t) + jnp.sin(2 * t) - jnp.cos(t) * jnp.sin(t)


def non_autonomous_exact(u0, t):
    # Analytical solution given u(0) = u0
    # derived via integrating factor
    return (u0 + 1.0) * jnp.exp(jnp.sin(t)) - jnp.sin(t) - 1.0


# --- Integration Loop Helper ---


def integrate_to_tensor(stepper_fn, f, u0, t0, tf, dt):
    """Integrates from t0 to tf using a fixed dt."""
    n_steps = int(jnp.round((tf - t0) / dt))
    # Recalculate precise dt to avoid floating point drift at tf
    dt_actual = (tf - t0) / n_steps

    u = jnp.array(u0, dtype=bkd.jnp_prec)
    t = t0

    for _ in range(n_steps):
        u = stepper_fn(u, t, dt_actual, f)
        t += dt_actual

    return u


TESTS = [
    ("ssp3", _TIMESTEPPERS["ssp3"], 3),
    ("ls54", _TIMESTEPPERS["ls54"], 4),
    ("ssp2", _TIMESTEPPERS["ssp2"], 2),
]

# --- Main Verification Routine ---
def test():
    # Enable Float64 for high accuracy checking

    problems = {
        "Autonomous System": {
            "f": autonomous_f,
            "exact": autonomous_exact,
            "u0": jnp.array([1.5], dtype=bkd.jnp_prec),
            "dts": [0.4, 0.2, 0.1, 0.05, 0.025],
        },
        "Non-Autonomous System": {
            "f": non_autonomous_f,
            "exact": non_autonomous_exact,
            "u0": jnp.array([1.0], dtype=bkd.jnp_prec),
            # Shorter time-steps for ls54 to clear the asymptotic error regime
            "dts": [0.2, 0.1, 0.05, 0.025, 0.0125],
        },
    }

    t0 = 0.0
    tf = 2.0

    print("=" * 65)
    print(f"CONVERGENCE TEST (t0 = {t0}, tf = {tf})")
    print("=" * 65)

    for prob_name, prob_data in problems.items():
        print(f"\n--- Problem: {prob_name} ---")
        f = prob_data["f"]
        exact_fn = prob_data["exact"]
        u0 = prob_data["u0"]
        dts = prob_data["dts"]

        u_exact = exact_fn(u0, tf)

        for name, stepper_fn, expected_order in TESTS:
            errors = []

            for dt in dts:
                u_num = integrate_to_tensor(stepper_fn, f, u0, t0, tf, dt)
                # Compute L-infinity error
                error = jnp.max(jnp.abs(u_num - u_exact))
                errors.append(float(error))

            # Compute empirical order of convergence via least-squares slope of log(error) vs log(dt)
            log_dts = np.log(dts)
            log_errors = np.log(errors)

            # Polyfit fits y = mx + c; index 0 is slope m
            order, _ = np.polyfit(log_dts, log_errors, 1)

            print(f"  Stepper: {name:<6} -> Calculated Convergence Order: {order:.4f}")
            assert order > expected_order - 0.1, f"Expected order ~{expected_order}, got {order:.4f}"
            # Optional: print individual rates to look for asymptotic behavior
            for i in range(len(errors) - 1):
                rate = np.log(errors[i] / errors[i + 1]) / np.log(dts[i] / dts[i + 1])
                print(f"    dt = {dts[i]:.4f} -> Rate: {rate:.4f}")
