import os
import jax
import jax.numpy as jnp
import numpy as np

PRECOMPUTE_PRECISION = os.environ.get("PYNDG_PRECOMPUTE_PRECISION", "f64")
RUN_COMPUTE_PRECISION = os.environ.get("PYNDG_RUN_COMPUTE_PRECISION", "f64")


_STR_TO_NP_PRECISION = {
    "f32": np.float32,
    "f64": np.float64,
    "f128": np.float128,
}

_STR_TO_JAX_PRECISION = {
    "f32": jnp.float32,
    "f64": jnp.float64,
}

np_prec = _STR_TO_NP_PRECISION[PRECOMPUTE_PRECISION]
jnp_prec = _STR_TO_JAX_PRECISION[RUN_COMPUTE_PRECISION]

if RUN_COMPUTE_PRECISION == "f64":
    jax.config.update("jax_enable_x64", True)
