import jax
import jax.numpy as jnp
from functools import partial

import pyndg.backend as bkd


def ssp2(u, time, dt, f, postproc=lambda x: x):
    u1 = postproc(u + dt * f(u, time))
    return postproc(0.5 * u + 0.5 * u1 + 0.5 * dt * f(u1, time + dt))


def ssp3(u, time, dt, f, postproc=lambda x: x):
    u1 = postproc(u + dt * f(u, time))
    u2 = postproc((3.0 * u + u1 + dt * f(u1, time + dt)) / 4.0)
    return postproc((u + 2.0 * u2 + 2.0 * dt * f(u2, time + 0.5 * dt)) / 3.0)


_A_LS54 = jnp.array(
    [
        0,
        -0.4178904745,
        -1.192151694643,
        -1.697784692471,
        -1.514183444257,
    ],
    dtype=bkd.jnp_prec,
)

_B_LS54 = jnp.array(
    [
        0.1496590219993,
        0.3792103129999,
        0.8229550293869,
        0.6994504559488,
        0.1530572479681,
    ],
    dtype=bkd.jnp_prec,
)

_C_LS54 = jnp.array(
    [
        0,
        0.1496590219993,
        0.3704009573644,
        0.6222557631345,
        0.9582821306748,
    ],
    dtype=bkd.jnp_prec,
)


def ls54(u, time, dt, f, postproc=lambda x: x):
    v = jnp.zeros_like(u)
    for i in range(5):
        stage_time = time + _C_LS54[i] * dt
        y = f(u, stage_time)
        v = _A_LS54[i] * v + dt * y
        u = postproc(u + _B_LS54[i] * v)
    return u


_TIMESTEPPERS = {"ssp3": ssp3, "ls54": ls54, "ssp2": ssp2}


def get_timestepper(name):
    global _TIMESTEPPERS
    return _TIMESTEPPERS[name]
