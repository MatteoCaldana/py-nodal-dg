#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import jax
import jax.numpy as jnp

NUMPY = 0
TORCH = 1
JAX = 2

if "PYNDG_BACKEND" in os.environ:
    backend_str = os.environ["PYNDG_BACKEND"]
    if backend_str == "numpy":
        BACKEND = NUMPY
    elif backend_str == "torch":
        BACKEND = TORCH
    elif backend_str == "jax":
        BACKEND = JAX
    else:
        raise ValueError(f"PYNDG_BACKEND={backend_str} is not a valid backend")
else:
    print("WARNING: 'PYNDG_BACKEND' not found, setting by default to torch")
    BACKEND = TORCH


if BACKEND == NUMPY:
    import numpy as bkd
if BACKEND == TORCH:
    import torch as bkd
if BACKEND == JAX:
    import jax.numpy as bkd


if (BACKEND != NUMPY) and ("PYNDG_PREC" in os.environ):
    PREC = os.environ["PYNDG_PREC"]
else:
    print("WARNING: 'PYNDG_PREC' not found, setting by default to f64")
    PREC = "f64"

if PREC == "f64":
    if BACKEND == TORCH:
        bkd.set_default_dtype(bkd.float64)
    if BACKEND == JAX:
        jax.config.update("jax_enable_x64", True)


TO_CONST = np.array(
    [
        lambda x: x,
        lambda x: (
            x.clone().detach()
            if isinstance(x, bkd.Tensor)
            else bkd.tensor(x).clone().detach()
        ),
        lambda x: x,
    ]
)


def to_const(x):
    return TO_CONST[BACKEND](x)


TO_VARIABLE = np.array(
    [
        lambda x: x,
        lambda x: bkd.tensor(x, requires_grad=True, device="cpu"),
        lambda x: x,
    ]
)


def to_variable(x):
    return TO_VARIABLE[BACKEND](x)


TO_NUMPY = np.array([lambda x: x, lambda x: x.detach().numpy(), np.array])


def to_numpy(x):
    return TO_NUMPY[BACKEND](x)


COPY = np.array([np.copy, lambda x: torch.clone(x), jnp.copy])


def copy(x):
    return COPY[BACKEND](x)


def maxval(*args, **kwargs):
    if (BACKEND == NUMPY) or (BACKEND == JAX):
        return bkd.max(*args, **kwargs)
    else:
        return bkd.max(*args, **kwargs).values


def minval(*args, **kwargs):
    if (BACKEND == NUMPY) or (BACKEND == JAX):
        return bkd.min(*args, **kwargs)
    else:
        return bkd.min(*args, **kwargs).values


def std(*args, **kwargs):
    if (BACKEND == NUMPY) or (BACKEND == JAX):
        return bkd.std(*args, **kwargs)
    else:
        return bkd.std(*args, correction=0, **kwargs)


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def th_reshape(a, shape, order="C"):
    if order == "C":
        return torch.reshape(a, shape)
    elif order == "F":
        return reshape_fortran(a, shape)
    else:
        raise ValueError(f"Unknown order {order}")


RESHAPE = np.array([np.reshape, th_reshape, jnp.reshape])


def reshape(*args, **kwargs):
    return RESHAPE[BACKEND](*args, **kwargs)
