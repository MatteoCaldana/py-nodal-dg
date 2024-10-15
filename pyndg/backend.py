#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np

NUMPY = 0
TORCH = 1

if "PYNDG_BACKEND" in os.environ:
    backend_str = os.environ["PYNDG_BACKEND"]
    if backend_str == "numpy":
        BACKEND = NUMPY
    elif backend_str == "torch":
        BACKEND = TORCH
    else:
        raise ValueError(f"PYNDG_BACKEND={backend_str} is not a valid backend")
else:
    print("WARNING: could not find env variable 'PYNDG_BACKEND', setting by default to torch")
    BACKEND = TORCH

if BACKEND == NUMPY:
    import numpy as bkd
if BACKEND == TORCH:
    import torch as bkd

    bkd.set_default_dtype(bkd.float64)


TO_CONST = np.array(
    [
        lambda x: x,
        lambda x: (
            x.clone().detach()
            if isinstance(x, bkd.Tensor)
            else bkd.tensor(x).clone().detach()
        ),
    ]
)


def to_const(x):
    return TO_CONST[BACKEND](x)


TO_VARIABLE = np.array(
    [
        lambda x: x,
        lambda x: bkd.tensor(x, requires_grad=True, device="cpu"),
    ]
)


def to_variable(x):
    return TO_VARIABLE[BACKEND](x)


TO_NUMPY = np.array([lambda x: x, lambda x: x.detach().numpy()])


def to_numpy(x):
    return TO_NUMPY[BACKEND](x)


COPY = np.array([np.copy, lambda x: bkd.clone(x)])


def copy(x):
    return COPY[BACKEND](x)


def maxval(*args, **kwargs):
    if BACKEND == NUMPY:
        return np.max(*args, **kwargs)
    else:
        return bkd.max(*args, **kwargs).values


def minval(*args, **kwargs):
    if BACKEND == NUMPY:
        return np.min(*args, **kwargs)
    else:
        return bkd.min(*args, **kwargs).values


def std(*args, **kwargs):
    if BACKEND == NUMPY:
        return np.std(*args, **kwargs)
    else:
        return bkd.std(*args, correction=0, **kwargs)
