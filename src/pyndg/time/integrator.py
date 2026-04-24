# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import jax.numpy as jnp
import pyndg.backend as bkd


class TimeIntegrator(ABC):
    @staticmethod
    @abstractmethod
    def integrate(u, dt, f):
        pass


class SSP3(TimeIntegrator):
    @staticmethod
    def integrate(u, dt, f):
        u1 = u + dt * f(u)
        u2 = (3 * u + u1 + dt * f(u1)) / 4
        return (u + 2 * u2 + 2 * dt * f(u2)) / 3


class LS54(TimeIntegrator):
    A = jnp.array(
        [
            0,
            -0.4178904745,
            -1.192151694643,
            -1.697784692471,
            -1.514183444257,
        ],
        dtype=bkd.jnp_prec,
    )
    B = jnp.array(
        [
            0.1496590219993,
            0.3792103129999,
            0.8229550293869,
            0.6994504559488,
            0.1530572479681,
        ],
        dtype=bkd.jnp_prec,
    )

    @staticmethod
    def integrate(u, dt, f):
        v = jnp.zeros_like(u)
        for i in range(5):
            y = f(u)
            v = LS54.A[i] * v + dt * y
            u = u + LS54.B[i] * v
        return u