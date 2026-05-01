# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import numpy as np

from . import backend as bkd
from .backend import bkd as bk


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
    A = np.array([0, -0.4178904745, -1.192151694643, -1.697784692471, -1.514183444257])
    B = np.array(
        [
            0.1496590219993,
            0.3792103129999,
            0.8229550293869,
            0.6994504559488,
            0.1530572479681,
        ]
    )
    # C = np.array(
    #     [0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306748]
    # )

    @staticmethod
    def integrate(u0, dt, f):
        u = bkd.copy(u0)
        v = bk.zeros_like(u)
        for i in range(5):
            y = f(u)
            v = LS54.A[i] * v + dt * y
            u = u + LS54.B[i] * v
        return u


def test_integrator(integrator, dt, timesteps, u0, f, y, plot=False):
    timesteps = int(timesteps)
    u = np.empty((timesteps + 1,))

    u[0] = u0
    for i in range(timesteps):
        u[i + 1] = integrator.integrate(u[i], dt, f)
    if plot:
        t = np.arange(timesteps + 1) * dt
        plt.plot(t, u, "-o")
        plt.plot(t, y(t))
    return np.abs(u[-1] - y(dt * timesteps))


def test1(alpha=1, u0=1):
    def f(y):
        return alpha * y

    def y(t):
        return u0 * np.exp(alpha * t)

    return u0, f, y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    T = 2
    dtv = 0.1 / 2 ** np.arange(10)
    errv_3 = [test_integrator(SSP3(), dt, T / dt, *test1()) for dt in dtv]
    errv_4 = [test_integrator(LS54(), dt, T / dt, *test1()) for dt in dtv]

    plt.plot(dtv, errv_3)
    plt.plot(dtv, errv_4)
    for i in range(4):
        plt.plot(dtv, dtv ** (i + 1), "--k")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
