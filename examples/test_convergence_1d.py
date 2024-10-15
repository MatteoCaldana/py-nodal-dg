# -*- coding: utf-8 -*-
import time

import matplotlib.pyplot as plt
import numpy as np

from pyndg.scalar_param_1d import ScalarParam1D
from pyndg.scalar_1d import Scalar1D

from pyndg.physic_model import Advection1D
from pyndg.viscosity_model import NoViscosity
from pyndg.time_integrator import LS54
from pyndg.bc_1d import BC

def calc_convergence(KK, table, plot):
    convergence = np.diff(np.log(table), axis=0) / -np.diff(np.log(KK))[:, None]
    if plot:
        N = table.shape[1]
        KK = np.array(KK)
        expected_rate = np.power.outer(KK, -(2.0 + np.arange(N)))
        expected_rate = expected_rate * table[0, :] / expected_rate[0, :]
        plt.plot(1.0 / KK, table, "-o", linewidth=2)
        plt.plot(1.0 / KK, expected_rate, "k--", zorder=-100, linewidth=2)
        plt.yscale("log")
        plt.xscale("log")
        plt.grid("on")
        plt.title(f"Convergence of {plot}D Advection")
        plt.xlabel("Mesh size $h$")
        plt.ylabel("$L^2$ Error")
        plt.legend(
            [f"k={i+1}" for i in range(N)] + ["$h^{k+1}$ rate"],
            loc="lower right",
            ncol=2,
            fontsize=12,
        )
        plt.tight_layout()
        plt.xlim([min(2e-2, min(1/KK)*0.8), 0.3])
    return convergence


def convergence(params, sol, KK, NN, plot=False):
    table = np.empty((len(KK), len(NN)))
    for i in range(len(KK)):
        for j in range(len(NN)):
            print(KK[i], NN[j])
            params["N"] = NN[j]
            params["K"] = KK[i]
            problem = Scalar1D(ScalarParam1D(**params))

            while problem.step():
                pass

            u_ex = sol(problem.mesh.x, params["final_time"])
            e = u_ex - problem.u

            def norm(x):
                return np.sqrt(np.sum((x.T @ problem.mesh.M) * x.T))

            err2 = norm(e) / norm(u_ex)
            ##
            table[i, j] = err2

    return table, calc_convergence(KK, table, plot)


def test_convergence_linear_advection():
    def ic(x):
        return np.sin(2 * np.pi * x)

    def u_ex(x, t):
        return ic(x - t)

    params = {
        "model": Advection1D(),
        "name": "Test",
        "N": -1,
        "K": -1,
        "u_IC": ic,
        "bnd": (0.0, 1.0),
        "mesh_pert": 0.0,
        "bc": ((BC.Periodic, 0.0), (BC.Periodic, 0.0)),
        "final_time": 0.2,
        "cfl": 0.1,
        "time_integrator": LS54(),
        "viscosity_model": NoViscosity(),
    }

    KK = [5, 10, 20, 40, 80]
    NN = [1, 2, 3, 4, 5]
    return convergence(params, u_ex, KK, NN, "1")


if __name__ == "__main__":
    import pyndg.backend as bkd
    assert bkd.BACKEND == bkd.NUMPY

    t0 = time.time()
    err, conv = test_convergence_linear_advection()
    t1 = time.time()
    print(f"Elapsed: {t1 - t0}")
    plt.show()
