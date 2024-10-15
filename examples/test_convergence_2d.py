# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from pyndg.scalar_param_2d import ScalarParam2D
from pyndg.scalar_2d import Scalar2D
from pyndg.physic_model import Advection2D
from pyndg.viscosity_model import NoViscosity
from pyndg.time_integrator import LS54
from pyndg.bc_2d import BC
from pyndg.limiters import Limiter

from test_convergence_1d import calc_convergence


def convergence(plot=False):
    def sol(x, y):
        return 1 + np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

    KK = [5, 10, 20, 40, 80]
    NN = [1, 2, 3]
    table = np.empty((len(KK), len(NN)))
    for i in range(len(KK)):
        for j in range(len(NN)):
            print(KK[i], NN[j])
            problem = Scalar2D(
                ScalarParam2D(
                    model=Advection2D(),
                    name="Test",
                    N=NN[j],
                    mesh_file=((0, 0), (1, 1), (KK[i],)*2),
                    u_IC=sol,
                    bc={int(k + 1e5): BC.Periodic for k in [1, 2, 3, 4]},
                    final_time=0.5,
                    cfl=0.1 + (NN[j] - 1) / 20,
                    time_integrator=LS54(),
                    viscosity_model=NoViscosity(),
                    limiter=Limiter(),
                )
            )

            while problem.step():
                pass
            T = problem.params.final_time
            u_ex = sol(problem.mesh.x - T, problem.mesh.y - T)
            e = u_ex - problem.u
            def norm(x):
                return np.sqrt(np.sum((x.T @ problem.mesh.M) * x.T))

            err2 = norm(e) / norm(u_ex)
            ##
            table[i, j] = err2

    return table, calc_convergence(KK, table, plot)


if __name__ == "__main__":
    import pyndg.backend as bkd
    assert bkd.BACKEND == bkd.NUMPY
    
    tab, conv = convergence(plot=2)
    plt.show()
