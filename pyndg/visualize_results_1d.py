#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch as th

import matplotlib.pyplot as plt
from matplotlib import cm, ticker
import time

from .mesh_1d import jacobiGL
from .backend import bkd as bk
from . import backend as bkd


V_INVS = []


def build_Vinvs():
    global V_INVS
    for N in range(8):
        if N == 0:
            V_INVS.append(None)
        else:
            Np = N + 1
            r = jacobiGL(0, 0, N)
            V = np.ones((Np, Np))
            for i in range(1, Np):
                V[:, i] = V[:, i - 1] * r
            Vinv = np.linalg.inv(V)
            V_INVS.append(Vinv)


build_Vinvs()


def lglexp(x, y, n=30):
    x = np.array(bkd.to_const(x))
    y = np.array(bkd.to_const(y))
    t = np.linspace(-1, 1, n)
    Np = x.shape[0]
    N = Np - 1
    lgl = np.empty((Np, n))
    for i in range(Np):
        lgl[i, :] = np.polyval(V_INVS[N][::-1, i], t)

    a, b = (x[-1, :] + x[0, :]) / 2, (x[-1, :] - x[0, :]) / 2
    xx = t.reshape((-1, 1)) * b.reshape((1, -1)) + a.reshape((1, -1))
    yy = np.einsum("ji,jk->ki", y, lgl)
    return xx, yy


def lglexp_at(x, y, t):
    n = t.numel()
    Np = x.shape[0]
    N = Np - 1
    lgl = np.empty((Np, n))
    for i in range(Np):
        lgl[i, :] = np.polyval(V_INVS[N][::-1, i], t)

    a, b = (x[-1, :] + x[0, :]) / 2, (x[-1, :] - x[0, :]) / 2
    xx = t.reshape((-1, 1)) * b.reshape((1, -1)) + a.reshape((1, -1))
    yy = bk.einsum("ji,jk->ki", y, bkd.to_const(lgl))
    return xx, yy


def problem_to_meshgrid(problem, n):
    problem.reset()
    D = 1 if len(problem.u.shape) == 2 else problem.u.shape[0]

    x = np.array(problem.mesh.x).reshape((-1,), order="F")
    t = np.empty((n,))
    u = np.empty((D, t.size, x.size))
    v = np.empty((t.size, x.size))

    for i in range(n):
        problem.step()
        t[i] = problem.time
        for k in range(D):
            ui = problem.u.reshape((D, *problem.mesh.x.shape))
            u[k, i, :] = np.array(ui[k]).reshape((-1,), order="F")
        v[i, :] = np.array(problem.mu_vals).reshape((-1,), order="F")

    xx, tt = np.meshgrid(x, t)
    return xx, tt, u, v


def contourf(problems, n, save=False):
    for p in problems:
        p.reset()
    D = 1 if len(problems[0].u.shape) == 2 else problems[0].u.shape[0]
    numpr = len(problems)
    FIG_CONST_SIZE = 4

    for k in range(D + 2):
        fig, ax = plt.subplots(
            numpr, 1, figsize=(FIG_CONST_SIZE, 0.7 * FIG_CONST_SIZE * numpr)
        )
        if not isinstance(ax, np.ndarray):
            ax = np.array([ax])
        cmap = cm.get_cmap("jet")

        max_vv = -1
        min_uu, max_uu = np.array((np.inf,) * D), np.array((-np.inf,) * D)
        for i, p in enumerate(problems):
            xx, tt, uu, vv = problem_to_meshgrid(p, n[i])
            max_vv = max([max_vv, vv.max()])
            max_uu = np.maximum(max_uu, uu.max(axis=(1, 2)))
            min_uu = np.minimum(min_uu, uu.min(axis=(1, 2)))

        for i, p in enumerate(problems):
            xx, tt, uu, vv = problem_to_meshgrid(p, n[i])

            if k == D:
                cs = ax[i].contourf(
                    xx, tt, vv, vmax=max_vv, vmin=0, levels=20, cmap=cmap
                )
                fig.colorbar(cs, ax=ax[i])
            elif k == D + 1:
                try:
                    cs = ax[i].contourf(
                        xx,
                        tt,
                        vv,
                        levels=10.0
                        ** np.arange(-13, np.ceil(np.log10(max_vv)) + 1, 0.5),
                        locator=ticker.LogLocator(),
                        cmap=cmap,
                    )
                    fig.colorbar(cs, ax=ax[i])
                except Exception as e:
                    print(f"{e}")
            else:
                cs = ax[i].contourf(
                    xx, tt, uu[k], vmin=min_uu[k], vmax=max_uu[k], cmap=cmap, levels=40
                )
                fig.colorbar(cs, ax=ax[i])

        plt.tight_layout()
