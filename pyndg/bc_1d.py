# -*- coding: utf-8 -*-
from enum import Enum
from .backend import bkd as bk


class BC(Enum):
    Periodic = 1
    Dirichlet = 2
    Neumann = 3

def inv_BC(bc):
    if bc == BC.Neumann:
        return BC.Dirichlet
    else:
        return BC.Neumann

def apply_BC_1D(u, bc):
    if bc[0][0] == BC.Periodic:
        assert bc[1][0] == BC.Periodic
        return bk.hstack([u[:, -1:], u, u[:, 0:1]])
    if bc[0][0] == BC.Neumann:
        ul = bk.flipud(u[:, 0:1])
    else:  # Dirichlet
        ul = -bk.flipud(u[:, 0:1]) + 2 * bc[0][1]
    if bc[1][0] == BC.Neumann:
        ur = bk.flipud(u[:, -1:])
    else:  # Dirichlet
        ur = -bk.flipud(u[:, -1:]) + 2 * bc[1][1]
    return bk.hstack([ul, u, ur])
