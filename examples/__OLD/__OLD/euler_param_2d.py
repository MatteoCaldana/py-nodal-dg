# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Callable, Union

from .utils import check_type
from .viscosity_model import ViscosityModel2D
from .time_integrator import TimeIntegrator
from .bc_2d import BC
from .physic_model import PhysicModel2D
from .backend import bkd as bk


class EulerPhysics2D(PhysicModel2D):
    def __init__(self, gas_gamma):
        self.gas_gamma = gas_gamma

    def eig(self, u):
        return bk.sqrt(u[1] * u[1] + u[2] * u[2]) / bk.abs(u[0])

    def idd(self, u):
        # viscosity has to be computed using density
        return u[0]

    def primitive(self, u):
        return u[0], u[1] / u[0], u[2] / u[0], self.pre(u)

    def flux(self, u):
        rho, vu, vv, p = self.primitive(u)
        F = (u[1], u[1] * vu + p, u[2] * vu, vu * (u[3] + p))
        G = (u[2], u[1] * vv, u[2] * vv + p, vv * (u[3] + p))
        return rho, vu, vv, p, F, G

    def dflux(self, u):
        # TODO: implement for real
        return 1.0

    def pre(self, u):
        return (self.gas_gamma - 1) * (u[3] - 0.5 * (u[1] * u[1] + u[2] * u[2]) / u[0])

    def entropy(self, u):
        return (
            u[0]
            / (self.gas_gamma - 1)
            * bk.log(bk.abs(self.pre(u) / (u[0] ** self.gas_gamma)))
        )

    def entropy_flux(self, u):
        entropy = self.entropy(u)
        return u[1] / u[0] * entropy, u[2] / u[0] * entropy


@dataclass(frozen=True)
class EulerParam2D:
    gas_const: float
    gas_gamma: float
    name: str
    N: int
    mesh_file: Union[str, tuple]
    uIC: Callable
    bc: dict
    final_time: float
    time_integrator: TimeIntegrator
    viscosity_model: ViscosityModel2D
    cfl: float = -1.0
    dt: float = -1.0
    dt_fixed: bool = False
    model: EulerPhysics2D = EulerPhysics2D(0.0)

    def __post_init__(self):
        check_type(self)

        if self.N <= 0:
            raise ValueError("Field 'N' cannot be non-positive.")

        n_periodic = 0
        for bc_key in self.bc:
            if not isinstance(bc_key, int) or not isinstance(self.bc[bc_key], BC):
                raise ValueError("Field 'bc' must be dict[int, BC].")
            n_periodic += self.bc[bc_key] == BC.Periodic
        object.__setattr__(self, "has_periodic_bc", n_periodic > 0)

        if self.final_time <= 0:
            raise ValueError("Field 'final_time' must be positive.")
        if self.cfl * self.dt >= 0:
            raise ValueError(
                "Must define or 'cfl' or 'dt' as positive "
                "for timestepping (the negative will be ignored)."
            )

        self.model.gas_gamma = self.gas_gamma
