# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Callable

from .utils import check_type
from .viscosity_model import ViscosityModel1D
from .time_integrator import TimeIntegrator
from .bc_1d import BC
from .physic_model import PhysicModel1D

from .backend import bkd as bk


class EulerPhysics1D(PhysicModel1D):
    def __init__(self, gas_gamma):
        self.gas_gamma = gas_gamma
        
    def idd(self, u):
        return u[0]

    def flux(self, u):
        raise NotImplementedError()

    def dflux(self, u):
        return 1.0
        raise NotImplementedError()

    def pre(self, u):
        return (self.gas_gamma - 1) * (u[2] - 0.5 * u[1] * u[1] / u[0])

    def entropy(self, u):
        return (
            u[0]
            / (self.gas_gamma - 1)
            * bk.log(bk.abs(self.pre(u) / (u[0] ** self.gas_gamma)))
        )

    def entropy_flux(self, u):
        return u[1] / u[0] * self.entropy(u)


@dataclass(frozen=True)
class EulerParam1D:
    gas_const: float
    gas_gamma: float
    name: str
    N: int
    K: int
    rho_IC: Callable
    vel_IC: Callable
    pre_IC: Callable
    bnd: tuple
    mesh_pert: float
    bcs: list
    final_time: float
    time_integrator: TimeIntegrator
    viscosity_model: ViscosityModel1D
    cfl: float = -1.0
    dt: float = -1.0
    dt_fixed: bool = False
    model: EulerPhysics1D = EulerPhysics1D(0.0)

    def __post_init__(self):
        check_type(self)

        if self.N <= 0:
            raise ValueError("Field 'N' cannot be non-positive.")
        if self.K <= 0:
            raise ValueError("Field 'K' cannot be non-positive.")
        if (
            len(self.bnd) != 2
            or not isinstance(self.bnd[0], float)
            or not isinstance(self.bnd[1], float)
        ):
            raise ValueError("Field 'bnd' must be tuple[float, float].")
        if self.bnd[1] <= self.bnd[0]:
            raise ValueError("Field 'bnd' must be increasing")
        if self.mesh_pert < 0 or self.mesh_pert >= 1:
            raise ValueError("Field 'mesh_pert' must be in [0, 1).")
        for bc in self.bcs:
            if len(bc) != 2:
                raise ValueError("Field 'bc' must have 2 entries (left and right).")
            for i in range(2):
                if not isinstance(bc[i], tuple):
                    raise ValueError(f"Field 'bc'[{i}] must be tuple")
                if len(bc[i]) != 2:
                    raise ValueError(f"Field 'bc'[{i}] must have len == 2")
                if not isinstance(bc[i][0], BC):
                    raise ValueError(f"Field 'bc'[{i}][0] must be BC")
                if not isinstance(bc[i][1], float):
                    raise ValueError(f"Field 'bc'[{i}][1] must be float")
        if self.final_time <= 0:
            raise ValueError("Field 'final_time' must be positive.")
        if self.cfl * self.dt >= 0:
            raise ValueError(
                "Must define or 'cfl' or 'dt' as positive "
                "for timestepping (the negative will be ignored)."
            )

        self.model.gas_gamma = self.gas_gamma


# TODO: limiters
if __name__ == "__main__":
    from viscosity_model import EV1D
    from time_integrator import SSP3

    def id(x):
        return x

    p = EulerParam1D(
        gas_const=1.0,
        gas_gamma=1.0,
        name="Test",
        rho_IC=id,
        vel_IC=id,
        pre_IC=id,
        N=1,
        K=1,
        bnd=(0.0, 1.0),
        mesh_pert=0.0,
        bc=[
            ((BC.Periodic, 0.0), (BC.Periodic, 0.0)),
            ((BC.Periodic, 0.0), (BC.Periodic, 0.0)),
            ((BC.Periodic, 0.0), (BC.Periodic, 0.0)),
        ],
        final_time=0.2,
        cfl=0.1,
        time_integrator=SSP3(),
        viscosity_model=EV1D(c_E=1.0, c_max=0.5),
    )
