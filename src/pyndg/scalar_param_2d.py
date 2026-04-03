# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Callable, Union

from .utils import check_type
from .physic_model import PhysicModel2D
from .viscosity_model import ViscosityModel2D
from .time_integrator import TimeIntegrator
from .bc_2d import BC
from .limiters import Limiter


@dataclass(frozen=True)
class ScalarParam2D:
    model: PhysicModel2D
    name: str
    N: int
    mesh_file: Union[str, tuple]
    u_IC: Callable
    bc: dict
    time_integrator: TimeIntegrator
    viscosity_model: ViscosityModel2D
    limiter: Limiter
    final_time: float
    cfl: float = -1.0
    dt: float = -1.0
    dt_fixed: bool = True

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
        if n_periodic % 2 == 1:
            raise ValueError(
                "The number of periodic face tags should be a multiple of 2."
            )
        if self.final_time <= 0:
            raise ValueError("Field 'final_time' must be positive.")
        if self.cfl * self.dt >= 0:
            raise ValueError("Must define or 'cfl' or 'dt'.")
