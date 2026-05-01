# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Callable

from .utils import check_type
from .physic_model import PhysicModel
from .viscosity_model import ViscosityModel1D
from .time_integrator import TimeIntegrator
from .bc_1d import BC


@dataclass(frozen=True)
class ScalarParam1D:
    model: PhysicModel
    name: str
    N: int
    K: int
    u_IC: Callable
    bnd: tuple
    mesh_pert: float
    bc: tuple
    final_time: float
    time_integrator: TimeIntegrator
    viscosity_model: ViscosityModel1D
    cfl: float = -1.0
    dt: float = -1.0
    dt_fixed: bool = False
    smooth_viscosity: bool = True

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
        if len(self.bc) != 2:
            raise ValueError("Field 'bc' must have 2 entries (left and right).")
        for i in range(2):
            if not isinstance(self.bc[i], tuple):
                raise ValueError(f"Field 'bc'[{i}] must be tuple")
            if len(self.bc[i]) != 2:
                raise ValueError(f"Field 'bc'[{i}] must have len == 2")
            if not isinstance(self.bc[i][0], BC):
                raise ValueError(f"Field 'bc'[{i}][0] must be BC")
            if not isinstance(self.bc[i][1], float):
                raise ValueError(f"Field 'bc'[{i}][1] must be float")
        if self.final_time <= 0:
            raise ValueError("Field 'final_time' must be positive.")
        if self.cfl * self.dt >= 0:
            raise ValueError(
                "Must define or 'cfl' or 'dt' as positive "
                "for timestepping (the negative will be ignored)."
            )
