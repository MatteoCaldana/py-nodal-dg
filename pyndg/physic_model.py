# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np

from .backend import bkd as bk
from . import backend as bkd


class PhysicModel(ABC):
    @staticmethod
    @abstractmethod
    def flux(u):
        pass

    @staticmethod
    @abstractmethod
    def dflux(u):
        pass

    @staticmethod
    @abstractmethod
    def entropy(u):
        pass

    @staticmethod
    @abstractmethod
    def entropy_flux(u):
        pass


class PhysicModel1D(PhysicModel):
    pass


class PhysicModel2D(PhysicModel):
    @staticmethod
    @abstractmethod
    def eig(u):
        pass


class Advection1D(PhysicModel1D):
    C = 1

    @staticmethod
    def idd(u):
        return u

    @staticmethod
    def flux(u):
        return Advection1D.C * u

    @staticmethod
    def dflux(u):
        return bk.ones((1,), dtype=u.dtype)

    @staticmethod
    def entropy(u):
        return 0.5 * u * u

    @staticmethod
    def entropy_flux(u):
        return Advection1D.C * 0.5 * u * u


class Burgers1D(PhysicModel1D):
    @staticmethod
    def idd(u):
        return u

    @staticmethod
    def flux(u):
        return 0.5 * u * u

    @staticmethod
    def dflux(u):
        return u

    @staticmethod
    def entropy(u):
        return 0.5 * u * u

    @staticmethod
    def entropy_flux(u):
        return u * u * u / 3


class BuckLev1D(PhysicModel1D):
    A = 0.5

    @staticmethod
    def flux(u):
        return u * u / (u * u + BuckLev1D.A * (1 - u) ** 2)

    @staticmethod
    def dflux(u):
        return 2 * BuckLev1D.A * u * (1 - u) / (u * u + BuckLev1D.A * (1 - u) ** 2) ** 2

    @staticmethod
    def entropy(u):
        return 0.5 * u * u

    @staticmethod
    def entropy_flux(u):
        return (
            -BuckLev1D.A
            / (BuckLev1D.A + 1) ** 2
            * (
                (BuckLev1D.A * (2 - 3 * u) + u) / (BuckLev1D.A * (u - 1) ** 2 + u * u)
                + bk.log(u * u + BuckLev1D.A * (1 - u) ** 2)
                + (BuckLev1D.A - 1)
                / bk.sqrt(BuckLev1D.A)
                * bk.atan((BuckLev1D.A * (u - 1) + u) / bk.sqrt(BuckLev1D.A))
            )
        )


class FourthOrder1D(PhysicModel1D):
    @staticmethod
    def flux(u):
        return u * u * u * u / 4

    @staticmethod
    def dflux(u):
        return u * u * u

    @staticmethod
    def entropy(u):
        return 0.5 * u * u

    @staticmethod
    def entropy_flux(u):
        return u * u * u * u * u / 5


PHYSIC_MODEL_1D_TABLE = {"Advection": Advection1D(), "Burgers": Burgers1D()}

###############################################################################


class Advection2D(PhysicModel2D):
    C = bkd.to_const(np.array([1, 1]))
    
    @staticmethod
    def idd(u):
        return u

    @staticmethod
    def eig(u):
        return bk.sqrt(bk.sum(Advection2D.C ** 2))

    @staticmethod
    def flux(u):
        return (Advection2D.C[i] * u for i in [0, 1])

    @staticmethod
    def dflux(u):
        return 1.0

    @staticmethod
    def entropy(u):
        return 0.5 * u * u

    @staticmethod
    def entropy_flux(u):
        return (Advection2D.C[i] * Advection2D.entropy(u) for i in [0, 1])


class Burgers2D(PhysicModel2D):
    @staticmethod
    def eig(u):
        return bk.abs(u)

    @staticmethod
    def flux(u):
        return (u * u / 2.0, u * u / 2.0)

    @staticmethod
    def dflux(u):
        raise NotImplementedError("In 2D dflux should not be needed")

    @staticmethod
    def entropy(u):
        return 0.5 * u * u

    @staticmethod
    def entropy_flux(u):
        return (u * u * u / 3.0, u * u * u / 3.0)


class KPP2D(PhysicModel2D):
    @staticmethod
    def idd(u):
        return u

    @staticmethod
    def eig(u):
        return bkd.to_const(1.0)

    @staticmethod
    def flux(u):
        return (bk.sin(u), bk.cos(u))

    @staticmethod
    def dflux(u):
        return 1.0

    @staticmethod
    def entropy(u):
        return 0.5 * u * u

    @staticmethod
    def entropy_flux(u):
        return (u * bk.sin(u) + bk.cos(u), u * bk.cos(u) - bk.sin(u))
