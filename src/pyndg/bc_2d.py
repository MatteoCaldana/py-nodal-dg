# -*- coding: utf-8 -*-
from enum import IntEnum


class BC(IntEnum):
    In = 1
    Out = 2
    Slip = 3
    Far = 4
    Dirichlet = 5
    Sym = 6
    Periodic = 7
    Wall = 8
    Cyl = 9
    Neuman = 10
