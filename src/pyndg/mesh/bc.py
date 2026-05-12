from enum import IntEnum


class BC(IntEnum):
    NONE = 0
    Dirichlet = 1
    Neumann = 2
    Robin = 3
    Periodic = 4
