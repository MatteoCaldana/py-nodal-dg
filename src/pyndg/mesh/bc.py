from enum import IntEnum


class BC(IntEnum):
    NONE = 0
    Neumann = 1
    Dirichlet = 2
    Robin = 3
    Periodic = 4