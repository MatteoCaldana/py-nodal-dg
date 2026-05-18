from enum import IntEnum


class BC(IntEnum):
    NONE = 0
    Dirichlet = 1
    Neumann = 2
    Robin = 3
    Periodic = 4


def invert_dir_neu_bc(bc_tags):
    inverted_bc_tags = {}
    for tag, bc in bc_tags.items():
        if bc == BC.Dirichlet:
            inverted_bc_tags[tag] = BC.Neumann
        elif bc == BC.Neumann:
            inverted_bc_tags[tag] = BC.Dirichlet
        else:
            inverted_bc_tags[tag] = bc
    return inverted_bc_tags
