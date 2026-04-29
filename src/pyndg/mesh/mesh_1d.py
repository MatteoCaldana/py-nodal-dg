import numpy as np


def generate_mesh_1d(xmin, xmax, K, mesh_per):
    Nv = K + 1
    h = (xmax - xmin) / K
    pert_scale = mesh_per * h
    VX = np.linspace(xmin, xmax, Nv)

    if mesh_per > 0:
        pert = (np.random.rand(Nv - 2) - 0.5) * pert_scale
        VX[1:-1] += pert
    hK = VX[1:] - VX[:-1]
    return Nv, VX, hK


class Mesh1D:
    def __init__(self, xmin, xmax, K, mesh_per):
        self.Nv, self.VX, self.hK = generate_mesh_1d(xmin, xmax, K, mesh_per)
