from pyndg.mesh.generate import generate_rectangular_mesh
from pyndg.mesh import Mesh

if __name__ == "__main__":
    e2v, vxy, boundary_data = generate_rectangular_mesh(1.0, 1.0, 10, structured=True)
    mesh = Mesh(vxy, e2v, boundary_data, None, None)
    mesh.plot()
