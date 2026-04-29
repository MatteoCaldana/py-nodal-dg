from pyndg.mesh.reader import read_gmsh_file_2d, mesh_reader_gambit

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors


def read_mesh(filename):
    readers = {"msh": read_gmsh_file_2d, "neu": mesh_reader_gambit}
    ext = filename.split(".")[-1]

    fields = readers[ext](filename)
    return Mesh2D(**fields)


def check_mesh_orientation(VXY, EToV):
    # Extract the coordinates for each vertex of every triangle
    p1 = VXY[EToV[:, 0]]
    p2 = VXY[EToV[:, 1]]
    p3 = VXY[EToV[:, 2]]

    # Compute the 2D cross product (twice the signed area)
    d1 = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1])
    d2 = (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1])
    val = d1 - d2

    # Find indices of improperly oriented triangles
    degenerate_triangles = np.where(val < 1e-7)[0]

    if len(degenerate_triangles) == 0:
        print("All triangles are properly oriented (CCW).")
    else:
        print(f"WARNING! Found {len(degenerate_triangles)} degenerate triangles.")

    return degenerate_triangles



class Mesh2D:
    def __init__(self, VXY, K, Nv, EToV, b_faces, PerBToB, PerBFToF):
        self.VXY = VXY
        self.K = K
        self.Nv = Nv
        self.EToV = EToV
        self.BFaces = b_faces
        self.PerBToB = PerBToB
        self.PerBFToF = PerBFToF

    def plot(self, show_elem_id=True, show_vtx_id=True):
        _, ax = plt.subplots(figsize=(10, 8))

        # Extract all triangle edges from EToV
        all_edges = []
        for tri in self.EToV:
            # tri contains indices of the 3 vertices
            pts = self.VXY[tri]
            all_edges.append((pts[0], pts[1]))
            all_edges.append((pts[1], pts[2]))
            all_edges.append((pts[2], pts[0]))

        # Plot mesh in light gray to make BCs stand out
        mesh_lc = LineCollection(
            all_edges, colors="lightgray", linewidths=0.5, alpha=1.0
        )
        ax.add_collection(mesh_lc)

        # Plot Boundary Edges by Tag
        color_cycle = list(mcolors.TABLEAU_COLORS.values())
        for i, (tag, faces) in enumerate(self.BFaces.items()):
            bc_edges = []
            for node_pair in faces:
                p1 = self.VXY[node_pair[0]]
                p2 = self.VXY[node_pair[1]]
                bc_edges.append((p1, p2))

            color = color_cycle[i % len(color_cycle)]
            bc_lc = LineCollection(
                bc_edges,
                colors=color,
                linewidths=2.5,
                label=f"Tag ID={tag}, type={tag}",
            )
            ax.add_collection(bc_lc)

        if show_elem_id:
            for eid, tri in enumerate(self.EToV):
                pts = self.VXY[tri].mean(axis=0)
                plt.text(pts[0], pts[1], f"{eid}")

        for eid in self.problematic_alpha_elements:
            tri = self.EToV[eid, :]
            pts = self.VXY[tri].mean(axis=0)
            plt.scatter([pts[0]], [pts[1]], marker="o", s=100, color="r")

        if show_vtx_id:
            for i in range(self.VXY.shape[0]):
                plt.text(self.VXY[i, 0], self.VXY[i, 1], f"{i}", color="r")

        ax.set_aspect("equal")
        ax.autoscale()
        plt.title(f"2D Mesh: K={self.K}, N={self.N}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
        plt.tight_layout()
        plt.show()
