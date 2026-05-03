PATH = "/home/matteo/Documents/nodal-dg/Codes1.1/Grid/"
filename = "Other/block2.neu"

# PATH = "/home/matteo/Documents/dgann/Examples/2D/Euler/Shock_vortex/"
# filename = "square.msh"

from pyndg.scalar_param_2d import ScalarParam2D
from pyndg.mesh_2d import Mesh2D

from pyndg.time_integrator import LS54
from pyndg.physic_model import Advection2D
from pyndg.bc_2d import BC
from pyndg.limiters import Limiter
from pyndg.viscosity_model import NoViscosity

import numpy as np
import pandas as pd

params = ScalarParam2D(
    model=Advection2D(),
    name="Test",
    N=5,
    mesh_file=PATH + filename,
    u_IC=lambda x: x,
    bc={},  # {101: BC.Out, 102: BC.Slip, 103: BC.In, 104: BC.Slip},
    final_time=1.0,
    cfl=1.0,
    time_integrator=LS54(),
    viscosity_model=NoViscosity(),
    limiter=Limiter(),
)

mesh = Mesh2D(params)
mesh.initialize()
# mesh.plot(show_elem_id=False, show_vtx_id=False)

n_elem = mesh.K
n_vtx = mesh.VX.size
n_faces_with_dupes = n_elem * 3
n_bnd_faces = np.sum(mesh.BCTag != 0)
n_int_faces = (n_faces_with_dupes - n_bnd_faces) // 2
n_faces = n_bnd_faces + n_int_faces

n_vxt_bnd = n_bnd_faces + 1  # approx, one bnd (eg, wrong for cyl)
n_vtx_int = n_vtx - n_vxt_bnd


print("n_elem", n_elem)
print("n_vtx", n_vtx)
print("n_faces_with_dupes", n_faces_with_dupes)
print("n_bnd_faces", n_bnd_faces)
print("n_faces", n_faces)

cost_table = {
    "lin": lambda x: x,
    "sq": lambda x: x * x,
    "slin": lambda x: x * np.log(x),
}

data = []
for solver_cost in cost_table:
    for N in range(1, 16):
        nodes_per_elem = (N + 1) * (N + 2) // 2

        n_elem_tmp = n_elem / 4
        n_faces_tmp = n_faces / 2
        n_vtx_tmp = n_vtx - n_faces_tmp

        for href in [0, 1, 2, 3]:
            n_elem_tmp *= 4
            n_vtx_tmp += n_faces_tmp
            n_faces_tmp *= 2

            total_nodes = nodes_per_elem * n_elem_tmp

            # approx of comp cost, should be better with higher N due to exponential convergence
            cg_nodes = (
                n_vtx_tmp
                + n_faces_tmp * max(0, N - 1)
                + n_elem_tmp * max(0, (N - 2) * (N - 1) // 2)
            )

            diam = 1 / 2**href
            n_tsteps = (N + 1) * (N + 1) / diam

            cost = cost_table[solver_cost](total_nodes) * n_tsteps

            data.append(
                {
                    "N": N,
                    "cost": cost,
                    "accuracy": cg_nodes,
                    "solver_cost": solver_cost,
                    "href": href,
                }
            )


df = pd.DataFrame(data)

import matplotlib.pyplot as plt
import seaborn as sns

# Create a single row of subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
solver_types = df['solver_cost'].unique()
cmap = plt.get_cmap("jet")

for i, solver in enumerate(solver_types):
    subset = df[df['solver_cost'] == solver]
    ax = axes[i]
    
    # Connect dots with the same N (Refinement paths)
    # units="N" ensures lines only connect points within the same N group
    sns.lineplot(data=subset, x="cost", y="accuracy", hue="N", palette=cmap, 
                 legend=False, ax=ax, units="N", estimator=None, alpha=0.4)
    
    # Scatter points for the actual data
    sns.scatterplot(data=subset, x="cost", y="accuracy", hue="N", style="href", 
                    palette=cmap, ax=ax, legend=(i == 2))
    
    ax.set_title(f"Solver: {solver}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="-", alpha=0.2)

# Position the legend outside the last subplot
axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="N (Color) / href (Marker)")

plt.tight_layout()
plt.show()