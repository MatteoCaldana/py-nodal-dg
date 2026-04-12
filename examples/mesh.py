PATH = "/home/matteo/Documents/nodal-dg/Codes1.1/Grid/"
filename = "Euler2D/FS_949.neu"

# PATH = "/home/matteo/Documents/dgann/Examples/2D/Euler/Shock_vortex/"
# filename = "square.msh"

from pyndg.scalar_param_2d import ScalarParam2D
from pyndg.mesh_2d import Mesh2D

from pyndg.time_integrator import LS54
from pyndg.physic_model import Advection2D
from pyndg.bc_2d import BC
from pyndg.limiters import Limiter
from pyndg.viscosity_model import NoViscosity

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
mesh.plot(show_elem_id=False, show_vtx_id=False)
