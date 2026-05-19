TODO:
infra:
- add ruff

mesh:
- gmsh in python
- test mesh 2d periodic gmsh
- test mesh 1d
- test mesh global h-refinement
- test mesh reorder

core:
- jax sparse utils
- jax.scan assembly loop with block

experimental:
- local h-refinement

physics:
- ins poisson solvers (with cholmod+richardson and MG in jax)
- shockless euler
