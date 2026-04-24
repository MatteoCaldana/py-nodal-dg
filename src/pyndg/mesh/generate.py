import gmsh
import numpy as np


def foo(callback=None):
    # Initialize gmsh
    gmsh.initialize()
    gmsh.model.add("Channel")

    # Parameters
    L = 1.0
    H = 1.0
    n = 5

    # --- Geometry definition (Points) ---
    # gmsh.model.geo.addPoint(x, y, z, meshSize, tag)
    gmsh.model.geo.addPoint(0, 0, 0, 1.0, 1)
    gmsh.model.geo.addPoint(L, 0, 0, 1.0, 2)
    gmsh.model.geo.addPoint(L, H, 0, 1.0, 3)
    gmsh.model.geo.addPoint(0, H, 0, 1.0, 4)

    # --- Geometry definition (Lines) ---
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)

    # --- Surface definition ---
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    # --- Transfinite (Structured Mesh) constraints ---
    # The sign in the .geo script (e.g., -3) indicates orientation,
    # which is handled here by the point ordering or explicit mesh constraints.
    gmsh.model.geo.mesh.setTransfiniteCurve(1, n)
    gmsh.model.geo.mesh.setTransfiniteCurve(2, n)
    gmsh.model.geo.mesh.setTransfiniteCurve(3, n)
    gmsh.model.geo.mesh.setTransfiniteCurve(4, n)
    gmsh.model.geo.mesh.setTransfiniteSurface(1, "Left", [1, 2, 3, 4])

    # Synchronize CAD kernel to the model before adding Physical Groups/Periodicity
    gmsh.model.geo.synchronize()

    # --- Physical Groups ---
    # (dim, tags, tag_id)
    gmsh.model.addPhysicalGroup(2, [1], 100000)  # Surface
    gmsh.model.addPhysicalGroup(1, [4], 100001)  # Left
    gmsh.model.addPhysicalGroup(1, [2], 100002)  # Right
    gmsh.model.addPhysicalGroup(1, [1], 100003)  # Bottom
    gmsh.model.addPhysicalGroup(1, [3], 100004)  # Top

    # --- Periodic Boundaries ---
    # Periodic Line {1} = {-3} means Line 1 is a slave to Line 3
    # shifted by a translation vector (0, H, 0).
    # The transformation matrix is a 4x4 row-major affine matrix.

    # Translation for Bottom (1) to Top (3)
    translation_top = [1, 0, 0, 0, 0, 1, 0, H, 0, 0, 1, 0, 0, 0, 0, 1]

    # Translation for Right (2) to Left (4)
    translation_left = [1, 0, 0, -L, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

    gmsh.model.mesh.setPeriodic(1, [3], [1], translation_top)
    gmsh.model.mesh.setPeriodic(1, [4], [2], translation_left)

    # --- Generate Mesh ---
    gmsh.model.mesh.generate(2)

    if callback is not None:
        callback(gmsh)
    else:
        gmsh.write("tmp.msh")

    # Always finalize when done
    gmsh.finalize()

    return
