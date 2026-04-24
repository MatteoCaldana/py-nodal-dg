from pyndg.core.basis import nodes_2d, nodes_3d, vandermonde_3d, xy_to_rs, vandermonde_2d, xyz_to_rst

import numpy as np


def equispaced_triangle_grid(n):
    t = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(t, t)
    r = X.flatten()
    s = Y.flatten()
    mask = (r >= -1) & (s >= -1) & (r + s <= 0)
    return r[mask], s[mask]


def equispaced_tetrahedron_grid(n):
    t = np.linspace(-1, 1, n)
    X, Y, Z = np.meshgrid(t, t, t)
    r = X.flatten()
    s = Y.flatten()
    t = Z.flatten()
    mask = (r >= -1) & (s >= -1) & (t >= -1) & (r + s + t <= -1)
    return r[mask], s[mask], t[mask]

N_max = 5

print("Lebesgue constants for 2D triangle:")
for N in range(1, N_max + 1):
    # 1. Define your interpolation nodes (already done in your snippet)
    x, y = nodes_2d(N)
    r, s = xy_to_rs(x, y)
    V = vandermonde_2d(N, r, s)

    # 2. Generate a very dense sampling grid (e.g., degree M >> N)
    # These points must be inside the triangle
    r_f, s_f = equispaced_triangle_grid(N * 20)

    # 3. Compute Vandermonde matrix at the fine sampling points
    V_fine = vandermonde_2d(N, r_f, s_f)

    # 4. Solve for the Lagrange basis values at the fine points
    # L_ij is the value of the j-th Lagrange polynomial at the i-th fine point
    L = np.linalg.solve(V.T, V_fine.T).T

    # 5. Compute the Lebesgue function at each fine point
    # Sum of absolute values of Lagrange polynomials at each point
    lebesgue_function = np.sum(np.abs(L), axis=1)

    # 6. The Lebesgue Constant is the maximum of this function
    lebesgue_constant = np.max(lebesgue_function)

    print(f"N={N:02}, Lebesgue Constant (Λ): {lebesgue_constant:.4f}")


print("Lebesgue constants for 3D tetrahedron:")
for N in range(1, N_max + 1):
    # 1. Define your interpolation nodes (already done in your snippet)
    x, y, z = nodes_3d(N)
    r, s, t = xyz_to_rst(x, y, z)
    V = vandermonde_3d(N, r, s, t)

    # 2. Generate a very dense sampling grid (e.g., degree M >> N)
    # These points must be inside the triangle
    r_f, s_f, t_f = equispaced_tetrahedron_grid(N * 20)

    # 3. Compute Vandermonde matrix at the fine sampling points
    V_fine = vandermonde_3d(N, r_f, s_f, t_f)

    # 4. Solve for the Lagrange basis values at the fine points
    # L_ij is the value of the j-th Lagrange polynomial at the i-th fine point
    L = np.linalg.solve(V.T, V_fine.T).T

    # 5. Compute the Lebesgue function at each fine point
    # Sum of absolute values of Lagrange polynomials at each point
    lebesgue_function = np.sum(np.abs(L), axis=1)

    # 6. The Lebesgue Constant is the maximum of this function
    lebesgue_constant = np.max(lebesgue_function)

    print(f"N={N:02}, Lebesgue Constant (Λ): {lebesgue_constant:.4f}")
