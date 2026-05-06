def jacobi(A_fn, D_inv, b, x0, num_iters, omega):
    x = x0.copy()
    for _ in range(num_iters):
        r = b - A_fn(x)
        x = (1 - omega) * x + omega * D_inv * r
    return x


def chebyshev(A_fn, b, x0, num_iters, lambda_min, lambda_max):
    x = x0.copy()
    d = (lambda_max + lambda_min) / 2
    c = (lambda_max - lambda_min) / 2

    rho_prev = None
    x_prev = None

    for i in range(num_iters):
        r = b - A_fn(x)

        if i == 0:
            rho = 1 / d
            x = x + rho * r
        elif i == 1:
            rho = 1 / (d - (c**2) / (2 * d))
            x_new = rho * (d * x - (c**2) / 2 * x_prev + r) / d
            x_prev = x
            x = x_new
        else:
            rho = 1 / (d - (c**2) / 4 * rho_prev)
            x_new = rho * (d * x - (c**2) / 4 * rho_prev * x_prev + r)
            x_prev = x
            x = x_new

        rho_prev = rho

    return x


def bjacobi(A_fn, Db_inv, b, x0, num_iters, omega):
    x = x0.copy()
    b_size = Db_inv.shape[2]
    for _ in range(num_iters):
        r = b - A_fn(x)
        x_new = Db_inv @ r.reshape((b_size, -1, 1))
        x = (1 - omega) * x + omega * x_new.squeeze(-1)
    return x
