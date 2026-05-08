def jacobi(A_fn, D_inv, b, x0, num_iters, omega):
    x = x0.copy()
    for _ in range(num_iters):
        r = b - A_fn(x)
        x += omega * D_inv * r
    return x


def chebyshev(Afn, b, x0, num_iters, lambda_min, lambda_max):
    x = x0.copy()

    # spectral interval parameters
    d = 0.5 * (lambda_max + lambda_min)
    c = 0.5 * (lambda_max - lambda_min)

    # initial residual
    r = b - Afn(x)

    # first step
    alpha = 1.0 / d
    p = alpha * r
    x += p

    if num_iters == 1:
        return x

    beta = 0.5 * (c * alpha) ** 2
    for _ in range(1, num_iters):
        r = b - Afn(x)
        alpha_new = 1.0 / (d - beta)
        beta = (0.5 * c * alpha_new) ** 2
        p = alpha_new * r + beta * p
        x += p
        alpha = alpha_new

    return x

def chebyshev_v2(A_fn, b, x, nu, lam_min, lam_max):
    theta = (lam_max + lam_min) / 2
    delta = (lam_max - lam_min) / 2
    
    r = b - A_fn(x)
    u_hat = r / theta
    x = x + u_hat
    
    if nu == 1:
        return x
        
    rho_prev = 1.0
    for _ in range(2, nu + 1):
        rho = 1.0 / (2.0 * theta / delta - rho_prev)
        r = b - A_fn(x)
        u_hat = rho * (2.0 / delta * r + rho_prev * u_hat)
        x = x + u_hat
        rho_prev = rho
        
    return x


def bjacobi(A_fn, Db_inv, b, x0, num_iters, omega):
    x = x0.copy()
    for _ in range(num_iters):
        r = b - A_fn(x)
        x_new = Db_inv @ r.reshape(Db_inv.shape[0], Db_inv.shape[2], 1)
        x += omega * x_new.reshape(-1)
    return x
