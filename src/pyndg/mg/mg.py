import scipy.sparse.linalg


def mg_iter(A, b, x, smoother, P, R, Ac):
    Np_f, Np_c = P.shape
    K = x.size // Np_f

    # x = smoother(A, b, x)
    r = b - A @ x
    r_c = R @ r.reshape(Np_f, K, order="F")
    e_c = scipy.sparse.linalg.spsolve(Ac, r_c.flatten(order="F"))
    e_f = P @ e_c.reshape(Np_c, K, order="F")
    x = x + e_f.flatten(order="F")
    # x = smoother(A, b, x)
    return x
