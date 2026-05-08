import scipy.sparse.linalg


def build_prolongator_restrictor(mesh_ops_c, mesh_ops_f):
    Icf = mesh_ops_c.ref_elem_ops.build_interp_mat(mesh_ops_f.ref_elem_ops.rst)
    P = Icf
    R = (
        mesh_ops_c.ref_elem_ops.int_phiphi_inv
        @ Icf.T
        @ mesh_ops_f.ref_elem_ops.int_phiphi
    )
    return P, R


def mg_iter(A_fn, b, x, smoother, P, R, Ac):
    Np_f, Np_c = P.shape
    K = x.size // Np_f

    x = smoother(A_fn, b, x)
    r = b - A_fn(x)
    r_c = R @ r.reshape(Np_f, K, order="F")
    e_c = scipy.sparse.linalg.spsolve(Ac, r_c.flatten(order="F"))
    e_f = P @ e_c.reshape(Np_c, K, order="F")
    x = x + e_f.flatten(order="F")
    x = smoother(A_fn, b, x)
    return x
