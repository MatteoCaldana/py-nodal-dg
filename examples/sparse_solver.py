import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

import numpy as np

import time
import scipy.io
import scipy.sparse

jax.config.update("jax_enable_x64", True)


def prepare(sparse_csr):
    # Calculate row lengths and find the maximum bandwidth
    row_lengths = sparse_csr.indptr[1:] - sparse_csr.indptr[:-1]
    max_bw = int(np.max(row_lengths))
    num_rows = sparse_csr.shape[0]
    num_cols = sparse_csr.shape[1]

    # Initialize padded arrays
    # data: (rows x max_bw) zero padded for neutral addition
    # col_idxs: (rows x max_bw) padded with num_cols (index out of bounds)
    padded_data = np.zeros((num_rows, max_bw), dtype=sparse_csr.dtype)
    padded_col_idxs = np.full((num_rows, max_bw), num_cols, dtype=np.int32)
    diag_vals = sparse_csr.diagonal()

    # Fill the arrays
    for i in range(num_rows):
        start = sparse_csr.indptr[i]
        end = sparse_csr.indptr[i + 1]
        length = end - start

        if length > 0:
            padded_data[i, :length] = sparse_csr.data[start:end]
            padded_col_idxs[i, :length] = sparse_csr.indices[start:end]

    # Convert to JAX device arrays for GPU/TPU acceleration
    return (
        jnp.array(padded_data),
        jnp.array(padded_col_idxs),
        jnp.array(diag_vals),
        max_bw,
    )


@jax.jit
def matvec_v0(data, cols, x):
    x_padded = jnp.concatenate([x, jnp.zeros((1,), dtype=x.dtype)])

    def row_dot(row_data, row_cols):
        return jnp.sum(row_data * x_padded[row_cols])

    return jax.vmap(row_dot)(data, cols)

@jax.jit
def matvec_v1(data, cols, x):
    x_padded = jnp.concatenate([x, jnp.zeros((1,), dtype=x.dtype)])
    return jax.vmap(jnp.dot)(data, x_padded[cols])

@jax.jit
def matvec_v2(data, cols, x):
    x_padded = jnp.concatenate([x, jnp.zeros((1,), dtype=x.dtype)])
    return jnp.sum(data * x_padded[cols], axis=1)

@jax.jit
def matvec_v3(data, cols, x):
    x_padded = jnp.concatenate([x, jnp.zeros((1,), dtype=x.dtype)])
    return jnp.einsum('ij,ij->i', data, x_padded[cols])

@jax.jit
def matvec_classic(mat, x):
    return mat @ x

def run_benchmark(name, fn, *args, iterations=100):
    # Warmup: Force JIT compilation before timing
    _ = fn(*args).block_until_ready()

    tic = time.perf_counter()
    for _ in range(iterations):
        res = fn(*args).block_until_ready()
    toc = time.perf_counter()

    avg_ms = ((toc - tic) / iterations) * 1000
    print(f"{name:15} | Avg Time: {avg_ms:10.4f} ms")
    return res


def run_matvec(mat):
    x_np = np.random.randn(mat.shape[1]).astype(np.float64)
    x_jax = jnp.array(x_np)

    d_jax, c_jax, diag_jax, max_bw = prepare(mat)

    bcoo_mat = jsparse.BCOO.from_scipy_sparse(mat, index_dtype=jnp.int32).astype(jnp.float64)
    bcsr_mat = jsparse.BCSR.from_scipy_sparse(mat, index_dtype=jnp.int32)

    print(f"Matrix Rows: {mat.shape[0]} | Max Bandwidth: {max_bw}")
    print("-" * 50)

    # scipy bench
    tic = time.perf_counter()
    for _ in range(100):
        res_sp = mat.dot(x_np)
    toc = time.perf_counter()
    print(f"{'SciPy CSR':15} | Avg Time: {((toc-tic)/100)*1000:10.4f} ms")

    res_v1 = run_benchmark("JAX Matvec V0", matvec_v0, d_jax, c_jax, x_jax)
    np.testing.assert_allclose(res_sp, np.array(res_v1), atol=1e-10)
    res_v1 = run_benchmark("JAX Matvec V1", matvec_v1, d_jax, c_jax, x_jax)
    np.testing.assert_allclose(res_sp, np.array(res_v1), atol=1e-10)
    res_v1 = run_benchmark("JAX Matvec V2", matvec_v2, d_jax, c_jax, x_jax)
    np.testing.assert_allclose(res_sp, np.array(res_v1), atol=1e-10)
    res_v1 = run_benchmark("JAX Matvec V3", matvec_v3, d_jax, c_jax, x_jax)
    np.testing.assert_allclose(res_sp, np.array(res_v1), atol=1e-10)
    res_bcoo = run_benchmark("JAX BCOO", matvec_classic, bcoo_mat, x_jax)
    np.testing.assert_allclose(res_sp, np.array(res_bcoo), atol=1e-10)
    res_bcoo = run_benchmark("JAX BCSR", matvec_classic, bcsr_mat, x_jax)
    np.testing.assert_allclose(res_sp, np.array(res_bcoo), atol=1e-10)

    print("-" * 50)
    print("Numerical validation passed!")
    print("-" * 50)
    print("-" * 50)


@jax.jit
def jax_sp_fw_sub(lower_vals, row_cols, diag_vals, b):
    n = b.shape[0]
    # x has n+1 elements. The index n is the "sink" for padding (always 0)
    initial_x = jnp.zeros(n + 1)

    def row_body(i, current_x):
        vals = lower_vals[i]
        cols = row_cols[i]
        row_sum = jnp.dot(vals, current_x[cols])
        xi = (b[i] - row_sum) / diag_vals[i]
        return current_x.at[i].set(xi)

    final_x = jax.lax.fori_loop(0, n, row_body, initial_x)
    return final_x[:n]


if __name__ == "__main__":
    DIR = "/home/matteo/Documents/nodal-dg/Codes1.1/"
    raw_data = scipy.io.loadmat(DIR + "INS2D_2.mat")
    mat = raw_data["PRsystemC"].T.tocsr().copy()

    matmat = mat @ mat.T
    d_jax, c_jax, diag_jax, max_bw = prepare(mat)
    print("size:", mat.shape, "bw:", max_bw)

    # TODO: - test matvec with bcoo vs mine on cpu and GPU
    #       - test [p]cg
    #       - add preconditioner


    run_matvec(mat)

    rhs = np.ones(mat.shape[0])

    tic = time.perf_counter()
    for _ in range(100):
        xsp = scipy.sparse.linalg.spsolve_triangular(mat, rhs, lower=True)
    toc = time.perf_counter()
    print(f"{'SciPy tri solve':15} | Avg Time: {((toc-tic)/100)*1000:10.4f} ms")

    d_jax, c_jax, diag_jax, max_bw = prepare(mat)
    xjx = jax_sp_fw_sub(d_jax, c_jax, diag_jax, jnp.asarray(rhs)).block_until_ready()

    iterations = 100
    tic = time.perf_counter()
    for _ in range(iterations):
        res = jax_sp_fw_sub(
            d_jax, c_jax, diag_jax, jnp.asarray(rhs)
        ).block_until_ready()
    toc = time.perf_counter()

    avg_ms = ((toc - tic) / iterations) * 1000
    print(f"{'jax tri solve':15} | Avg Time: {avg_ms:10.4f} ms")
    print("Numerical validation jax_sp_fw_sub: ", np.max(np.abs(xsp - xjx)) < 1e-10)



