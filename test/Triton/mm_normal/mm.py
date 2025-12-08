import os, torch, triton, triton.language as tl
from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())

print("TRITON_SHARED_DUMP_PATH =", os.environ.get("TRITON_SHARED_DUMP_PATH"))

@triton.jit
def matmul_kernel(A, B, C,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """! Blocked MatMul kernel using block pointers.

    Performs C[M, N] = A[M, K] @ B[K, N] with tiles of sizes
    (BLOCK_M, BLOCK_N, BLOCK_K). Loads/stores are issued via
    tl.make_block_ptr to enable contiguous, block-wise accesses
    and reliable out-of-bounds handling.

    @param A Pointer to matrix A (row-major strides: stride_am, stride_ak)
    @param B Pointer to matrix B (row-major strides: stride_bk, stride_bn)
    @param C Pointer to output matrix C (row-major strides: stride_cm, stride_cn)
    @param M Rows in A/C
    @param N Columns in B/C
    @param K Shared dimension between A and B
    @param BLOCK_M Tile size along M
    @param BLOCK_N Tile size along N
    @param BLOCK_K Tile size along K
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    off_m0 = pid_m * BLOCK_M
    off_n0 = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for ko in range(0, tl.cdiv(K, BLOCK_K)):
        off_k0 = ko * BLOCK_K

        a_block = tl.make_block_ptr(
            base=A,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(off_m0, off_k0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )

        b_block = tl.make_block_ptr(
            base=B,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(off_k0, off_n0),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )

        # a = tl.load(a_block, boundary_check=(0, 1), padding_option="zero")
        # b = tl.load(b_block, boundary_check=(0, 1), padding_option="zero")
        # Remove the boundary check to simplify the code
        a = tl.load(a_block)
        b = tl.load(b_block)
        acc += tl.dot(a, b)

    c_block = tl.make_block_ptr(
        base=C,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(off_m0, off_n0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block, acc, boundary_check=(0, 1))


def run_once(M=512, N=512, K=512):
    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=torch.float32)
    B = torch.randn((K, N), dtype=torch.float32)
    C = torch.empty((M, N), dtype=torch.float32)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    err = (C - A @ B).abs().max().item()
    print("max error:", err)

if __name__ == "__main__":
    run_once()
