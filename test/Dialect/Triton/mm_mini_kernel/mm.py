import os, torch, triton, triton.language as tl
from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())

print("TRITON_SHARED_DUMP_PATH =", os.environ.get("TRITON_SHARED_DUMP_PATH"))

@triton.jit(noinline=True)
def _matmul_compute_tile(A, B, C,
                         M, N, K,
                         stride_am, stride_ak,
                         stride_bk, stride_bn,
                         stride_cm, stride_cn,
                         off_m0, off_n0,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """! Compute one C tile at offsets (off_m0, off_n0).

    Performs the K-loop for a single (BLOCK_M x BLOCK_N) output tile using
    block-pointer loads/stores. No program-id mapping is performed here.

    @param off_m0 Row offset of the tile in C/A
    @param off_n0 Column offset of the tile in C/B
    """

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
    tl.store(c_block, acc)


@triton.jit
def matmul_kernel(A, B, C,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_M: tl.constexpr):
    """! Wrapper MatMul kernel computing tile mapping, then invoking tile compute.

    Computes program-id mapping for (pid_m, pid_n) using GROUP_M grouping, then
    calls the inner tile-compute routine with explicit tile offsets.
    """

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_id  = pid // (GROUP_M * num_pid_n)
    first_m   = group_id * GROUP_M
    pid_m     = first_m + (pid % GROUP_M)
    pid_n     = (pid // GROUP_M) % num_pid_n

    off_m0 = pid_m * BLOCK_M
    off_n0 = pid_n * BLOCK_N

    _matmul_compute_tile(
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        off_m0, off_n0,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )


def run_once(M=512, N=512, K=512):
    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=torch.float32)
    B = torch.randn((K, N), dtype=torch.float32)
    C = torch.empty((M, N), dtype=torch.float32)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    GROUP_M = 8
    # Compile and run only the inner tile kernel to dump its lowering without inlining.
    grid = (1,)
    off_m0, off_n0 = 0, 0
    _matmul_compute_tile[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        off_m0, off_n0,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    err = (C - A @ B).abs().max().item()
    print("max error:", err)

if __name__ == "__main__":
    run_once()
