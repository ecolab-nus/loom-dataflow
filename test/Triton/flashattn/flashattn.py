import argparse
import os

import torch
import triton  # pylint: disable=import-error
import triton.language as tl  # pylint: disable=import-error
from triton.backends.triton_shared.driver import CPUDriver  # pylint: disable=import-error


triton.runtime.driver.set_active(CPUDriver())

DEFAULT_SEQ_LEN = 512
DEFAULT_HEAD_DIM = 32
DEFAULT_BLOCK_Q = 32
DEFAULT_BLOCK_KV = 32
DEFAULT_SEED = 0


@triton.jit
def flashattn_fwd(
    Q,      # (SEQ_LEN, HEAD_DIM), row-major
    K_T,    # (HEAD_DIM, SEQ_LEN), row-major (i.e., K^T contiguous)
    V,      # (SEQ_LEN, HEAD_DIM), row-major
    O,      # (SEQ_LEN, HEAD_DIM), row-major (output)
    softmax_scale: tl.constexpr,

    # explicit element-wise strides
    stride_qs: tl.constexpr, stride_qd: tl.constexpr,
    stride_kt: tl.constexpr, stride_ks: tl.constexpr,    # for K_T: (dim=head, seq)
    stride_vs: tl.constexpr, stride_vd: tl.constexpr,
    stride_os: tl.constexpr, stride_od: tl.constexpr,

    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """!FlashAttention forward kernel specialized for CPUDriver."""

    pid_q = tl.program_id(0)                       # which query block
    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_kv = tl.arange(0, BLOCK_KV)

    # Block pointers
    Q_blk = tl.make_block_ptr(
        base=Q, shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_qs, stride_qd),
        offsets=(pid_q * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, HEAD_DIM),
        order=(1, 0),
    )
    V_blk = tl.make_block_ptr(
        base=V, shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_vs, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_KV, HEAD_DIM),
        order=(1, 0),
    )
    Kt_blk = tl.make_block_ptr(
        base=K_T, shape=(HEAD_DIM, SEQ_LEN),        # K^T layout
        strides=(stride_kt, stride_ks),             # (head, seq)
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_KV),           # we slide across seq
        order=(0, 1),
    )
    O_blk = tl.make_block_ptr(
        base=O, shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_os, stride_od),
        offsets=(pid_q * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, HEAD_DIM),
        order=(1, 0),
    )

    # load Q block (stays in SRAM)
    Q_block = tl.load(Q_blk)

    # streaming softmax accumulators per row
    m_i = tl.full([BLOCK_Q], -float("inf"), tl.float32)    # running max
    l_i = tl.zeros([BLOCK_Q], tl.float32) + 1.0            # running sum
    O_acc = tl.zeros([BLOCK_Q, HEAD_DIM], tl.float32)      # output accumulator

    # iterate over K/V along sequence (BLOCK_KV at a time)
    for start in range(0, SEQ_LEN, BLOCK_KV):
        start = tl.multiple_of(start, BLOCK_KV)

        K_block = tl.load(Kt_blk)                          # (HEAD_DIM, BLOCK_KV)
        # qk = Q_block @ K_block  -> (BLOCK_Q, BLOCK_KV)
        QK = tl.dot(Q_block, K_block) * softmax_scale

        # update running max/log-sum-exp
        m_ij = tl.maximum(m_i, tl.max(QK, axis=1))
        QK = QK - m_ij[:, None]
        P = tl.exp(QK)
        l_ij = tl.sum(P, axis=1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_blk)                           # (BLOCK_KV, HEAD_DIM)
        P = P.to(tl.float16)
        O_acc = O_acc * alpha[:, None]
        O_acc = tl.dot(P, V_block, O_acc)                  # fused: O_acc += P @ V

        m_i = m_ij

        # advance K and V to next seq tile
        V_blk = tl.advance(V_blk, (BLOCK_KV, 0))
        Kt_blk = tl.advance(Kt_blk, (0, BLOCK_KV))

    # normalize and store
    O_acc = O_acc / l_i[:, None]
    tl.store(O_blk, O_acc.to(O.type.element_ty))


def run_flashattn(seq_len: int, head_dim: int, block_q: int, block_kv: int, seed: int) -> None:
    """!Run the FlashAttention kernel for a single tensor/block configuration."""

    if seq_len % block_q != 0:
        raise ValueError(f"SEQ_LEN ({seq_len}) must be divisible by BLOCK_Q ({block_q}) for boundary-free tiles.")
    if seq_len % block_kv != 0:
        raise ValueError(f"SEQ_LEN ({seq_len}) must be divisible by BLOCK_KV ({block_kv}) for boundary-free tiles.")

    print("TRITON_SHARED_DUMP_PATH =", os.environ.get("TRITON_SHARED_DUMP_PATH"))
    torch.manual_seed(seed)

    # Row-major 2D tensors for simplicity
    Q = torch.randn((seq_len, head_dim), dtype=torch.float16)
    K = torch.randn((seq_len, head_dim), dtype=torch.float16)
    V = torch.randn((seq_len, head_dim), dtype=torch.float16)
    O = torch.empty_like(Q)

    # We'll pass K^T as a contiguous (HEAD_DIM, SEQ_LEN) row-major tensor
    K_T = K.transpose(0, 1).contiguous()

    # Explicit element-wise strides
    stride_qs, stride_qd = Q.stride()          # (HEAD_DIM, 1) for row-major (seq, dim)
    stride_vs, stride_vd = V.stride()
    stride_os, stride_od = O.stride()
    # For K_T (HEAD_DIM, SEQ_LEN), row-major => strides ~ (SEQ_LEN, 1)
    stride_kt, stride_ks = K_T.stride()

    softmax_scale = 1.0 / (head_dim ** 0.5)

    # 1D grid over query tiles
    grid = (triton.cdiv(seq_len, block_q),)

    flashattn_fwd[grid](
        Q, K_T, V, O, softmax_scale,
        stride_qs, stride_qd,
        stride_kt, stride_ks,
        stride_vs, stride_vd,
        stride_os, stride_od,
        SEQ_LEN=seq_len,
        HEAD_DIM=head_dim,
        BLOCK_Q=block_q,
        BLOCK_KV=block_kv,
        num_warps=4, num_stages=2,
    )

    # quick correctness check vs reference
    with torch.no_grad():
        P = (Q.float() @ K.float().transpose(0, 1)) * softmax_scale
        P = torch.softmax(P, dim=-1).half()
        O_ref = P @ V
        max_err = (O_ref - O).abs().max().item()
        print(f"max |O_ref - O| = {max_err:.3e}")


def parse_args() -> argparse.Namespace:
    """!Parse CLI arguments for configuring flash attention runs."""

    parser = argparse.ArgumentParser(description="Run the FlashAttention example with configurable tiling.")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="Sequence length (# of query tokens).")
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM, help="Head dimension size.")
    parser.add_argument("--block-q", type=int, default=DEFAULT_BLOCK_Q, help="Tile size along the query dimension.")
    parser.add_argument("--block-kv", type=int, default=DEFAULT_BLOCK_KV, help="Tile size used for streaming K/V.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    """!CLI entrypoint for the FlashAttention Triton example."""

    args = parse_args()
    run_flashattn(
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        block_q=args.block_q,
        block_kv=args.block_kv,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
