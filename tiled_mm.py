import triton
import triton.language as tl
import torch


@triton.jit
def tiled_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Which output tile does this program instance own?
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Row/col offsets for this tile (vectors of indices)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator: a BLOCK_M x BLOCK_N block of fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K in BLOCK_K-wide strips
    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Build 2-D pointer arrays for the A and B tiles
        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
        # B tile: [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

        # Boundary masks — handles M/N/K not divisible by block sizes
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        b_mask = (rk[:, None] < K) & (rn[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Single tensor-core matmul over the tile — this is the key op
        acc += tl.dot(a, b)

    # Write the output tile back to C
    c_ptrs = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def tiled_matmul(A: torch.Tensor, B: torch.Tensor,
                 BLOCK_M=64, BLOCK_N=64, BLOCK_K=32) -> torch.Tensor:
    assert A.ndim == 2 and B.ndim == 2
    assert A.shape[1] == B.shape[0]
    assert A.is_cuda and B.is_cuda

    M, K = A.shape
    _, N  = B.shape

    A = A.to(torch.float16)
    B = B.to(torch.float16)
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    tiled_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


# --- Correctness check ---
if __name__ == "__main__":
    torch.manual_seed(0)
    A = torch.randn(256, 512, device="cuda")
    B = torch.randn(512, 128, device="cuda")

    C_triton = tiled_matmul(A, B)
    C_ref    = (A @ B).to(torch.float16)

    print("Max abs error:", (C_triton - C_ref).abs().max().item())