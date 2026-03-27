import triton
import triton.language as tl
import torch


@triton.jit
def naive_matmul_kernel(
    A_ptr, B_ptr, C_ptr, # A_ptr is the base address — a raw memory address pointing to the first element A[0, 0] in the GPU's global memory address space.
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
):
    # Each program instance (block instance) handles one output element C[m, n]
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Guard against out-of-bounds programs
    if pid_m >= M or pid_n >= N:
        return

    # Accumulator for the dot product
    acc = tl.zeros((), dtype=tl.float32)

    for k in range(K): # do the dot product of A[pid_m, :] and B[:, pid_n]
        # tl.load(A_ptr + offset) : Each Triton program instance (a GPU thread group) is issuing a memory instruction to read from GPU global memory into a register.
        # offset = base + row * K + col * 1
        a = tl.load(A_ptr + pid_m * stride_am + k * stride_ak) # A[offset]
        b = tl.load(B_ptr + k   * stride_bk  + pid_n * stride_bn) # B[offset]
        acc += a * b

    tl.store(C_ptr + pid_m * stride_cm + pid_n * stride_cn, acc)


def naive_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.ndim == 2 and B.ndim == 2
    assert A.shape[1] == B.shape[0], "Inner dimensions must match"
    assert A.is_cuda and B.is_cuda

    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    # Output buffer (fp32 accumulation)
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # One program per output element → 2-D grid
    grid = (M, N)

    naive_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),   # stride_am, stride_ak
        B.stride(0), B.stride(1),   # stride_bk, stride_bn
        C.stride(0), C.stride(1),   # stride_cm, stride_cn
    )
    return C


# --- Quick correctness check ---
if __name__ == "__main__":
    torch.manual_seed(0)
    A = torch.randn(64, 128, device="cuda", dtype=torch.float32) # This allocates a buffer in GPU DRAM (global memory) and fills it there.
    B = torch.randn(128, 32, device="cuda", dtype=torch.float32)

    C_triton = naive_matmul(A, B)
    C_ref    = A @ B

    print("Max abs error:", (C_triton - C_ref).abs().max().item())
    # Expect something like 1e-5 or smaller