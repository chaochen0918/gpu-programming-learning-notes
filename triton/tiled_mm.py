import triton
import triton.language as tl
import torch

@triton.jit
def tiled_matmul_kernel(
    A_ptr, 
    B_ptr, 
    C_ptr,
    M, N, K,
    stride_a_m, stride_a_k, 
    stride_b_n, stride_b_k,
    stride_c_m, stride_c_n,
    BLOCK_SIZE_M:tl.constexpr, 
    BLOCL_SIZE_N:tl.constexpr,
    BLOCK_SIZE_K:tl.constexpr
    ):
    # step 1: init pid in two dim, m and n
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # step 2: locate A tile and B tile
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCL_SIZE_N + tl.arange(0, BLOCL_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = A_ptr + (offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_b_k + offs_n[None, :] * stride_b_n)

        # Boundary mask
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # load a_tile and b_tile from HBM to SRAM
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0)

        # do matmul
        acc += tl.dot(a_tile, b_tile)
    
    # locate C tile
    c_ptrs = C_ptr + (offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)

def tile_matmul(
    A: torch.Tensor, 
    B: torch.Tensor, 
    BLOCK_SIZE_M=64,
    BLOCK_SIZE_N=64,
    BLOCK_SIZE_K=64
    ) -> torch.Tensor:
    assert A.ndim == 2 and B.ndim == 2 
    assert A.shape[1] == B.shape[0]

    M, K = A.shape
    _, N = B.shape
    device = A.device

    C = torch.empty((M,N), device = device, dtype = torch.float16)

    program_num_m = triton.cdiv(M, BLOCK_SIZE_M)
    program_num_n = triton.cdiv(N, BLOCK_SIZE_N)
    
    tiled_matmul_kernel[(program_num_m, program_num_n)](
        A, B, C, M, N, K, 
        A.stride(0), A.stride(1), B.stride(1), B.stride(0), C.stride(0), C.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )

    return C

if __name__ == '__main__':
    torch.manual_seed(0)
    A = torch.randn((5000, 1000), device='cuda')
    B = torch.randn((1000, 4000), device='cuda')

    C_triton = tile_matmul(A, B)
    C_pytorch = (A @ B).to(torch.float16)

    print("Max abs error:", (C_triton - C_pytorch).abs().max().item())










    

