import torch
from torch.types import Device
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
        input_ptr, 
        output_ptr, 
        input_stride, 
        output_stride, 
        n_rows, 
        n_cols,
        BLOCK_SIZE:tl.constexpr, 
        num_stages:tl.constexpr
    ):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        input_start_ptr = input_ptr + row_idx * input_stride
        offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = input_start_ptr + offsets 
        mask = offsets < n_cols

        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row = row - tl.max(row, axis=0)
        numerator = tl.exp(row)
        denominator = tl.sum(numerator, axis=0)
        output_row = numerator / denominator

        output_start_ptr = output_ptr + row_idx * output_stride
        output_ptrs = output_start_ptr + offsets
        tl.store(output_ptrs, output_row, mask=mask)

# for computing how many programs (block) to launch
DEVICE = triton.runtime.driver.active.get_active_torch_device()
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGISTER_PER_SM = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]

def softmax(x): # (M x N)
    n_rows, n_cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8 # determine by the programmer
    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    y = torch.empty_like(x)

    kernel = softmax_kernel.warmup(x, y, x.stride(0), y.stride(0), n_rows, n_cols, 
                                BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages, grid=(1,))
    
    kernel._init_handles()
    n_regs = kernel.n_regs# the register number that compiler optimized given a kernel
    size_mem = kernel.metadata.shared# the shared memory compiler allocated given a kernel

    # assume a cuda backend
    block_per_sm = min(
        NUM_REGISTER_PER_SM // (n_regs * WARP_SIZE * num_warps), # block num from register
        SIZE_SMEM // size_mem # block num from share memory
    )
    # return output # (M x N)
    num_programs = NUM_SM * block_per_sm

    kernel[(num_programs,1,1)](x, y, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)

    return y

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch'],  # possible values for `line_arg``
        line_names=["Triton", "Torch"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device=DEVICE)

    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis = 1)

    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

    benchmark.run(show_plots=True, print_data=True, save_path='./')



