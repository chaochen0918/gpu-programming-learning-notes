import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def vecadd_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    len_vector,
    BLOCK_SIZE:tl.constexpr
    ):
    pid = tl.program_id(axis=0)
    pid_start = pid * BLOCK_SIZE
    offsets = pid_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < len_vector

    x = tl.load(x_ptr + offsets, mask = mask)
    y = tl.load(y_ptr + offsets, mask = mask)
    z = x + y
    
    tl.store(z_ptr + offsets, z, mask = mask)

def vecadd(x: torch.Tensor, y: torch.Tensor):
    z = torch.empty_like(x)

    n_elements = len(z)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    vecadd_kernel[grid](x, y, z, n_elements, BLOCK_SIZE)

    return z

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: vecadd(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == '__main__':
    torch.manual_seed(12)
    size = 967389
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)

    z_torch = x + y
    z_triton = vecadd(x, y)
    print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(z_torch - z_triton))}')
    
    benchmark.run(print_data=True, show_plots=True, save_path = './')


