import torch
import torch.nn as nn
import random
import time
from TemporalShift.TemporalShift import TemporalShift


class IdentityModule(nn.Module):
    def forward(self, x):
        return x

def benchmark_memory(x, n_segment, fold_div, inplace=False):
    torch.cuda.empty_cache()  # Clear cache
    torch.cuda.reset_peak_memory_stats()  # Reset memory tracker


    if inplace:
        _ = TemporalShift.shift(x, n_segment, fold_div, inplace=True)  # In-place modifies `x`
    else:
        _ = TemporalShift.shift(x, n_segment, fold_div, inplace=False)  # Out-of-place creates new tensor

    peak_mem = torch.cuda.max_memory_allocated(device='cuda')  # Peak memory in bytes
    return peak_mem / (1024 ** 2)  # Convert to MB

def single_test_temporal_shift_operation(x, n_segment, fold_div, vect=False):

    x1 = x.clone().cuda()
    x1.requires_grad_()

    IM = IdentityModule()

    x_shifted_inplace = TemporalShift.shift(x1, n_segment, fold_div, inplace=True, vect=vect)

    # Compute dummy loss
    loss = torch.sum(x_shifted_inplace ** 2)
    loss.backward()
    grad_inplace = x1.grad.clone()

    x2 = x.clone().cuda()
    x2.requires_grad_()

    x_shifted_original = TemporalShift.shift(x2, n_segment, fold_div, inplace=False, vect=vect)
    loss = torch.sum(x_shifted_original ** 2)
    loss.backward()
    grad_original = x2.grad.clone()

    if not torch.allclose(x_shifted_inplace, x_shifted_original, atol=1e-6):
        print("\nError in forward pass - tensors don't match!")
        print(f"Tensor shapes - Inplace: {x_shifted_inplace.shape}, Original: {x_shifted_original.shape}")

        # Print channel-wise comparison
        for c in range(x_shifted_inplace.size(1)):  # Loop through channels
            channel_inplace = x_shifted_inplace[:, c, :, :]
            channel_original = x_shifted_original[:, c, :, :]

            if not torch.allclose(channel_inplace, channel_original, atol=1e-6):
                print(f"\nChannel {c} mismatch:")
                print("Inplace values (sample of first batch element):")
                print(channel_inplace[0])  # First element in batch
                print("\nOriginal values (sample of first batch element):")
                print(channel_original[0])

                # Calculate and print difference statistics
                diff = channel_inplace - channel_original
                print(f"\nMax difference: {diff.abs().max().item()}")
                print(f"Mean difference: {diff.abs().mean().item()}")
                print(f"Number of differing elements: {(diff.abs() > 1e-6).sum().item()}")

        raise AssertionError("Error: forward pass tensors don't match")
    else:
        print("Forward pass check passed successfully")

    assert torch.allclose(x_shifted_inplace, x_shifted_original, atol=1e-6), "Error: forward"
    assert torch.allclose(grad_inplace, grad_original, atol=1e-6), "Error: backward"


def test_control():

    BATCH_SIZE = 2
    N_SEGMENT = 2
    CHANNELS = 64
    HEIGHT = 18
    WIDTH = 20
    FOLD_DIV = 4

    # Create the tensor with shape (BATCH_SIZE * N_SEGMENT, CHANNELS, HEIGHT, WIDTH)

    x = torch.zeros(BATCH_SIZE * N_SEGMENT, CHANNELS, HEIGHT, WIDTH, dtype=torch.float32)
    # Fill each channel with its channel index value
    for c in range(CHANNELS):
        x[:, c, :, :] = c


    single_test_temporal_shift_operation(x, N_SEGMENT, FOLD_DIV, vect=False)
    single_test_temporal_shift_operation(x, N_SEGMENT, FOLD_DIV, vect=True)



def test_correctness_temporal_shift_operation(vect=False):

    current_seed = int(time.time() * 1e9) % (2 ** 32 - 1)
    torch.manual_seed(current_seed)
    random.seed(current_seed)

    # Define min/max thresholds for each parameter
    param_ranges = {
        'BATCH_SIZE': (1, 20),
        'N_SEGMENT': (2, 32),
        'CHANNELS': (8, 128),
        'SPATIAL_DIM': (16, 64),  # Same for HEIGHT and WIDTH
        'FOLD_DIV': (2, 8)

    }

    # Example loop for multiple runs
    NUM_RUNS = 10
    for run in range(NUM_RUNS):
        print(f"\nRun {run + 1}/{NUM_RUNS}")
        # Your benchmark code here
        # Can access parameters via params dictionary
        # Generate random parameters within thresholds
        params = {
            'BATCH_SIZE': random.randint(*param_ranges['BATCH_SIZE']),
            'N_SEGMENT': random.randint(*param_ranges['N_SEGMENT']),
            'CHANNELS': random.randint(*param_ranges['CHANNELS']),
            'HEIGHT': random.randint(*param_ranges['SPATIAL_DIM']),
            'WIDTH': random.randint(*param_ranges['SPATIAL_DIM']),
            'FOLD_DIV': random.randint(*param_ranges['FOLD_DIV']),

        }

        if vect:
            # Ensure height and width are multiples of 4
            params['HEIGHT'] = (params['HEIGHT'] // 2) * 2
            params['WIDTH'] = (params['WIDTH'] // 2) * 2

        # Create input tensor with random parameters
        x = torch.randn(
            params['BATCH_SIZE'] * params['N_SEGMENT'],
            params['CHANNELS'],
            params['HEIGHT'],
            params['WIDTH']
        )
        print(f"Processing tensor of shape {x.shape}...")
        # Example operation
        single_test_temporal_shift_operation(x, params['N_SEGMENT'], params['FOLD_DIV'], vect=vect)
        print(f"run sucsessfull...")

    print(f"All ({NUM_RUNS}) correctness tests pass sucsessfully.")



def benchmark_temporal_shift_operation(x, n_segment, fold_div, inplace=True, vect=False, num_runs=10):
    """
    Benchmark the forward and backward pass of TemporalShift operation.

    Args:
        x (torch.Tensor): Input tensor
        n_segment (int): Number of temporal segments
        fold_div (int): Division factor for folding
        inplace (bool): Whether to perform operation inplace
        num_runs (int): Number of benchmark runs to average

    Returns:
        tuple: (avg_forward_time, avg_backward_time) in seconds
    """
    #compiling none inplace mode (improve perfomance)
    # Get GPU compute capability
    cuda_capability = torch.cuda.get_device_capability()
    major_version = cuda_capability[0]  # First number is major version

    tsm = TemporalShift(IdentityModule(), n_segment, fold_div, inplace=False, vect=False)
    # Conditionally compile based on CUDA capability
    if major_version >= 7:  # 7.0+ supports Triton
        print(f"GPU detected (CUDA {cuda_capability[0]}.{cuda_capability[1]}), using torch.compile()")
        tsm_compiled = torch.compile(tsm, fullgraph=True)
    else:
        print(f"GPU too old (CUDA {cuda_capability[0]}.{cuda_capability[1]}), skipping compilation")
        tsm_compiled = tsm  # Fallback to uncompiled model


    def single_run(x, vect=False):
        """Perform a single forward-backward pass and measure timings."""
        # Forward pass
        if not inplace:
            torch.cuda.synchronize()
            start_forward = time.time()
            x_shifted = tsm_compiled(x)
            torch.cuda.synchronize()
            forward_time = time.time() - start_forward
        else:
            torch.cuda.synchronize()
            start_forward = time.time()
            x_shifted = TemporalShift.shift(x, n_segment, fold_div, inplace, vect)
            torch.cuda.synchronize()
            forward_time = time.time() - start_forward


        # Compute dummy loss
        loss = torch.mean(x_shifted ** 2)

        # Backward pass
        torch.cuda.synchronize()
        start_backward = time.time()
        loss.backward()
        torch.cuda.synchronize()
        backward_time = time.time() - start_backward

        torch.cuda.empty_cache()  # Free unused GPU memory

        return forward_time, backward_time

    # Warm-up run (not timed)
    _ = single_run(x)

    # Benchmark runs
    total_forward, total_backward = 0, 0
    for _ in range(num_runs):
        f_time, b_time = single_run(x, vect=vect)
        total_forward += f_time
        total_backward += b_time

    return total_forward / num_runs, total_backward / num_runs


def print_benchmark_results(mode, forward_time, backward_time):
    """Print formatted benchmark results."""
    print(f"\nBenchmark results for {mode} mode:")
    print(f"- Average forward time:  {forward_time:.6f} seconds")
    print(f"- Average backward time: {backward_time:.6f} seconds")



if __name__ == '__main__':

    #use for debuging
    #test_control()


    # Correctness test
    print("Runing correctness tests ...")
    test_correctness_temporal_shift_operation(vect=False)
    test_correctness_temporal_shift_operation(vect=True)

    # Benchmark test
    print("\n-----------------------------------\n")
    print("Runing benchmark tests ...")

    # Configuration Benchmark
    torch.manual_seed(42)
    BATCH_SIZE = 10
    N_SEGMENT = 16
    CHANNELS = 256
    HEIGHT = WIDTH = 32
    FOLD_DIV = 4
    NUM_RUNS = 100

    print("Initializing benchmark...")
    print(f"Configuration:")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Temporal segments: {N_SEGMENT}")
    print(f"- Channels: {CHANNELS}")
    print(f"- Spatial dimensions: {HEIGHT}x{WIDTH}")
    print(f"- Fold division: {FOLD_DIV}")
    print(f"- Benchmark runs: {NUM_RUNS}")

    # Create input tensor
    x = torch.randn(BATCH_SIZE * N_SEGMENT, CHANNELS, HEIGHT, WIDTH)
    x = x.clone().cuda()
    x.requires_grad_()

    print("\nRunning benchmarks...")
    print("\n\n\nMemory test ...")
    peak_mem_outplace = benchmark_memory(x, N_SEGMENT, FOLD_DIV, inplace=False)
    peak_mem_inplace = benchmark_memory(x, N_SEGMENT, FOLD_DIV, inplace=True)

    # Benchmark
    print("In-place peak memory (MB):", peak_mem_inplace)
    print("Out-of-place peak memory (MB):", peak_mem_outplace)
    print(f"In-place consumes {peak_mem_outplace / peak_mem_inplace:.2f} times less memory than out-of-place.")

    print("\n\n\nPerfomance test ...")

    torch.cuda.empty_cache()

    x = x.clone().cuda()
    x.requires_grad_()

    # Benchmark out-of-place mode
    forward_outplace, backward_outplace = benchmark_temporal_shift_operation(
        x, N_SEGMENT, FOLD_DIV, inplace=False, num_runs=NUM_RUNS
    )
    print_benchmark_results("OUT-OF-PLACE", forward_outplace, backward_outplace)

    torch.cuda.empty_cache()

    x = x.clone().cuda()
    x.requires_grad_()

    forward_inplace_vect, backward_inplace_vect = benchmark_temporal_shift_operation(
        x, N_SEGMENT, FOLD_DIV, inplace=True, vect=True, num_runs=NUM_RUNS
    )
    print_benchmark_results("INPLACE vect=True", forward_inplace_vect, backward_inplace_vect)

    torch.cuda.empty_cache()

    x = x.clone().cuda()
    x.requires_grad_()

    # Benchmark inplace mode
    forward_inplace, backward_inplace = benchmark_temporal_shift_operation(
        x, N_SEGMENT, FOLD_DIV, inplace=True, vect=False, num_runs=NUM_RUNS
    )
    print_benchmark_results("INPLACE", forward_inplace, backward_inplace)




    # Performance comparison
    print("\nPerformance comparison Inplace vs Out-of-place:")
    print(f"Inplace is {forward_outplace / forward_inplace:.2f}x faster in forward pass")
    print(f"Inplace (vectorized) is {forward_outplace / forward_inplace_vect:.2f}x faster in forward pass")
    print(f"Inplace is {backward_outplace / backward_inplace:.2f}x faster in backward pass")
    print(f"Inplace (vectorized) is {backward_outplace / backward_inplace_vect:.2f}x faster in backward pass")

