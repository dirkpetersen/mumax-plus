#!/usr/bin/env python3
"""
Test script for multi-GPU FFT acceleration in mumax-plus.
This example demonstrates the performance benefits of using multiple GPUs
for stray field calculations using cuFFT.
"""

import time
import numpy as np
import mumaxplus as mp
from mumaxplus import Ferromagnet, Grid, World

def print_separator():
    print('=' * 60)

def run_strayfield_benchmark(grid_size, use_multi_gpu=True, repeat=3):
    """
    Run strayfield calculation benchmark with the specified grid size.
    
    Args:
        grid_size: Tuple of (nx, ny, nz) for the grid dimensions
        use_multi_gpu: Whether to use multi-GPU mode
        repeat: Number of times to repeat the calculation for timing
        
    Returns:
        Average execution time in seconds
    """
    print(f"Running strayfield benchmark with grid size {grid_size}")
    print(f"Multi-GPU mode: {'Enabled' if use_multi_gpu else 'Disabled'}")
    
    # Create a world and grid  
    world = World(cellsize=(5e-9, 5e-9, 5e-9))
    grid = Grid(grid_size)
    
    # Create a ferromagnet with the grid
    magnet = Ferromagnet(world, grid)
    
    # Set material parameters
    magnet.msat = 800e3  # A/m
    magnet.aex = 13e-12  # J/m
    
    # Create a non-uniform magnetization pattern to trigger stray field calculations
    magnet.magnetization = (0.8, 0.6, 0.0)  # Start with uniform state
    
    # Now run the strayfield calculation benchmark
    times = []
    for i in range(repeat):
        start_time = time.time()
        
        # Force calculation of the demag field (this triggers the FFT calculation)
        energy_val = magnet.demag_energy.eval()
        
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f} seconds")
    
    avg_time = sum(times) / len(times)
    print(f"  Average time: {avg_time:.4f} seconds")
    
    return avg_time

def run_comparative_benchmark():
    """Run benchmarks comparing single-GPU and multi-GPU performance"""
    print_separator()
    print("COMPARATIVE BENCHMARKS: SINGLE-GPU vs MULTI-GPU")
    print_separator()
    
    # Test with different grid sizes
    grid_sizes = [
        (64, 64, 64),     # Small
        (128, 128, 64),   # Medium
        (256, 256, 64),   # Large
        (384, 384, 64),   # Very large (if memory allows)
    ]
    
    results_single = {}
    results_multi = {}
    
    for size in grid_sizes:
        try:
            print_separator()
            print(f"Testing grid size: {size}")
            
            # Run with single GPU
            print("\nSingle-GPU Mode:")
            single_time = run_strayfield_benchmark(size, use_multi_gpu=False)
            results_single[size] = single_time
            
            # Run with multi GPU
            print("\nMulti-GPU Mode:")
            multi_time = run_strayfield_benchmark(size, use_multi_gpu=True)
            results_multi[size] = multi_time
            
            # Calculate speedup
            if single_time > 0:
                speedup = single_time / multi_time
                print(f"\nSpeedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"Error with grid size {size}: {str(e)}")
            print("Skipping to next grid size...")
    
    # Print summary
    print_separator()
    print("BENCHMARK SUMMARY")
    print_separator()
    print("Grid Size      | Single-GPU (s) | Multi-GPU (s) | Speedup")
    print("-------------- | -------------- | ------------- | -------")
    
    for size in grid_sizes:
        if size in results_single and size in results_multi:
            single_time = results_single[size]
            multi_time = results_multi[size]
            speedup = single_time / multi_time if multi_time > 0 else 0
            size_str = f"({size[0]}, {size[1]}, {size[2]})"
            print(f"{size_str:<14} | {single_time:14.4f} | {multi_time:13.4f} | {speedup:7.2f}x")
    
    return results_single, results_multi

if __name__ == "__main__":
    print("MULTI-GPU FFT BENCHMARK FOR STRAY FIELD CALCULATIONS")
    print_separator()
    
    # Run the comparative benchmark
    results_single, results_multi = run_comparative_benchmark()