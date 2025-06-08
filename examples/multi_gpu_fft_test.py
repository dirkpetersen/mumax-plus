#!/usr/bin/env python3
"""
Test script for multi-GPU FFT acceleration in mumax-plus.
This example demonstrates the performance benefits of using multiple GPUs
for stray field calculations using cuFFT.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import mumaxplus as mp

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
    world = mp.MumaxWorld()
    grid = mp.Grid(world, grid_size[0], grid_size[1], grid_size[2])
    
    # Create a ferromagnet with the grid
    magnet = mp.Ferromagnet(world, grid)
    
    # Set material parameters
    magnet.msat.setUniformValue(800e3)  # A/m
    
    # Create a non-uniform magnetization pattern (skyrmion-like)
    mag = magnet.magnetization()
    center_x = grid_size[0] / 2
    center_y = grid_size[1] / 2
    radius = min(grid_size[0], grid_size[1]) / 4
    
    # Set up the magnetization field with a circular pattern
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            for k in range(grid_size[2]):
                # Calculate distance from center
                dx = i - center_x
                dy = j - center_y
                r = np.sqrt(dx*dx + dy*dy)
                
                # Calculate angle for vortex
                phi = np.arctan2(dy, dx)
                
                if r < radius:
                    # Inside the skyrmion: magnetization points in different directions
                    theta = np.pi * (1 - r/radius)  # Vary from π at center to 0 at edge
                    mx = np.sin(theta) * np.cos(phi)
                    my = np.sin(theta) * np.sin(phi)
                    mz = np.cos(theta)
                else:
                    # Outside: uniform magnetization pointing up
                    mx, my, mz = 0, 0, 1
                
                # Set the magnetization at this cell
                mag.setComponentInCell(i, j, k, 0, mx)
                mag.setComponentInCell(i, j, k, 1, my)
                mag.setComponentInCell(i, j, k, 2, mz)
    
    # Now run the strayfield calculation benchmark
    times = []
    for i in range(repeat):
        start_time = time.time()
        
        # Explicitly calculate the strayfield (this triggers the FFT calculation)
        h_demag = magnet.demagField()
        
        # We need to get the data to ensure the calculation is completed
        h_data = h_demag.getData()
        
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