#!/usr/bin/env python3
"""
Example demonstrating the use of multi-GPU support in mumax-plus.
This example utilizes the CUDA Managed Memory allocator.
"""

import numpy as np
import mumaxplus as mp
import time

def print_separator():
    print('-' * 60)

# Create a simple benchmark function
def run_benchmark(grid_size, repeat=3):
    """Run a benchmark with the specified grid size."""
    print(f"Benchmarking with grid size: {grid_size}")
    
    # Create a world with the specified grid size
    world = mp.MumaxWorld()
    grid = mp.Grid(world, grid_size[0], grid_size[1], grid_size[2])
    
    # Create a ferromagnet with the grid
    magnet = mp.Ferromagnet(world, grid)
    
    # Set material parameters
    magnet.msat.setUniformValue(800e3)  # A/m
    magnet.aex.setUniformValue(13e-12)  # J/m
    
    # Create a Ferromagnet with standard parameters
    magnet.magnetization().randomize()
    
    # Create a time solver
    solver = mp.TimeSolver(world)
    
    # Run the simulation
    dt = 1e-12  # 1 ps
    
    # Time the execution
    times = []
    for i in range(repeat):
        start_time = time.time()
        solver.step(magnet, 1000 * dt)  # Run for 1000 steps
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"  Run {i+1}: {times[-1]:.4f} seconds")
    
    avg_time = sum(times) / len(times)
    print(f"  Average time: {avg_time:.4f} seconds")
    return avg_time

def main():
    # Print information about GPUs
    world = mp.MumaxWorld()
    
    print("Starting multi-GPU benchmark with managed memory:")
    print_separator()
    
    # Try different grid sizes
    grid_sizes = [
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 32)
    ]
    
    results = {}
    for size in grid_sizes:
        print_separator()
        avg_time = run_benchmark(size)
        results[size] = avg_time
    
    # Print summary
    print_separator()
    print("Benchmark Summary:")
    for size, avg_time in results.items():
        print(f"Grid Size {size}: {avg_time:.4f} seconds")

if __name__ == "__main__":
    main()