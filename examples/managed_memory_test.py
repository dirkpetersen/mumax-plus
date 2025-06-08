#!/usr/bin/env python3
"""
Example demonstrating the use of multi-GPU support in mumax-plus.
This example utilizes the CUDA Managed Memory allocator.
"""

import numpy as np
import mumaxplus as mp
from mumaxplus import Ferromagnet, Grid, World
import time

def print_separator():
    print('-' * 60)

# Create a simple benchmark function
def run_benchmark(grid_size, repeat=3):
    """Run a benchmark with the specified grid size."""
    print(f"Benchmarking with grid size: {grid_size}")
    
    # Create a world with the specified grid size
    world = World(cellsize=(5e-9, 5e-9, 5e-9))
    grid = Grid(grid_size)
    
    # Create a ferromagnet with the grid
    magnet = Ferromagnet(world, grid)
    
    # Set material parameters
    magnet.msat = 800e3  # A/m
    magnet.aex = 13e-12  # J/m
    magnet.alpha = 0.02
    
    # Create a magnetization pattern
    magnet.magnetization = (0.7, 0.7, 0.0)
    
    # Run the simulation
    dt = 1e-12  # 1 ps
    
    # Time the execution - measure energy calculations which trigger stray field computation
    times = []
    for i in range(repeat):
        start_time = time.time()
        # Force multiple energy calculations to exercise the memory system
        for _ in range(100):
            total_energy = magnet.total_energy()
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"  Run {i+1}: {times[-1]:.4f} seconds")
    
    avg_time = sum(times) / len(times)
    print(f"  Average time: {avg_time:.4f} seconds")
    return avg_time

def main():
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