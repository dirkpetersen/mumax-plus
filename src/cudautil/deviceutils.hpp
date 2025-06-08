#pragma once

#include <cuda_runtime.h>
#include <vector>
#include "cudaerror.hpp"

/**
 * Utility functions for device management
 */
namespace DeviceUtils {

/**
 * Get the number of available CUDA devices
 * 
 * @return The number of CUDA devices available
 */
inline int getDeviceCount() {
    int count = 0;
    checkCudaError(cudaGetDeviceCount(&count));
    return count;
}

/**
 * Check if any GPU is idle (0% utilization)
 * 
 * @return Number of idle GPUs available
 */
inline int getIdleGPUCount() {
    int deviceCount = getDeviceCount();
    int idleCount = 0;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        checkCudaError(cudaGetDeviceProperties(&prop, i));
        
        // A very basic idle check - in a production system you would
        // want to use the CUDA management API or NVML to get actual utilization
        // For now, we just count all detected devices as potentially idle
        idleCount++;
    }
    
    return idleCount;
}

/**
 * Check if multi-GPU operations are supported
 * 
 * @return True if multiple GPUs are available and peer access is possible
 */
inline bool isMultiGPUSupported() {
    int deviceCount = getDeviceCount();
    
    // Need at least 2 GPUs for multi-GPU operations
    if (deviceCount < 2) {
        return false;
    }
    
    // Check peer access between first two devices
    int canAccessPeer = 0;
    checkCudaError(cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1));
    
    return canAccessPeer > 0;
}

/**
 * Enable peer access between all available devices
 */
inline void enablePeerAccess() {
    int deviceCount = getDeviceCount();
    
    if (deviceCount <= 1) {
        return; // Nothing to do
    }
    
    for (int i = 0; i < deviceCount; i++) {
        checkCudaError(cudaSetDevice(i));
        for (int j = 0; j < deviceCount; j++) {
            if (i != j) {
                int canAccessPeer = 0;
                cudaError_t err = cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
                if (err == cudaSuccess && canAccessPeer) {
                    cudaDeviceEnablePeerAccess(j, 0);
                }
            }
        }
    }
}

} // namespace DeviceUtils