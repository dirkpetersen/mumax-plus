#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
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
    
    // Store original device to restore later
    int originalDevice;
    checkCudaError(cudaGetDevice(&originalDevice));
    
    for (int i = 0; i < deviceCount; i++) {
        try {
            cudaDeviceProp prop;
            checkCudaError(cudaGetDeviceProperties(&prop, i));
            
            // Check if device is accessible
            checkCudaError(cudaSetDevice(i));
            
            // A basic check - we assume all accessible devices are potentially usable
            // In production, you would use NVML or similar to check actual utilization
            idleCount++;
        } catch (const std::exception& e) {
            // Device not accessible, skip it
            continue;
        }
    }
    
    // Restore original device
    checkCudaError(cudaSetDevice(originalDevice));
    
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
    
    // Store original device to restore later
    int originalDevice;
    checkCudaError(cudaGetDevice(&originalDevice));
    
    for (int i = 0; i < deviceCount; i++) {
        checkCudaError(cudaSetDevice(i));
        for (int j = 0; j < deviceCount; j++) {
            if (i != j) {
                int canAccessPeer = 0;
                cudaError_t err = cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
                if (err == cudaSuccess && canAccessPeer) {
                    // Check if peer access is already enabled
                    cudaError_t enableErr = cudaDeviceEnablePeerAccess(j, 0);
                    if (enableErr != cudaSuccess && enableErr != cudaErrorPeerAccessAlreadyEnabled) {
                        // Log warning but don't fail - peer access is optional optimization
                        std::cerr << "Warning: Failed to enable peer access from device " 
                                  << i << " to device " << j << std::endl;
                    }
                }
            }
        }
    }
    
    // Restore original device
    checkCudaError(cudaSetDevice(originalDevice));
}

} // namespace DeviceUtils