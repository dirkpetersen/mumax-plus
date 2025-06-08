#include "deviceutils.hpp"
#include "cudaerror.hpp"
#include <iostream>

// Implementation of more complex device utility functions that aren't inline

namespace DeviceUtils {

/**
 * Print information about all available GPUs
 */
void printDeviceInfo() {
    int deviceCount = getDeviceCount();
    
    std::cout << "CUDA Device Information:" << std::endl;
    std::cout << "======================" << std::endl;
    std::cout << "Total CUDA devices found: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        checkCudaError(cudaGetDeviceProperties(&prop, i));
        
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Unified memory support: " << (prop.unifiedAddressing ? "Yes" : "No") << std::endl;
        std::cout << "  Concurrent kernel execution: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
    }
    
    if (deviceCount > 1) {
        std::cout << "\nPeer-to-Peer Access Matrix:" << std::endl;
        std::cout << "==========================" << std::endl;
        
        for (int i = 0; i < deviceCount; i++) {
            for (int j = 0; j < deviceCount; j++) {
                if (i != j) {
                    int canAccessPeer = 0;
                    cudaError_t err = cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
                    if (err == cudaSuccess) {
                        std::cout << "Device " << i << " -> Device " << j << ": " 
                                  << (canAccessPeer ? "Yes" : "No") << std::endl;
                    } else {
                        std::cout << "Device " << i << " -> Device " << j << ": Error checking" << std::endl;
                    }
                }
            }
        }
    }
    
    // Check if Unified Memory is supported
    std::cout << "\nUnified Memory Support:" << std::endl;
    std::cout << "======================" << std::endl;
    
    cudaDeviceProp prop;
    checkCudaError(cudaGetDeviceProperties(&prop, 0));
    
    if (prop.unifiedAddressing) {
        std::cout << "Unified Memory is supported on this system" << std::endl;
    } else {
        std::cout << "Unified Memory is NOT supported on this system" << std::endl;
        std::cout << "Warning: cudaMallocManaged will fall back to regular cudaMalloc behavior" << std::endl;
    }
    
    std::cout << "======================" << std::endl;
}

} // namespace DeviceUtils