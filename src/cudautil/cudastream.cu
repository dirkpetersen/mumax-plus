#include "cudastream.hpp"
#include "cudaerror.hpp"

cudaStream_t stream0;

cudaStream_t getCudaStream() {
  if (!stream0)
    cudaStreamCreate(&stream0);
  return stream0;
}

// Multi-GPU utility functions
void initializeMultiGpu() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount > 1) {
        // Enable peer access between all GPUs
        for (int i = 0; i < deviceCount; i++) {
            cudaSetDevice(i);
            for (int j = 0; j < deviceCount; j++) {
                if (i != j) {
                    int canAccess;
                    cudaDeviceCanAccessPeer(&canAccess, i, j);
                    if (canAccess) {
                        cudaDeviceEnablePeerAccess(j, 0);
                    }
                }
            }
        }
        cudaSetDevice(0); // Reset to default device
    }
}

int getNumDevices() {
    int count;
    cudaGetDeviceCount(&count);
    return count;
}

void setCurrentDevice(int deviceId) {
    cudaSetDevice(deviceId);
}

void synchronizeAllDevices() {
    int deviceCount = getNumDevices();
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
    cudaSetDevice(0);
}
