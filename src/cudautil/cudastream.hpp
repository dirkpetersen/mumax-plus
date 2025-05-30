#pragma once

#include <cuda_runtime.h>

cudaStream_t getCudaStream();

// Multi-GPU support functions
void initializeMultiGpu();
int getNumDevices();
void setCurrentDevice(int deviceId);
void synchronizeAllDevices();
