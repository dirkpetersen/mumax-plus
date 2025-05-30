#pragma once

#include "field.hpp"
#include "grid.hpp"
#include <vector>
#include <memory>

/**
 * Multi-GPU field distribution and management
 */
class MultiGpuField {
public:
    MultiGpuField(std::shared_ptr<const System> system, int nComponents);
    ~MultiGpuField();
    
    // Distribute field data across GPUs
    void distributeField(const Field& sourceField);
    
    // Gather field data from all GPUs
    Field gatherField() const;
    
    // Get field segment for specific GPU
    Field& getDeviceField(int deviceId);
    const Field& getDeviceField(int deviceId) const;
    
    // Synchronize data between GPUs
    void synchronize();
    
    // Get number of devices
    int getNumDevices() const { return numDevices_; }
    
    // Get device assignment for cell index
    int getDeviceForCell(int cellIdx) const;
    
private:
    std::shared_ptr<const System> system_;
    int nComponents_;
    int numDevices_;
    std::vector<std::unique_ptr<Field>> deviceFields_;
    std::vector<Grid> deviceGrids_;
    std::vector<int> cellToDevice_;
    
    void setupDeviceGrids();
    void setupCellMapping();
};

/**
 * Multi-GPU kernel execution utilities
 */
namespace MultiGpu {
    void initializeMultiGpu();
    bool isMultiGpuEnabled();
    void setMultiGpuEnabled(bool enabled);
    
    // Distribute work across GPUs for field operations
    template<typename KernelFunc, typename... Args>
    void executeOnAllDevices(const MultiGpuField& field, KernelFunc kernel, Args... args);
}
