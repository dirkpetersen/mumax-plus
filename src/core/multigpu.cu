#include "multigpu.hpp"
#include "cudastream.hpp"
#include "cudalaunch.hpp"
#include "cudaerror.hpp"
#include <algorithm>

static bool multiGpuEnabled = false;

MultiGpuField::MultiGpuField(std::shared_ptr<const System> system, int nComponents)
    : system_(system), nComponents_(nComponents) {
    numDevices_ = getNumDevices();
    
    if (numDevices_ > 1 && multiGpuEnabled) {
        setupDeviceGrids();
        setupCellMapping();
        
        // Create field segments for each device
        for (int i = 0; i < numDevices_; i++) {
            auto deviceSystem = std::make_shared<System>(system_->world(), deviceGrids_[i]);
            deviceFields_.push_back(std::make_unique<Field>(deviceSystem, nComponents_));
        }
    } else {
        // Single GPU fallback
        deviceFields_.push_back(std::make_unique<Field>(system_, nComponents_));
        deviceGrids_.push_back(system_->grid());
        numDevices_ = 1;
    }
}

MultiGpuField::~MultiGpuField() = default;

void MultiGpuField::setupDeviceGrids() {
    Grid originalGrid = system_->grid();
    int3 totalSize = originalGrid.size();
    
    // Distribute along Z-axis for better memory coalescing
    int cellsPerDevice = (totalSize.z + numDevices_ - 1) / numDevices_;
    
    for (int i = 0; i < numDevices_; i++) {
        int startZ = i * cellsPerDevice;
        int endZ = std::min(startZ + cellsPerDevice, totalSize.z);
        int deviceSizeZ = endZ - startZ;
        
        if (deviceSizeZ > 0) {
            int3 deviceSize = {totalSize.x, totalSize.y, deviceSizeZ};
            int3 deviceOrigin = {originalGrid.origin().x, originalGrid.origin().y, 
                               originalGrid.origin().z + startZ};
            deviceGrids_.push_back(Grid(deviceSize, deviceOrigin));
        }
    }
}

void MultiGpuField::setupCellMapping() {
    Grid originalGrid = system_->grid();
    int totalCells = originalGrid.ncells();
    cellToDevice_.resize(totalCells);
    
    for (int cellIdx = 0; cellIdx < totalCells; cellIdx++) {
        int3 coord = originalGrid.index2coord(cellIdx);
        
        // Find which device this cell belongs to based on Z coordinate
        int deviceId = 0;
        for (int dev = 0; dev < numDevices_; dev++) {
            if (coord.z >= deviceGrids_[dev].origin().z && 
                coord.z < deviceGrids_[dev].origin().z + deviceGrids_[dev].size().z) {
                deviceId = dev;
                break;
            }
        }
        cellToDevice_[cellIdx] = deviceId;
    }
}

void MultiGpuField::distributeField(const Field& sourceField) {
    if (numDevices_ == 1) {
        *deviceFields_[0] = sourceField;
        return;
    }
    
    // Copy data to each device's portion
    for (int dev = 0; dev < numDevices_; dev++) {
        setCurrentDevice(dev);
        
        Grid deviceGrid = deviceGrids_[dev];
        Field& deviceField = *deviceFields_[dev];
        
        // Copy relevant portion of source field to device field
        for (int deviceIdx = 0; deviceIdx < deviceGrid.ncells(); deviceIdx++) {
            int3 deviceCoord = deviceGrid.index2coord(deviceIdx);
            int3 globalCoord = deviceCoord; // Already in global coordinates
            
            if (sourceField.system()->grid().cellInGrid(globalCoord)) {
                int globalIdx = sourceField.system()->grid().coord2index(globalCoord);
                
                for (int comp = 0; comp < nComponents_; comp++) {
                    // This would need proper memory transfer implementation
                    // For now, this is a conceptual framework
                }
            }
        }
    }
}

Field MultiGpuField::gatherField() const {
    Field result(system_, nComponents_);
    
    if (numDevices_ == 1) {
        result = *deviceFields_[0];
        return result;
    }
    
    // Gather data from all devices
    for (int dev = 0; dev < numDevices_; dev++) {
        const Field& deviceField = *deviceFields_[dev];
        Grid deviceGrid = deviceGrids_[dev];
        
        // Copy device field data back to result
        for (int deviceIdx = 0; deviceIdx < deviceGrid.ncells(); deviceIdx++) {
            int3 deviceCoord = deviceGrid.index2coord(deviceIdx);
            
            if (result.system()->grid().cellInGrid(deviceCoord)) {
                int globalIdx = result.system()->grid().coord2index(deviceCoord);
                
                for (int comp = 0; comp < nComponents_; comp++) {
                    // This would need proper memory transfer implementation
                }
            }
        }
    }
    
    return result;
}

Field& MultiGpuField::getDeviceField(int deviceId) {
    if (deviceId >= 0 && deviceId < numDevices_) {
        return *deviceFields_[deviceId];
    }
    return *deviceFields_[0];
}

const Field& MultiGpuField::getDeviceField(int deviceId) const {
    if (deviceId >= 0 && deviceId < numDevices_) {
        return *deviceFields_[deviceId];
    }
    return *deviceFields_[0];
}

int MultiGpuField::getDeviceForCell(int cellIdx) const {
    if (cellIdx >= 0 && cellIdx < cellToDevice_.size()) {
        return cellToDevice_[cellIdx];
    }
    return 0;
}

void MultiGpuField::synchronize() {
    if (numDevices_ > 1) {
        synchronizeAllDevices();
    }
}

namespace MultiGpu {
    void initializeMultiGpu() {
        ::initializeMultiGpu();
    }
    
    bool isMultiGpuEnabled() {
        return multiGpuEnabled && getNumDevices() > 1;
    }
    
    void setMultiGpuEnabled(bool enabled) {
        multiGpuEnabled = enabled;
        if (enabled) {
            initializeMultiGpu();
        }
    }
}
