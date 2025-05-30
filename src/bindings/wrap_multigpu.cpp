#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "multigpu.hpp"
#include "field.hpp"
#include "system.hpp"
#include "wrappers.hpp"

namespace py = pybind11;

void wrap_multigpu(py::module& m) {
    // MultiGpuField class
    py::class_<MultiGpuField>(m, "MultiGpuField")
        .def(py::init<std::shared_ptr<const System>, int>(),
             py::arg("system"), py::arg("nComponents"),
             "Create a multi-GPU field with given system and number of components")
        .def("distribute_field", &MultiGpuField::distributeField,
             py::arg("source_field"),
             "Distribute a field across multiple GPUs")
        .def("gather_field", &MultiGpuField::gatherField,
             "Gather field data from all GPUs into a single field")
        .def("get_device_field", 
             py::overload_cast<int>(&MultiGpuField::getDeviceField),
             py::arg("device_id"),
             "Get the field segment for a specific GPU",
             py::return_value_policy::reference_internal)
        .def("synchronize", &MultiGpuField::synchronize,
             "Synchronize data between GPUs")
        .def("get_num_devices", &MultiGpuField::getNumDevices,
             "Get the number of GPUs being used")
        .def("get_device_for_cell", &MultiGpuField::getDeviceForCell,
             py::arg("cell_idx"),
             "Get which GPU handles a specific cell index");

    // Multi-GPU namespace functions
    py::module multigpu = m.def_submodule("multigpu", "Multi-GPU utilities");
    
    multigpu.def("initialize", &MultiGpu::initializeMultiGpu,
                 "Initialize multi-GPU support");
    multigpu.def("is_enabled", &MultiGpu::isMultiGpuEnabled,
                 "Check if multi-GPU support is enabled");
    multigpu.def("set_enabled", &MultiGpu::setMultiGpuEnabled,
                 py::arg("enabled"),
                 "Enable or disable multi-GPU support");
}
