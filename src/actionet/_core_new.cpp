// Pybind11 interface for ACTIONet
// Main module definition that includes all submodules

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations for initialization functions
void init_action(py::module_ &m);
void init_network(py::module_ &m);
void init_annotation(py::module_ &m);
void init_decomposition(py::module_ &m);
void init_tools(py::module_ &m);
void init_visualization(py::module_ &m);

PYBIND11_MODULE(_core, m) {
    m.doc() = "ACTIONet C++ core bindings";

    // Initialize all submodules
    init_action(m);
    init_network(m);
    init_annotation(m);
    init_decomposition(m);
    init_tools(m);
    init_visualization(m);
}
