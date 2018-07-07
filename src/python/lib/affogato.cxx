#include <pybind11/pybind11.h>
namespace py = pybind11;


PYBIND11_MODULE(_affogato, m) {

    py::options options;
    m.doc() = "affogato python bindings";
}
