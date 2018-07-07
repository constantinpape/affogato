#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyarray.hpp"

#include "affogato/learning/malis.hxx"

namespace py = pybind11;

PYBIND11_MODULE(_learning, m)
{
    xt::import_numpy();
    m.doc() = "learning module of affogato";

    using namespace affogato;

    // FIXME weird xtensor error prevents nd impl
    m.def("compute_malis_2d", [](const xt::pytensor<float, 3> & affinities,
                                 const xt::pytensor<uint64_t, 2> & labels,
                                 const std::vector<std::vector<int>> & offsets) {
        //
        const auto & affShape = affinities.shape();
        double loss;
        xt::pytensor<float, 3> gradients(affShape);
        {
            py::gil_scoped_release allowThreads;
            loss = learning::constrained_malis(affinities, labels, gradients, offsets);
        }
        return std::make_pair(loss, gradients);
    }, py::arg("affinities"),
       py::arg("labels"),
       py::arg("offsets"));

    m.def("compute_malis_3d", [](const xt::pytensor<float, 4> & affinities,
                                 const xt::pytensor<uint64_t, 3> & labels,
                                 const std::vector<std::vector<int>> & offsets) {
        //
        const auto & affShape = affinities.shape();
        double loss;
        xt::pytensor<float, 4> gradients(affShape);
        {
            py::gil_scoped_release allowThreads;
            loss = learning::constrained_malis(affinities, labels, gradients, offsets);
        }
        return std::make_pair(loss, gradients);
    }, py::arg("affinities"),
       py::arg("labels"),
       py::arg("offsets"));
}
