#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyarray.hpp"

#include "affogato/learning/malis.hxx"
#include "affogato/learning/mutex_malis.hxx"

namespace py = pybind11;

PYBIND11_MODULE(_learning, m)
{
    xt::import_numpy();
    m.doc() = "learning module of affogato";

    using namespace affogato;

    // FIXME weird xtensor error prevents nd impl
    m.def("compute_malis_2d", [](const xt::pytensor<float, 3> & affinities,
                                 const xt::pytensor<uint64_t, 2> & labels,
                                 const std::vector<std::vector<int>> & offsets,
                                 const int pass) {
        //
        const auto & aff_shape = affinities.shape();
        double loss;
        xt::pytensor<float, 3> gradients = xt::zeros<float>(aff_shape);
        {
            py::gil_scoped_release allowThreads;
            if(pass == 0) {
                // pass = 0: we compute the constrained malis gradient
                loss = learning::constrained_malis(affinities, labels, gradients, offsets);
            } else if(pass == 1) {
                // pass = 1: we compute the unconstrained malis gradient for the positive pass
                loss = learning::malis_gradient(affinities, labels, gradients, offsets, true);
            } else if(pass == 2) {
                // pass = 2: we compute the unconstrained malis gradient for the negative pass
                loss = learning::malis_gradient(affinities, labels, gradients, offsets, false);
            } else {
                throw std::runtime_error("Invalid malis pass option");
            }
        }
        return std::make_pair(loss, gradients);
    }, py::arg("affinities"),
       py::arg("labels"),
       py::arg("offsets"),
       py::arg("pass")=0);

    m.def("compute_malis_3d", [](const xt::pytensor<float, 4> & affinities,
                                 const xt::pytensor<uint64_t, 3> & labels,
                                 const std::vector<std::vector<int>> & offsets,
                                 const int pass) {
        //
        const auto & aff_shape = affinities.shape();
        double loss;
        xt::pytensor<float, 4> gradients = xt::zeros<float>(aff_shape);
        {
            py::gil_scoped_release allowThreads;
            if(pass == 0) {
                // pass = 0: we compute the constrained malis gradient
                loss = learning::constrained_malis(affinities, labels, gradients, offsets);
            } else if(pass == 1) {
                // pass = 1: we compute the unconstrained malis gradient for the positive pass
                loss = learning::malis_gradient(affinities, labels, gradients, offsets, true);
            } else if(pass == 2) {
                // pass = 2: we compute the unconstrained malis gradient for the negative pass
                loss = learning::malis_gradient(affinities, labels, gradients, offsets, false);
            } else {
                throw std::runtime_error("Invalid malis pass option");
            }
        }
        return std::make_pair(loss, gradients);
    }, py::arg("affinities"),
       py::arg("labels"),
       py::arg("offsets"),
       py::arg("pass")=0);


    m.def("compute_mutex_malis", [](const xt::pytensor<float, 1> & flat_weights,
                                    const xt::pytensor<uint64_t, 1> & sorted_flat_indices,
                                    const xt::pytensor<bool, 1> & valid_edges,
                                    const xt::pytensor<uint64_t, 1> & gt_labels,
                                    const std::vector<std::vector<int>> & offsets,
                                    const size_t number_of_attractive_channels,
                                    const std::vector<int> & image_shape,
                                    const int pass) {
        double loss = 0;

        xt::pytensor<float, 1> gradients = xt::zeros<float>({flat_weights.size()});
        {
            // TODO properly switch between the passes
            py::gil_scoped_release allowThreads;
            loss = learning::compute_mutex_malis_gradient(flat_weights, sorted_flat_indices,
                                                          valid_edges, gt_labels,
                                                          offsets, number_of_attractive_channels,
                                                          image_shape, 0, gradients);
        }
        return std::make_pair(loss, gradients);
    }, py::arg("flat_weights"),
       py::arg("sorted_flat_indices"),
       py::arg("valid_edges"),
       py::arg("gt_labels"),
       py::arg("offsets"),
       py::arg("number_of_attractive_channels"),
       py::arg("image_shape"),
       py::arg("pass")=0);
}
