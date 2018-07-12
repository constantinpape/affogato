#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyarray.hpp"

#include "affogato/affinities/affinities.hxx"
#include "affogato/affinities/multiscale_affinities.hxx"

namespace py = pybind11;

PYBIND11_MODULE(_affinities, m)
{
    xt::import_numpy();
    m.doc() = "affinity module of affogato";

    using namespace affogato;

    m.def("compute_multiscale_affinities", [](const xt::pyarray<uint64_t> & labels,
                                              const std::vector<int> & block_shape,
                                              const bool have_ignore_label,
                                              const uint64_t ignore_label) {
            // compute the out shape
            typedef typename xt::pyarray<float>::shape_type ShapeType;
            const auto & shape = labels.shape();
            const unsigned ndim = shape.size();
            ShapeType out_shape(ndim + 1);
            out_shape[0] = ndim;
            for(unsigned d = 0; d < ndim; ++d) {
                // integer division should do the right thing in all cases
                out_shape[d + 1] = (shape[d] % block_shape[d]) ? shape[d] / block_shape[d] + 1 : shape[d] / block_shape[d];
            }

            // allocate the output
            xt::pyarray<float> affs(out_shape);
            xt::pyarray<uint8_t> mask(out_shape);
            {
                py::gil_scoped_release allowThreads;
                affinities::compute_multiscale_affinities(labels, block_shape,
                                                          affs, mask,
                                                          have_ignore_label, ignore_label);
            }
            return std::make_pair(affs, mask);
        }, py::arg("labels"),
           py::arg("block_shape"),
           py::arg("have_ignore_label")=false,
           py::arg("ignore_label")=0);


    m.def("compute_affinities", [](const xt::pyarray<uint64_t> & labels,
                                   const std::vector<std::vector<int>> & offsets,
                                   const bool have_ignore_label,
                                   const uint64_t ignore_label) {
            // compute the out shape
            typedef typename xt::pyarray<float>::shape_type ShapeType;
            const auto & shape = labels.shape();
            const unsigned ndim = labels.dimension();
            ShapeType out_shape(ndim + 1);
            out_shape[0] = offsets.size();
            for(unsigned d = 0; d < ndim; ++d) {
                out_shape[d + 1] = shape[d];
            }

            // allocate the output
            xt::pyarray<float> affs = xt::zeros<float>(out_shape);
            xt::pyarray<uint8_t> mask = xt::zeros<uint8_t>(out_shape);
            {
                py::gil_scoped_release allowThreads;
                affinities::compute_affinities(labels, offsets,
                                               affs, mask,
                                               have_ignore_label, ignore_label);
            }
            return std::make_pair(affs, mask);
        }, py::arg("labels"),
           py::arg("offset"),
           py::arg("have_ignore_label")=false,
           py::arg("ignore_label")=0);
}
