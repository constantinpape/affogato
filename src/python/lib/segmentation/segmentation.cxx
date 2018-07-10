#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyarray.hpp"

#include "affogato/segmentation/mutex_watershed.hxx"
#include "affogato/segmentation/connected_components.hxx"

namespace py = pybind11;

PYBIND11_MODULE(_segmentation, m)
{
    xt::import_numpy();
    m.doc() = "segmentation module of affogato";

    using namespace affogato;

    m.def("connected_components", [](const xt::pyarray<float> & affinities,
                                     const float threshold) {
        typedef xt::pyarray<uint64_t>::shape_type ShapeType;
        ShapeType shape(affinities.shape().begin() + 1, affinities.shape().end());
        xt::pyarray<uint64_t> labels = xt::zeros<uint64_t>(shape);
        size_t max_label;
        {
            py::gil_scoped_release allowThreads;
            max_label = segmentation::connected_components(affinities, labels, threshold);
        }
        return std::make_pair(labels, max_label);
    }, py::arg("affinities"),
       py::arg("threshold")
    );


    m.def("compute_mws_clustering",[](const uint64_t number_of_labels,
                                      const xt::pytensor<uint64_t, 2> & uvs,
                                      const xt::pytensor<uint64_t, 2> & mutex_uvs,
                                      const xt::pytensor<float, 1> & weights,
                                      const xt::pytensor<float, 1> & mutex_weights){
        xt::pytensor<uint32_t, 1> node_labeling = xt::zeros<uint64_t>({(int64_t) number_of_labels});
        {
            py::gil_scoped_release allowThreads;
            segmentation::compute_mws_clustering(number_of_labels, uvs,
                                                 mutex_uvs, weights,
                                                 mutex_weights, node_labeling);
        }
        return node_labeling;
    }, py::arg("number_of_labels"),
       py::arg("uvs"), py::arg("mutex_uvs"),
       py::arg("weights"), py::arg("mutex_weights"));
}
