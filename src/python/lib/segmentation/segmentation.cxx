#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyarray.hpp"

#include "affogato/segmentation/mutex_watershed.hxx"
#include "affogato/segmentation/connected_components.hxx"
#include "affogato/segmentation/zwatershed.hxx"

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
        xt::pytensor<uint64_t, 1> node_labeling = xt::zeros<uint64_t>({(int64_t) number_of_labels});
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


    m.def("compute_mws_segmentation_impl",[](const size_t number_of_attractive_channels,
                                        const std::vector<std::vector<int>> & offsets,
                                        const std::vector<int> & image_shape,
                                        const xt::pytensor<int64_t, 1> & sorted_flat_indices,
                                        const xt::pytensor<bool, 1> & valid_edges){
        int64_t number_of_nodes = 1;
        for (auto & s: image_shape){
            number_of_nodes *= s;
        }
        xt::pytensor<uint64_t, 1> node_labeling = xt::zeros<uint64_t>({number_of_nodes});
        {
            py::gil_scoped_release allowThreads;
            segmentation::compute_mws_segmentation(number_of_attractive_channels,
                                                   offsets,
                                                   image_shape,
                                                   sorted_flat_indices,
                                                   valid_edges,
                                                   node_labeling);
        }
        return node_labeling;
    }, py::arg("number_of_attractive_channels"),
       py::arg("offsets"),
       py::arg("image_shape"),
       py::arg("sorted_flat_indices"),
       py::arg("valid_edges"));


    m.def("compute_zws_segmentation",[](const xt::pyarray<float> & edge_weights,
							            const double lower_threshold,
        					            const double upper_threshold,
							            const double merge_threshold,
        					            const size_t size_threshold) {

        typedef typename xt::pyarray<uint64_t>::shape_type ShapeType;
        ShapeType shape(edge_weights.shape().begin() + 1, edge_weights.shape().end());

        size_t n_labels;
        xt::pyarray<uint64_t> labels = xt::zeros<uint64_t>(shape);
        {
            py::gil_scoped_release allowThreads;
            n_labels = segmentation::compute_zws_segmentation(edge_weights,
                                                              lower_threshold, upper_threshold,
                                                              merge_threshold, size_threshold, labels);
        }
        return std::make_pair(labels, n_labels);
    }, py::arg("edge_weights"),
       py::arg("lower_threshold"),
       py::arg("upper_threshold"),
       py::arg("size_threshold"),
       py::arg("merge_threshold"));
}
