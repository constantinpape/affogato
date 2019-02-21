#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyarray.hpp"

#include "affogato/segmentation/mutex_watershed.hxx"
#include "affogato/segmentation/connected_components.hxx"
#include "affogato/segmentation/zwatershed.hxx"
#include "affogato/segmentation/grid_graph.hxx"

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


    m.def("compute_mws_segmentation_impl",[](const xt::pytensor<int64_t, 1> & sorted_flat_indices,
                                             const xt::pytensor<bool, 1> & valid_edges,
                                             const std::vector<std::vector<int>> & offsets,
                                             const size_t number_of_attractive_channels,
                                             const std::vector<int> & image_shape){
        int64_t number_of_nodes = 1;
        for (auto & s: image_shape){
            number_of_nodes *= s;
        }
        xt::pytensor<uint64_t, 1> node_labeling = xt::zeros<uint64_t>({number_of_nodes});
        {
            py::gil_scoped_release allowThreads;
            segmentation::compute_mws_segmentation(sorted_flat_indices,
                                                   valid_edges,
                                                   offsets,
                                                   number_of_attractive_channels,
                                                   image_shape,
                                                   node_labeling);
        }
        return node_labeling;
    }, py::arg("sorted_flat_indices"),
       py::arg("valid_edges"),
       py::arg("offsets"),
       py::arg("number_of_attractive_channels"),
       py::arg("image_shape"));


    m.def("compute_mws_prim_segmentation_impl",[](const xt::pytensor<float, 1> & edge_weights,
                                                  const xt::pytensor<bool, 1> & valid_edges,
                                                  const std::vector<std::vector<int>> & offsets,
                                                  const size_t number_of_attractive_channels,
                                                  const std::vector<int> & image_shape){
        int64_t number_of_nodes = 1;
        for (auto & s: image_shape){
            number_of_nodes *= s;
        }
        xt::pytensor<uint64_t, 1> node_labeling = xt::zeros<uint64_t>({number_of_nodes});
        {
            py::gil_scoped_release allowThreads;
            segmentation::compute_mws_prim_segmentation(edge_weights,
                                                        valid_edges,
                                                        offsets,
                                                        number_of_attractive_channels,
                                                        image_shape,
                                                        node_labeling);
        }
        return node_labeling;
    }, py::arg("edge_weights"),
       py::arg("valid_edges"),
       py::arg("offsets"),
       py::arg("number_of_attractive_channels"),
       py::arg("image_shape"));


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


    // TODO lift gil where appropriate
    typedef segmentation::MWSGridGraph GraphType;
    py::class_<GraphType>(m, "MWSGridGraph")
        .def(py::init<const xt::pyarray<float> &, const std::vector<std::vector<std::size_t>> &,
                      const std::vector<std::size_t> &, const bool>(),
             py::arg("affinities"), py::arg("offsets"),
             py::arg("strides")=std::vector<std::size_t>({1, 1, 1}),
             py::arg("randomize_strides")=true)

        .def(py::init<const xt::pyarray<float> &, const xt::pyarray<bool> &, const std::vector<std::vector<std::size_t>> &,
                      const std::vector<std::size_t> &, const bool>(),
             py::arg("affinities"), py::arg("mask"), py::arg("offsets"),
             py::arg("strides")=std::vector<std::size_t>({1, 1, 1}),
             py::arg("randomize_strides")=true)

        .def("uv_ids", [](const GraphType & self){
            const auto & uvs = self.uv_ids();
            xt::pytensor<uint64_t, 2> uv_ids = xt::zeros<uint64_t>({static_cast<int64_t>(uvs.size()), static_cast<int64_t>(2)});
            for(std::size_t e = 0; e < uvs.size(); ++e) {
                uv_ids(e, 0) = uvs[e].first;
                uv_ids(e, 1) = uvs[e].second;
            }
            return uv_ids;
        })

        .def("lr_uv_ids", [](const GraphType & self){
            const auto & uvs = self.lr_uv_ids();
            xt::pytensor<uint64_t, 2> uv_ids = xt::zeros<uint64_t>({static_cast<int64_t>(uvs.size()), static_cast<int64_t>(2)});
            for(std::size_t e = 0; e < uvs.size(); ++e) {
                uv_ids(e, 0) = uvs[e].first;
                uv_ids(e, 1) = uvs[e].second;
            }
            return uv_ids;
        })

        .def("weights", [](const GraphType & self){
            const auto & w = self.weights();
            xt::pytensor<float, 1> weights = xt::zeros<float>({static_cast<int64_t>(w.size())});
            for(std::size_t e = 0; e < w.size(); ++e) {
                weights[e] = w[e];
            }
            return weights;
        })

        .def("lr_weights", [](const GraphType & self){
            const auto & w = self.lr_weights();
            xt::pytensor<float, 1> weights = xt::zeros<float>({static_cast<int64_t>(w.size())});
            for(std::size_t e = 0; e < w.size(); ++e) {
                weights[e] = w[e];
            }
            return weights;
        })

        .def("get_causal_edges", [](const GraphType & self,
                                    const xt::pyarray<float> & affs,
                                    const xt::pyarray<uint64_t> & labels,
                                    const std::vector<std::vector<std::size_t>> & offsets){
            std::vector<std::pair<uint64_t, uint64_t>> uvs;
            std::vector<float> w;
            self.get_causal_edges(affs, labels, offsets, uvs, w);

            xt::pytensor<uint64_t, 2> uv_ids = xt::zeros<uint64_t>({static_cast<int64_t>(uvs.size()), static_cast<int64_t>(2)});
            xt::pytensor<float, 1> weights = xt::zeros<float>({static_cast<int64_t>(w.size())});

            for(std::size_t e = 0; e < w.size(); ++e) {
                uv_ids(e, 0) = uvs[e].first;
                uv_ids(e, 1) = uvs[e].second;
                weights[e] = w[e];
            }

            return std::make_pair(uv_ids, weights);
        })
    ;

}
