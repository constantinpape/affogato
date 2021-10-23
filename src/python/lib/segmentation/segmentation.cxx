#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyarray.hpp"

#include "affogato/segmentation/mutex_watershed.hxx"
#include "affogato/segmentation/semantic_mutex_watershed.hxx"
#include "affogato/segmentation/connected_components.hxx"
#include "affogato/segmentation/zwatershed.hxx"
#include "affogato/segmentation/grid_graph.hxx"
#include "affogato/segmentation/single_linkage.hxx"

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


    m.def("compute_mws_segmentation_v2_impl",[](const xt::pytensor<int64_t, 1> & sorted_flat_indices,
                                             const xt::pytensor<bool, 1> & valid_edges,
                                             const xt::pytensor<bool, 1> & mutex_edges,
                                             const std::vector<std::vector<int>> & offsets,
                                             const std::vector<int> & image_shape){
              int64_t number_of_nodes = 1;
              for (auto & s: image_shape){
                  number_of_nodes *= s;
              }
              xt::pytensor<uint64_t, 1> node_labeling = xt::zeros<uint64_t>({number_of_nodes});
              {
                  py::gil_scoped_release allowThreads;
                  segmentation::compute_mws_segmentation_v2(sorted_flat_indices,
                                                         valid_edges,
                                                         mutex_edges,
                                                         offsets,
                                                         image_shape,
                                                         node_labeling);
              }
              return node_labeling;
          }, py::arg("sorted_flat_indices"),
          py::arg("valid_edges"),
          py::arg("mutex_edges"),
          py::arg("offsets"),
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

    m.def("compute_semantic_mws_clustering",[](const uint64_t number_of_labels,
                                      const xt::pytensor<uint64_t, 2> & uvs,
                                      const xt::pytensor<uint64_t, 2> & mutex_uvs,
                                      const xt::pytensor<uint64_t, 2> & semantic_uts,
                                      const xt::pytensor<float, 1> & weights,
                                      const xt::pytensor<float, 1> & mutex_weights,
                                      const xt::pytensor<float, 1> & semantic_weights){
        xt::pytensor<uint64_t, 1> node_labeling = xt::zeros<uint64_t>({(int64_t) number_of_labels});
        xt::pytensor<int64_t, 1> semantic_labeling = - xt::ones<int64_t>({(int64_t) number_of_labels});
        {
            py::gil_scoped_release allowThreads;
            segmentation::compute_semantic_mws_clustering(number_of_labels,
                                                 uvs, mutex_uvs, semantic_uts,
                                                 weights, mutex_weights, semantic_weights,
                                                 node_labeling, semantic_labeling);
        }
        py::tuple out = py::make_tuple(node_labeling, semantic_labeling);
        return out;
    }, py::arg("number_of_labels"),
       py::arg("uvs"), py::arg("mutex_uvs"), py::arg("semantic_uts"),
       py::arg("weights"), py::arg("mutex_weights"), py::arg("semantic_weights"));


    m.def("compute_semantic_mws_segmentation_impl",[](const xt::pytensor<int64_t, 1> & sorted_flat_indices,
                                             const xt::pytensor<bool, 1> & valid_edges,
                                             const std::vector<std::vector<int>> & offsets,
                                             const size_t number_of_attractive_channels,
                                             const std::vector<int> & image_shape){
        int64_t number_of_nodes = 1;
        for (auto & s: image_shape){
            number_of_nodes *= s;
        }
        xt::pytensor<uint64_t, 1> node_labeling = xt::zeros<uint64_t>({number_of_nodes});
        xt::pytensor<int64_t, 1> semantic_labeling = - xt::ones<int64_t>({number_of_nodes});
        {
            py::gil_scoped_release allowThreads;
            segmentation::compute_semantic_mws_segmentation(sorted_flat_indices,
                                                   valid_edges,
                                                   offsets,
                                                   number_of_attractive_channels,
                                                   image_shape,
                                                   node_labeling,
                                                   semantic_labeling);
        }
        py::tuple out = py::make_tuple(node_labeling, semantic_labeling);
        return out;
    }, py::arg("sorted_flat_indices"),
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
        .def(py::init<const std::vector<std::size_t> &>(), py::arg("shape"))
        .def_property_readonly("n_nodes", &GraphType::n_nodes)
        .def_property("add_attractive_seed_edges",
                      &GraphType::get_add_attactive_seed_edges,
                      &GraphType::set_add_attactive_seed_edges)

        //
        // mask functionality
        //

        .def("set_mask", [](GraphType & self,
                            const xt::pyarray<bool> & mask){
            self.set_mask(mask);
        }, py::arg("mask"))
        .def("clear_mask", &GraphType::clear_mask)

        //
        // seed functionality
        //

        .def("update_seeds", [](GraphType & self,
                                const xt::pyarray<uint64_t> & seeds){
            self.update_seeds(seeds);
        }, py::arg("seeds"))
        .def("clear_seeds", &GraphType::clear_seeds)

        //
        // node and coordinate functionality
        //

        .def("get_node", [](const GraphType & self, const std::vector<int64_t> & coordinate){
            return self.get_node(coordinate);
        }, py::arg("coordinate"))

        .def("get_nodes", [](const GraphType & self,
                             const xt::pytensor<int64_t, 2> & coordinates){
            const auto & shape = coordinates.shape();
            xt::pytensor<uint64_t, 1>  nodes = xt::zeros<uint64_t>({shape[0]});
            xt::xindex coord(shape[1]);
            for(std::size_t ii = 0; ii < shape[0]; ++ii) {
                for(unsigned d = 0; d < shape[1]; ++d) {
                    coord[d] = coordinates(ii, d);
                }
                nodes(ii) = self.get_node(coord);
            }
            return nodes;
        }, py::arg("coordinates"))

        .def("get_coordinate", [](const GraphType & self, const uint64_t node){
            return self.get_coordinate(node);
        }, py::arg("node"))

        .def("get_coordinates", [](const GraphType & self,
                                   const xt::pytensor<uint64_t, 1> & nodes){
            const auto & shape = nodes.shape();
            const unsigned ndim = self.ndim();
            xt::pytensor<int64_t, 2> coordinates = xt::zeros<int64_t>(
                {static_cast<int64_t>(shape[0]), static_cast<int64_t>(ndim)}
            );

            for(std::size_t ii = 0; ii < shape[0]; ++ii) {
                const auto coord = self.get_coordinate(nodes(ii));
                for(unsigned d = 0; d < ndim; ++d) {
                    coordinates(ii, d) = coord[d];
                }
            }

            return coordinates;
        }, py::arg("nodes"))

        .def("relabel_to_seeds", [](const GraphType & self,
                                    xt::pytensor<uint64_t, 1> & node_labels){
            self.relabel_to_seeds(node_labels);
            return node_labels;
        }, py::arg("node_labels"))

        .def("compute_state_for_segmentation", [](const GraphType & self,
                                                  const xt::pyarray<float> & affs,
                                                  const xt::pyarray<uint64_t> & seg,
                                                  const std::vector<std::vector<int>> & offsets,
                                                  const unsigned n_attactive_channels,
                                                  const bool ignore_label) {

            typedef std::unordered_map<std::pair<uint64_t, uint64_t>,
                                       std::pair<float, bool>,
                                       boost::hash<std::pair<uint64_t, uint64_t>>
                                      > StateType;
            StateType state;
            self.compute_state_for_segmentation(affs, seg, offsets,
                                                n_attactive_channels, ignore_label, state);

            const int64_t n_edges = state.size();
            xt::pytensor<uint64_t, 2> edges = xt::zeros<uint64_t>(
                {n_edges, static_cast<int64_t>(n_edges)}
            );
            xt::pytensor<float, 1> weights = xt::zeros<float>({n_edges});
            xt::pytensor<bool, 1> is_attractive = xt::zeros<bool>({n_edges});

            std::size_t edge_id = 0;
            for(const auto & edge : state) {
                const auto & uv = edge.first;
                const auto & edge_state = edge.second;

                edges(edge_id, 0) = uv.first;
                edges(edge_id, 1) = uv.second;

                weights(edge_id) = edge_state.first;
                is_attractive(edge_id) = edge_state.second;

                ++edge_id;
            }

            return std::make_tuple(edges, weights, is_attractive);

        }, py::arg("affinities"), py::arg("segmentation"),
           py::arg("offsets"), py::arg("n_attractive_channels"),
           py::arg("ignore_label")=true)

        .def("compute_nh_and_weights", [](const GraphType & self,
                                          const xt::pyarray<float> & affs,
                                          const std::vector<std::vector<int>> & offsets,
                                          const std::vector<std::size_t> & strides,
                                          const bool randomize_strides){

            std::vector<std::pair<uint64_t, uint64_t>> uvs;
            std::vector<float> w;
            self.compute_nh_and_weights(affs, offsets, strides, randomize_strides, uvs, w);

            xt::pytensor<uint64_t, 2> uv_ids = xt::zeros<uint64_t>({static_cast<int64_t>(uvs.size()), static_cast<int64_t>(2)});
            for(std::size_t e = 0; e < uvs.size(); ++e) {
                uv_ids(e, 0) = uvs[e].first;
                uv_ids(e, 1) = uvs[e].second;
            }

            xt::pytensor<float, 1> weights = xt::zeros<float>({static_cast<int64_t>(w.size())});
            for(std::size_t e = 0; e < w.size(); ++e) {
                weights[e] = w[e];
            }

            return std::make_pair(uv_ids, weights);
        },py::arg("affinities"), py::arg("offsets"),
          py::arg("strides")=std::vector<std::size_t>({1, 1, 1}),
          py::arg("randomize_strides")=true)
    ;

    m.def("compute_single_linkage_clustering",[](const uint64_t number_of_labels,
                                                 const xt::pytensor<uint64_t, 2> & uvs,
                                                 const xt::pytensor<uint64_t, 2> & mutex_uvs,
                                                 const xt::pytensor<float, 1> & weights,
                                                 const xt::pytensor<float, 1> & mutex_weights){
              xt::pytensor<uint64_t, 1> node_labeling = xt::zeros<uint64_t>({(int64_t) number_of_labels});
              {
                  py::gil_scoped_release allowThreads;
                  segmentation::compute_single_linkage_clustering(number_of_labels, uvs,
                                                    mutex_uvs, weights,
                                                    mutex_weights, node_labeling);
              }
              return node_labeling;
          }, py::arg("number_of_labels"),
          py::arg("uvs"), py::arg("mutex_uvs"),
          py::arg("weights"), py::arg("mutex_weights"));

}
