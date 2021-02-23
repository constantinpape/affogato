#include "boost/pending/disjoint_sets.hpp"
#include "xtensor/xtensor.hpp"
#include "affogato/util.hxx"


namespace affogato {
    namespace segmentation {
        template<class EDGE_ARRAY, class WEIGHT_ARRAY, class NODE_ARRAY>
        void compute_single_linkage_clustering(const size_t number_of_labels,
                                               const xt::xexpression<EDGE_ARRAY> & uvs_exp,
                                               const xt::xexpression<EDGE_ARRAY> & mutex_uvs_exp,
                                               const xt::xexpression<WEIGHT_ARRAY> & weights_exp,
                                               const xt::xexpression<WEIGHT_ARRAY> & mutex_weights_exp,
                                               xt::xexpression<NODE_ARRAY> & node_labeling_exp) {

            // casts
            const auto & uvs = uvs_exp.derived_cast();
            const auto & mutex_uvs = mutex_uvs_exp.derived_cast();
            const auto & weights = weights_exp.derived_cast();
            const auto & mutex_weights = mutex_weights_exp.derived_cast();
            auto & node_labeling = node_labeling_exp.derived_cast();

            // make ufd
            std::vector<uint64_t> ranks(number_of_labels);
            std::vector<uint64_t> parents(number_of_labels);
            boost::disjoint_sets<uint64_t*, uint64_t*> ufd(&ranks[0], &parents[0]);
            for(uint64_t label = 0; label < number_of_labels; ++label) {
                ufd.make_set(label);
            }

            // determine number of edge types
            const size_t num_edges = uvs.shape()[0];
            const size_t num_mutex = mutex_uvs.shape()[0];

            // argsort ALL edges
            // we sort in ascending order
            std::vector<size_t> indices(num_edges + num_mutex);
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(), [&](const size_t a, const size_t b){
                const double val_a = (a < num_edges) ? weights(a) : -mutex_weights(a - num_edges);
                const double val_b = (b < num_edges) ? weights(b) : -mutex_weights(b - num_edges);
                return val_a > val_b;
            });


            // iterate over all edges
            int counter = 0;
            for(const size_t edge_id : indices) {
                counter++;

                // check whether this edge is mutex via the edge offset
                const bool is_mutex_edge = edge_id >= num_edges;

                if (is_mutex_edge) {
                    break;
                }

                // find the edge-id or mutex id and the connected nodes
                const size_t id = is_mutex_edge ? edge_id - num_edges : edge_id;
                const uint64_t u = is_mutex_edge ? mutex_uvs(id, 0) : uvs(id, 0);
                const uint64_t v = is_mutex_edge ? mutex_uvs(id, 1) : uvs(id, 1);

                // find the current representatives
                uint64_t ru = ufd.find_set(u);
                uint64_t rv = ufd.find_set(v);

                // if the nodes are not connected yet, merge
                if(ru != rv) {
                    // link the nodes and merge their mutex constraints
                    ufd.link(u, v);
                    // check  if we have to swap the roots
                    if(ufd.find_set(ru) == rv) {
                        std::swap(ru, rv);
                    }
                }

            }
            // get node labeling into output
            for(size_t label = 0; label < number_of_labels; ++label) {
                node_labeling[label] = ufd.find_set(label);
            }
        }
    }
}

