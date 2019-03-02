#pragma once
#include <boost/pending/disjoint_sets.hpp>
#include "xtensor/xtensor.hpp"
#include <queue>
#include <functional>
#include "affogato/segmentation/mutex_watershed.hxx"
#include "affogato/util.hxx"


namespace affogato {
namespace segmentation {

    //
    // semantic helper functions:
    // check_semantic_constraint: check if two representatives have different semantic labels
    // assign_semantic_label: assign the semantic label of a representative if not yet existent
    // merge_semantic_labels: assign two representatives to the same semantic label
    //

    inline bool check_semantic_constraint(const uint64_t ru, const uint64_t rv,
                                          const auto & semantic_labeling) {
        int64_t slu = semantic_labeling[ru];
        int64_t slv = semantic_labeling[rv];

        if (slu >= 0 && slv >= 0)
            return slu != slv;
        return false;
    }


    // set semantic label of 'ru' to 'sl'
    inline void assign_semantic_label(const uint64_t ru, const int64_t sl,
                                      auto & semantic_labeling) {
        if(semantic_labeling[ru] < 0){
            semantic_labeling[ru] = sl;
        }
    }


    // assign same semantic label to 'ru' and 'rv', assume they don't have different labels
    inline void merge_semantic_labels(const uint64_t ru, const uint64_t rv,
                                      auto & semantic_labeling) {
        int64_t slu = semantic_labeling[ru];
        int64_t slv = semantic_labeling[rv];

        if (slu >= 0 && slv < 0)
            semantic_labeling[rv] = slu;
        if (slu < 0 && slv >= 0)
            semantic_labeling[ru] = slv;
    }


    // compute mutex clustering for a graph with attrative and mutex edges
    template<class EDGE_ARRAY, class WEIGHT_ARRAY, class NODE_ARRAY, class SEMANTIC_ARRAY>
    void compute_semantic_mws_clustering(const size_t number_of_labels,
                                         const xt::xexpression<EDGE_ARRAY> & uvs_exp,
                                         const xt::xexpression<EDGE_ARRAY> & mutex_uvs_exp,
                                         const xt::xexpression<EDGE_ARRAY> & semantic_uts_exp,
                                         const xt::xexpression<WEIGHT_ARRAY> & weights_exp,
                                         const xt::xexpression<WEIGHT_ARRAY> & mutex_weights_exp,
                                         const xt::xexpression<WEIGHT_ARRAY> & semantic_weights_exp,
                                         xt::xexpression<NODE_ARRAY> & node_labeling_exp,
                                         xt::xexpression<SEMANTIC_ARRAY> & semantic_labeling_exp) {

        // casts
        const auto & uvs = uvs_exp.derived_cast();
        const auto & mutex_uvs = mutex_uvs_exp.derived_cast();
        const auto & semantic_uts = semantic_uts_exp.derived_cast();
        const auto & weights = weights_exp.derived_cast();
        const auto & mutex_weights = mutex_weights_exp.derived_cast();
        const auto & semantic_weights = semantic_weights_exp.derived_cast();
        auto & node_labeling = node_labeling_exp.derived_cast();
        auto & semantic_labeling = semantic_labeling_exp.derived_cast();

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
        const size_t num_semantic = semantic_uts.shape()[0];
        const size_t num_internal = num_edges + num_mutex;

        // argsort ALL edges
        // we sort in ascending order
        std::vector<size_t> indices(num_edges + num_mutex + num_semantic);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](const size_t a, const size_t b){
            const double val_a = (a >= num_internal) ? semantic_weights(a - num_internal) :
                                 (a >= num_edges)    ? mutex_weights(a - num_edges)
                                                     : weights(a);
            const double val_b = (b >= num_internal) ? semantic_weights(b - num_internal) :
                                 (b >= num_edges)    ? mutex_weights(b - num_edges)
                                                     : weights(b);
            return val_a > val_b;
        });

        // data-structure storing mutex edges
        typedef std::vector<std::vector<uint64_t>> MutexStorage;
        MutexStorage mutexes(number_of_labels);

        // iterate over all edges
        for(const size_t edge_id : indices) {

            // check wether this edge is semantic
            const bool is_semantic_edge = edge_id >= num_internal;
            // check whether this edge is mutex via the edge offset
            const bool is_mutex_edge = edge_id >= num_edges;

            const size_t id = is_semantic_edge ? edge_id - num_internal :
                              is_mutex_edge ? edge_id - num_edges : edge_id;

            // get first incidental node and its current representative of edge_id
            const uint64_t u = is_semantic_edge ? semantic_uts(id, 0) :
                               is_mutex_edge ? mutex_uvs(id, 0) : uvs(id, 0);
            uint64_t ru = ufd.find_set(u);

            if(!is_semantic_edge){

                // find the edge-id or mutex id and the connected nodes
                const uint64_t v = is_mutex_edge ? mutex_uvs(id, 1) : uvs(id, 1);

                // find the current representatives
                uint64_t rv = ufd.find_set(v);

                // if the nodes are already connected, do nothing
                if(ru == rv) {
                    continue;
                }

                if(check_semantic_constraint(ru, rv, semantic_labeling)){
                    continue;
                }

                // if we already have a mutex, we do not need to do anything
                // (if this is a regular edge, we do not link, if it is a mutex edge
                //  we do not need to insert the redundant mutex constraint)
                if(check_mutex(ru, rv, mutexes)) {
                    continue;
                }

                if(is_mutex_edge) {

                    // insert mutex constraint
                    insert_mutex(ru, rv, id, mutexes);

                } else {

                    // link the nodes and merge their mutex constraints
                    ufd.link(u, v);
                    // check  if we have to swap the roots
                    if(ufd.find_set(ru) == rv) {
                        std::swap(ru, rv);
                    }
                    // merge mutexes from rv -> ru
                    merge_mutexes(rv, ru, mutexes);
                }
            } else{
                // find semantic label associated to edge
                int64_t t = semantic_uts(id, 1);
                assign_semantic_label(ru, t, semantic_labeling);
            }
        }

        // get node labeling into output
        util::export_consecutive_labels(ufd, number_of_labels, node_labeling);

        for(size_t label = 0; label < number_of_labels; ++label) {
            uint64_t root = ufd.find_set(label);
            semantic_labeling[label] = semantic_labeling[root];
        }
    }


    // compute mutex segmentation via kruskal
    template<class WEIGHT_ARRAY, class NODE_ARRAY, class SEMANTIC_ARRAY, class INDICATOR_ARRAY>
    void compute_semantic_mws_segmentation(const xt::xexpression<WEIGHT_ARRAY> & sorted_flat_indices_exp,
                                           const xt::xexpression<INDICATOR_ARRAY> & valid_edges_exp,
                                           const std::vector<std::vector<int>> & offsets,
                                           const size_t number_of_attractive_channels,
                                           const std::vector<int> & image_shape,
                                           xt::xexpression<NODE_ARRAY> & node_labeling_exp,
                                           xt::xexpression<SEMANTIC_ARRAY> & semantic_labeling_exp) {

        // casts
        const auto & sorted_flat_indices = sorted_flat_indices_exp.derived_cast();
        const auto & valid_edges = valid_edges_exp.derived_cast();
        auto & node_labeling = node_labeling_exp.derived_cast();
        auto & semantic_labeling = semantic_labeling_exp.derived_cast();

        // determine number of nodes and attractive edges
        const size_t number_of_nodes = node_labeling.size();
        const size_t number_of_attractive_edges = number_of_nodes * number_of_attractive_channels;
        const size_t number_of_offsets = offsets.size();
        const size_t ndims = offsets[0].size();
        const size_t number_of_internal_edges = number_of_nodes * number_of_offsets;

        std::vector<int64_t> array_stride(ndims);
        int64_t current_stride = 1;
        for (int i = ndims-1; i >= 0; --i){
            array_stride[i] = current_stride;
            current_stride *= image_shape[i];
        }

        std::vector<int64_t> offset_strides;
        for (const auto & offset: offsets){
            int64_t stride = 0;
            for (int i = 0; i < offset.size(); ++i){
                stride += offset[i] * array_stride[i];
            }
            offset_strides.push_back(stride);
        }

        // make ufd
        std::vector<uint64_t> ranks(number_of_nodes);
        std::vector<uint64_t> parents(number_of_nodes);
        boost::disjoint_sets<uint64_t*, uint64_t*> ufd(&ranks[0], &parents[0]);
        for(uint64_t label = 0; label < number_of_nodes; ++label) {
            ufd.make_set(label);
        }

        // data-structure for storing semantic labels, unassigned are set to 0
        // std::vector<uint64_t> semantic_labeling(number_of_nodes, 0);

        // data-structure storing mutex edges
        typedef std::vector<std::vector<uint64_t>> MutexStorage;
        MutexStorage mutexes(number_of_nodes);

        // iterate over all edges
        for(const size_t edge_id : sorted_flat_indices) {

            if(!valid_edges(edge_id)){
                continue;
            }

            // check wether this edge is semantic
            const bool is_semantic_edge = edge_id >= number_of_internal_edges;
            // check whether this edge is mutex via the edge offset
            const bool is_mutex_edge = edge_id >= number_of_attractive_edges;

            // get first incidental node and its current representative of edge_id
            const uint64_t u = edge_id % number_of_nodes;
            uint64_t ru = ufd.find_set(u);

            if(!is_semantic_edge){
                // const auto affCoord_ = xt::unravel_from_strides(edge_id, strides, layout);


                // get second incidental node
                const uint64_t v = u + offset_strides[edge_id / number_of_nodes];
                // find the current representative
                uint64_t rv = ufd.find_set(v);

                // if the nodes are already connected, do nothing
                if(ru == rv) {
                    continue;
                }

                if(check_semantic_constraint(ru, rv, semantic_labeling)){
                    continue;
                }

                // if we already have a mutex, we do not need to do anything
                // (if this is a regular edge, we do not link, if it is a mutex edge
                //  we do not need to insert the redundant mutex constraint)
                if(check_mutex(ru, rv, mutexes)) {
                    continue;
                }

                if(is_mutex_edge) {

                    // insert the mutex edge into both mutex edge storages
                    insert_mutex(ru, rv, edge_id, mutexes);

                } else {

                    ufd.link(u, v);
                    // check  if we have to swap the roots
                    if(ufd.find_set(ru) == rv) {
                        std::swap(ru, rv);
                    }
                    // merge mutexes from rv -> ru
                    merge_mutexes(rv, ru, mutexes);
                    merge_semantic_labels(ru, rv, semantic_labeling);
                }
            }else{
                // find semantic label associated to edge
                int64_t sl = edge_id / number_of_nodes - number_of_offsets;

                // add semantic class to cluster
                assign_semantic_label(ru, sl, semantic_labeling);
            }
        }

        // get node labeling into output
        util::export_consecutive_labels(ufd, number_of_nodes, node_labeling);
        
        for(size_t label = 0; label < number_of_nodes; ++label) {
            uint64_t root = ufd.find_set(label);
            semantic_labeling[label] = semantic_labeling[root];
        }
    }
}
}
