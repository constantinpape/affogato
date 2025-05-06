#pragma once
#include <boost/pending/disjoint_sets.hpp>
#include <boost/container/flat_set.hpp>
#include "xtensor.hpp"
#include "affogato/util.hxx"
#include <queue>
#include <functional>

namespace affogato {
namespace segmentation {

    //
    // mutex helper functions:
    // check_mutex: check if mutex exists between two representatives
    // insert_mutex: insert mutex between two representatives
    // merge_mutexex: merge mutex constrained of two mutex stores
    //


    template<class MUTEX_STORAGE>
    inline bool check_mutex(const uint64_t ru, const uint64_t rv,
                            const MUTEX_STORAGE & mutexes) {
        const auto & size_set_u  = mutexes[ru].size();
        const auto & size_set_v  = mutexes[rv].size();
        const auto & smaller_set = (size_set_u < size_set_v) ? mutexes[ru] : mutexes[rv];
        const auto & node_to_check = (size_set_u < size_set_v) ? rv : ru;
        // Look for the other node in the smaller constraints set:
        if(smaller_set.find(node_to_check) != smaller_set.end()){
            return true;
        } else{
            return false;
        }
    }


    // Constrain the two root nodes 'ru' and 'rv'
    template<class MUTEX_STORAGE>
    inline void insert_mutex(const uint64_t ru, const uint64_t rv,
                             MUTEX_STORAGE & mutexes) {
        mutexes[ru].insert(rv);
        mutexes[rv].insert(ru);
    }


    // merge the mutex edges by merging from 'root_from' to 'root_to'
    // and avoid double constraints between the same two clusters
    template<class MUTEX_STORAGE>
    inline void merge_mutexes(const uint64_t root_from, const uint64_t root_to,
                              MUTEX_STORAGE & mutexes) {
        auto & mutexes_to = mutexes[root_to];
        auto & mutexes_from = mutexes[root_from];
        mutexes_to.insert(mutexes_from.begin(), mutexes_from.end());

        for(const auto v : mutexes_from){
            auto & nlc = mutexes[v];

            // best way to change values in set...
            nlc.erase(root_from);
            nlc.insert(root_to);
        }
        mutexes_to.erase(root_from);
        mutexes_from.clear();
    }


    // compute mutex clustering for a graph with attrative and mutex edges
    template<class EDGE_ARRAY, class WEIGHT_ARRAY, class NODE_ARRAY>
    void compute_mws_clustering(const size_t number_of_labels,
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
            const double val_a = (a < num_edges) ? weights(a) : mutex_weights(a - num_edges);
            const double val_b = (b < num_edges) ? weights(b) : mutex_weights(b - num_edges);
            return val_a > val_b;
        });

        // data-structure storing mutex edges
        typedef boost::container::flat_set<uint64_t> SetType;
        typedef std::vector<SetType> MutexStorage;
        MutexStorage mutexes(number_of_labels);

        // iterate over all edges
        for(const size_t edge_id : indices) {

            // check whether this edge is mutex via the edge offset
            const bool is_mutex_edge = edge_id >= num_edges;

            // find the edge-id or mutex id and the connected nodes
            const size_t id = is_mutex_edge ? edge_id - num_edges : edge_id;
            const uint64_t u = is_mutex_edge ? mutex_uvs(id, 0) : uvs(id, 0);
            const uint64_t v = is_mutex_edge ? mutex_uvs(id, 1) : uvs(id, 1);

            // find the current representatives
            uint64_t ru = ufd.find_set(u);
            uint64_t rv = ufd.find_set(v);

            // if the nodes are already connected, do nothing
            if(ru == rv) {
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
                insert_mutex(ru, rv, mutexes);

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
        }

        // get node labeling into output
        for(size_t label = 0; label < number_of_labels; ++label) {
            node_labeling[label] = ufd.find_set(label);
        }

        util::export_consecutive_labels(ufd, number_of_labels, node_labeling);
    }

    // compute mutex segmentation via kruskal
    template<class WEIGHT_ARRAY, class NODE_ARRAY, class INDICATOR_ARRAY>
    void compute_mws_segmentation(const xt::xexpression<WEIGHT_ARRAY> & sorted_flat_indices_exp,
                                  const xt::xexpression<INDICATOR_ARRAY> & valid_edges_exp,
                                  const xt::xexpression<INDICATOR_ARRAY> & mutex_edges_exp,
                                  const std::vector<std::vector<int>> & offsets,
                                  const std::vector<int> & image_shape,
                                  xt::xexpression<NODE_ARRAY> & node_labeling_exp) {

        // casts
        const auto & sorted_flat_indices = sorted_flat_indices_exp.derived_cast();
        const auto & valid_edges = valid_edges_exp.derived_cast();
        const auto & mutex_edges = mutex_edges_exp.derived_cast();
        auto & node_labeling = node_labeling_exp.derived_cast();

        // determine number of nodes and attractive edges (considering all edges, not only valid ones)
        const size_t number_of_nodes = node_labeling.size();
        const size_t number_of_offsets = offsets.size();
        const size_t ndims = offsets[0].size();

        // Define helper vector to find pair (u,v) from edge_id:
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

        // New data structure for storing mutex edges:
        typedef boost::container::flat_set<uint64_t> SetType;
        typedef std::vector<SetType> MutexStorage;
        MutexStorage mutexes(number_of_nodes);

        // iterate over all edges
        for(const size_t edge_id : sorted_flat_indices) {

            if(!valid_edges(edge_id)){
                continue;
            }

            // check whether this edge is mutex
            const bool is_mutex_edge = mutex_edges(edge_id);

            // get nodes connected by edge of edge_id

            // const auto affCoord_ = xt::unravel_from_strides(edge_id, strides, layout);
            const uint64_t u = edge_id % number_of_nodes;
            const uint64_t v = u + offset_strides[edge_id / number_of_nodes];

            // find the current representatives
            uint64_t ru = ufd.find_set(u);
            uint64_t rv = ufd.find_set(v);

            // if the nodes are already connected, do nothing
            if(ru == rv) {
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
                insert_mutex(ru, rv, mutexes);

            } else {

                ufd.link(u, v);
                // check  if we have to swap the roots
                if(ufd.find_set(ru) == rv) {
                    std::swap(ru, rv);
                }
                // merge mutexes from rv -> ru
                merge_mutexes(rv, ru, mutexes);
            }
        }

        // get node labeling into output (with consecutive relabeling)
        util::export_consecutive_labels(ufd, number_of_nodes, node_labeling);
    }

    // helper function for mws prim implementation:
    // add all neighbors of given node to the priority queue
    template <class WEIGHT_ARRAY, class VALID_ARRAY, class UFD, class PRIORITY_QUEUE>
    inline void add_neighbours(const uint64_t & position,
                               const std::vector<int64_t> & offset_strides,
                               const size_t & number_of_nodes,
                               const WEIGHT_ARRAY & edge_weights,
                               const VALID_ARRAY & valid_edges,
                               UFD & ufd,
                               const xt::xtensor<bool, 1> & visited,
                               PRIORITY_QUEUE & pq){

        const uint64_t ru = ufd.find_set(position);
        for(int i = 0; i < offset_strides.size(); ++i){
            // go in positive offset direction
            const uint64_t edge_id = position + i * number_of_nodes;
            if (valid_edges(edge_id) && !visited(edge_id)){
                const uint64_t neighbour = position + offset_strides[i];
                const uint64_t rv = ufd.find_set(neighbour);
                if (ru != rv){
                    pq.push(std::make_tuple(edge_weights(edge_id), edge_id, position, neighbour));
                }
            }

            // check that position - offset_stride lies within 0 and number_of_nodes
            const bool within_bounds = (offset_strides[i] > 0 || position < number_of_nodes + offset_strides[i])
                                    && (offset_strides[i] < 0 || offset_strides[i] <= position);

            // go in negative offset direction
            if (within_bounds){
                const uint64_t neg_neighbour = position - offset_strides[i];
                const uint64_t neg_edge_id = neg_neighbour + i * number_of_nodes;
                if (valid_edges(neg_edge_id) && !visited(neg_edge_id)){
                    const uint64_t rv = ufd.find_set(neg_neighbour);
                    if (ru != rv){
                        pq.push(std::make_tuple(edge_weights(neg_edge_id), neg_edge_id, position, neg_neighbour));
                    }
                }
            }
        }
    }


    // compute mutex segmentation via prim's algorithm
    template<class WEIGHT_ARRAY, class NODE_ARRAY, class INDICATOR_ARRAY>
    void compute_mws_prim_segmentation(const xt::xexpression<WEIGHT_ARRAY> & edge_weight_exp,
                                       const xt::xexpression<INDICATOR_ARRAY> & valid_edges_exp,
                                       const std::vector<std::vector<int>> & offsets,
                                       const size_t number_of_attractive_channels,
                                       const std::vector<int> & image_shape,
                                       xt::xexpression<NODE_ARRAY> & node_labeling_exp) {
        // typedef
        typedef std::tuple<float, uint64_t, uint64_t, uint64_t> PQElement;
        auto pq_compare = [](PQElement left, PQElement right) {return std::get<0>(left) < std::get<0>(right);};
        typedef std::priority_queue<PQElement, std::vector<PQElement>,
                                    decltype(pq_compare)> EdgePriorityQueue;
        typedef boost::disjoint_sets<uint64_t*, uint64_t*> NodeUnionFind;

        // casts
        const auto & edge_weights = edge_weight_exp.derived_cast();
        const auto & valid_edges = valid_edges_exp.derived_cast();
        auto & node_labeling = node_labeling_exp.derived_cast();

        const size_t number_of_nodes = node_labeling.size();
        const size_t number_of_attractive_edges = number_of_nodes * number_of_attractive_channels;
        const size_t number_of_offsets = offsets.size();
        const size_t ndims = offsets[0].size();
        xt::xtensor<bool, 1> visited = xt::zeros<bool>({edge_weights.size()});

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
        NodeUnionFind node_ufd(&ranks[0], &parents[0]);
        for(uint64_t label = 0; label < number_of_nodes; ++label) {
            node_ufd.make_set(label);
        }

        // data-structure storing mutex edges
        typedef boost::container::flat_set<uint64_t> SetType;
        typedef std::vector<SetType> MutexStorage;
        MutexStorage mutexes(number_of_nodes);
        EdgePriorityQueue pq(pq_compare);

        // start prim from top left node
        add_neighbours(0,
                       offset_strides,
                       number_of_nodes,
                       edge_weights,
                       valid_edges,
                       node_ufd,
                       visited,
                       pq);

        // iterate over all edges
        while(!pq.empty()) {
            // extract next element from the queue
            const PQElement position_vector = pq.top();
            pq.pop();
            const uint64_t edge_id = std::get<1>(position_vector);
            const uint64_t u = std::get<2>(position_vector);
            const uint64_t v = std::get<3>(position_vector);

            if(visited(edge_id)) {
                continue;
            }
            visited(edge_id) = 1;

            // find the current representatives
            // and skip if roots are identical
            uint64_t ru = node_ufd.find_set(u);
            uint64_t rv = node_ufd.find_set(v);
            if(ru == rv) {
                continue;
            }

            // check whether this edge is mutex via the edge offset
            const bool is_mutex_edge = edge_id >= number_of_attractive_edges;

            // if we already have a mutex, we do not need to do anything
            // (if this is a regular edge, we do not link, if it is a mutex edge
            //  we do not need to insert the redundant mutex constraint)
            if(check_mutex(ru, rv, mutexes)) {
                continue;
            }

            if(is_mutex_edge) {
                insert_mutex(ru, rv, mutexes);
            } else {

                node_ufd.link(u, v);
                // check  if we have to swap the roots
                if(node_ufd.find_set(ru) == rv) {
                    std::swap(ru, rv);
                }
                merge_mutexes(rv, ru, mutexes);
            }
            // add the next node to pq
            add_neighbours(v,
                           offset_strides,
                           number_of_nodes,
                           edge_weights,
                           valid_edges,
                           node_ufd,
                           visited,
                           pq);
        }

        // get node labeling into output (with consecutive relabeling)
        util::export_consecutive_labels(node_ufd, number_of_nodes, node_labeling);
    }

}
}
