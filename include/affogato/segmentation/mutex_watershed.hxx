#pragma once
#include <boost/pending/disjoint_sets.hpp>
#include "xtensor/xtensor.hpp"
#include <queue>
#include <functional>

namespace affogato {
namespace segmentation {

    // the datastructure to hold the mutex edges for a single cluster and all clusters
    // typedef std::unordered_set<uint64_t> MutexSet;
    // typedef std::vector<MutexSet> MutexStorage;
    typedef std::tuple<float, uint64_t, uint64_t, uint64_t> PQElement;
    auto pq_compare = [](PQElement left, PQElement right) { return std::get<0>(left) < std::get<0>(right);};
    // typedef std::priority_queue<> EdgePriorityQueue;
    typedef std::priority_queue<PQElement, std::vector<PQElement>, decltype(pq_compare)> EdgePriorityQueue;
    typedef boost::disjoint_sets<uint64_t*, uint64_t*> NodeUnionFind;

    // template<class UFD>
    // inline bool check_mutex(const uint32_t u, const uint32_t rv,
    //                         UFD & ufd, const MutexStorage & mutexes) {
    //     // the mutex storages are symmetric, so we only need to check one of them
    //     const auto & mutex_u = mutexes[u];
    //     bool have_mutex = false;
    //     // we check for all representatives of mutex edges if
    //     // they are the same as the reperesentative of v
    //     for(const auto mu : mutex_u) {
    //         if(ufd.find_set(mu) == rv) {
    //             have_mutex = true;
    //             break;


    typedef std::vector<std::vector<uint64_t>> MutexStorage;

    inline bool check_mutex(const uint64_t ru, const uint64_t rv,
                            const MutexStorage & mutexes) {
        // get iterators to the mutex vectors of rep u and rep v
        auto mutex_it_u = mutexes[ru].begin();
        auto mutex_it_v = mutexes[rv].begin();

        // check if the mutex vectors contain the same mutex edge
        while (mutex_it_u != mutexes[ru].end() && mutex_it_v != mutexes[rv].end()) {
            if (*mutex_it_u < *mutex_it_v) {
                ++mutex_it_u;
            } else  {
                if (!(*mutex_it_v < *mutex_it_u)) {
                    return true;
                }
                ++mutex_it_v;
            }
        }
        return false;
    }


    // insert 'mutex_edge_id' into the vectors containing mutexes of 'ru' and 'rv'
    inline void insert_mutex(const uint64_t ru, const uint64_t rv,
                             const uint64_t mutex_edge_id, MutexStorage & mutexes) {
        // in order to keep the mutex vectors handy, we only insert the mutex edge,
        // if the two sets don't share a mutex yet
        if (!check_mutex(ru, rv, mutexes)){
            mutexes[ru].insert(std::upper_bound(mutexes[ru].begin(), mutexes[ru].end(), mutex_edge_id), mutex_edge_id);
            mutexes[rv].insert(std::upper_bound(mutexes[rv].begin(), mutexes[rv].end(), mutex_edge_id), mutex_edge_id);
        }
    }


    // merge the mutex edges by merging from 'root_from' to 'root_to'
    inline void merge_mutexes(const uint64_t root_from, const uint64_t root_to, MutexStorage & mutexes) {
        if (mutexes[root_from].size() == 0) {
            return;
        }

        if (mutexes[root_to].size() == 0){
            mutexes[root_to] = mutexes[root_from];
            return;
        }

        std::vector<uint64_t> merge_buffer;
        merge_buffer.reserve(std::max(mutexes[root_from].size(), mutexes[root_to].size()));

        std::merge(mutexes[root_from].begin(), mutexes[root_from].end(),
                   mutexes[root_to].begin(), mutexes[root_to].end(),
                   std::back_inserter(merge_buffer));

        mutexes[root_to] = merge_buffer;
        mutexes[root_from].clear();
    }


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
            return val_a < val_b;
        });

        MutexStorage mutexes(number_of_labels);

        // iterate over all edges
        for(const size_t edge_id : indices) {

            // check whether this edge is mutex via the edge offset
            const bool is_mutex = edge_id >= num_edges;

            if(is_mutex) {
                // find the mutex id and the connected nodes
                const size_t mutex_id = edge_id - num_edges;
                const uint64_t u = mutex_uvs(mutex_id, 0);
                const uint64_t v = mutex_uvs(mutex_id, 1);

                // find the current representatives
                const uint64_t ru = ufd.find_set(u);
                const uint64_t rv = ufd.find_set(v);

                // if the nodes are already connected, do nothing
                if(ru == rv) {
                    continue;
                }

                // otherwise, insert the mutex
                insert_mutex(ru, rv, mutex_id, mutexes);

            } else {

                // find the connected nodes
                const uint64_t u = uvs(edge_id, 0);
                const uint64_t v = uvs(edge_id, 1);

                // find the current representatives
                uint64_t ru = ufd.find_set(u);
                uint64_t rv = ufd.find_set(v);

                // if the nodes are already connected, do nothing
                if(ru == rv) {
                    continue;
                }

                // otherwise, check if we have an active constraint / mutex edge
                const bool have_mutex = check_mutex(ru, rv, mutexes);

                // only merge if we don't have a mutex
                if(!have_mutex) {
                    ufd.link(u, v);
                    // check  if we have to swap the roots
                    if(ufd.find_set(ru) == rv) {
                        std::swap(ru, rv);
                    }
                    // merge mutexes from rv -> ru
                    merge_mutexes(rv, ru, mutexes);
                }

            }
        }

        // get node labeling into output
        for(size_t label = 0; label < number_of_labels; ++label) {
            node_labeling[label] = ufd.find_set(label);
        }
    }


    template<class WEIGHT_ARRAY, class NODE_ARRAY, class INDICATOR_ARRAY>
    void compute_mws_segmentation(const size_t number_of_attractive_channels,
                                const std::vector<std::vector<int>> & offsets,
                                const std::vector<int> & image_shape,
                                const xt::xexpression<WEIGHT_ARRAY> & sorted_flat_indices_exp,
                                const xt::xexpression<INDICATOR_ARRAY> & valid_edges_exp,
                                xt::xexpression<NODE_ARRAY> & node_labeling_exp) {

        // casts
        const auto & sorted_flat_indices = sorted_flat_indices_exp.derived_cast();
        const auto & valid_edges = valid_edges_exp.derived_cast();
        auto & node_labeling = node_labeling_exp.derived_cast();

        // determine number of nodes and attractive edges
        const size_t number_of_nodes = node_labeling.size();
        const size_t number_of_attractive_edges = number_of_nodes * number_of_attractive_channels;
        const size_t number_of_offsets = offsets.size();
        const size_t ndims = offsets[0].size();

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

        MutexStorage mutexes(number_of_nodes);

        // iterate over all edges
        for(const size_t edge_id : sorted_flat_indices) {

            if(!valid_edges(edge_id)){
                continue;
            }

            // check whether this edge is mutex via the edge offset
            const bool is_mutex = edge_id >= number_of_attractive_edges;

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

            if(is_mutex) {

                // insert the mutex edge into both mutex edge storages
                insert_mutex(ru, rv, edge_id, mutexes);

            } else {

                // otherwise, check if we have an active constraint / mutex edge
                const bool have_mutex = check_mutex(ru, rv, mutexes);

                // only merge if we don't have a mutex
                if(!have_mutex) {
                    ufd.link(u, v);
                    // check  if we have to swap the roots
                    if(ufd.find_set(ru) == rv) {
                        std::swap(ru, rv);
                    }
                    // merge mutexes from rv -> ru
                    merge_mutexes(rv, ru, mutexes);
                }
            }
        }

        // get node labeling into output
        for(size_t label = 0; label < number_of_nodes; ++label) {
            node_labeling[label] = ufd.find_set(label);
        }
    }

    template <class WEIGHT_ARRAY, class VALID_ARRAY>
    inline void add_neighbours(const uint64_t & position,
                               const std::vector<int64_t> & offset_strides, 
                               const size_t & number_of_nodes,
                               const WEIGHT_ARRAY & edge_weights,
                               const VALID_ARRAY & valid_edges,
                               NodeUnionFind & ufd,
                               xt::pytensor<bool, 1> & visited,
                               EdgePriorityQueue & pq){


        const uint64_t ru = ufd.find_set(position);
        for(int i = 0; i < offset_strides.size(); ++i){
            // go in positive offset direction
            const uint64_t edge_id = position + i * number_of_nodes;
            if (valid_edges(edge_id) and !visited(edge_id)){
                const uint64_t neighbour = position + offset_strides[i];
                const uint64_t rv = ufd.find_set(neighbour);
                if (ru != rv){
                    pq.push(std::make_tuple(edge_weights(edge_id), edge_id, position, neighbour));
                    // visited(edge_id) = 1;
                }
            }

            // go in negative offset direction
            if (offset_strides[i] >= position){
                const uint64_t neg_neighbour = position - offset_strides[i];
                if (neg_neighbour < number_of_nodes){
                    const uint64_t neg_edge_id = neg_neighbour + i * number_of_nodes;
                    if (valid_edges(neg_edge_id) and !visited(edge_id)){
                        const uint64_t rv = ufd.find_set(neg_neighbour);
                        if (ru != rv){
                            pq.push(std::make_tuple(edge_weights(neg_edge_id), neg_edge_id, position, neg_neighbour));
                            // visited(neg_edge_id) = 1;
                        }
                    }
                }
            }
        }
    }

    template<class WEIGHT_ARRAY, class NODE_ARRAY, class INDICATOR_ARRAY>
    void compute_mws_prim_segmentation(const size_t number_of_attractive_channels,
                                const std::vector<std::vector<int>> & offsets,
                                const std::vector<int> & image_shape,
                                const xt::xexpression<WEIGHT_ARRAY> & edge_weight_exp,
                                const xt::xexpression<INDICATOR_ARRAY> & valid_edges_exp,
                                xt::xexpression<NODE_ARRAY> & node_labeling_exp) {

        // casts
        const auto & edge_weights = edge_weight_exp.derived_cast();
        const auto & valid_edges = valid_edges_exp.derived_cast();
        auto & node_labeling = node_labeling_exp.derived_cast();


        const size_t number_of_nodes = node_labeling.size();
        const size_t number_of_attractive_edges = number_of_nodes * number_of_attractive_channels;
        const size_t number_of_offsets = offsets.size();
        const size_t ndims = offsets[0].size();
        xt::pytensor<bool, 1> visited = xt::zeros<bool>({edge_weights.size()});

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
            const bool is_mutex = edge_id >= number_of_attractive_edges;

            if(is_mutex) {
                insert_mutex(ru, rv, edge_id, mutexes);
            } else {
                // otherwise, check if we have an active constraint / mutex edge
                const bool have_mutex = check_mutex(ru, rv, mutexes);

                //const bool have_mutex = check_mutex_edge(u, v, mutexes);
                // only merge if we don't have a mutex
                if(!have_mutex) {
                    node_ufd.link(u, v);
                    // check  if we have to swap the roots
                    if(node_ufd.find_set(ru) == rv) {
                        std::swap(ru, rv);
                    }
                    merge_mutexes(rv, ru, mutexes);
                }
            }
            add_neighbours(v,
                       offset_strides, 
                       number_of_nodes,
                       edge_weights,
                       valid_edges,
                       node_ufd,
                       visited,
                       pq);
        }

        // get node labeling into output
        for(size_t label = 0; label < number_of_nodes; ++label) {
            node_labeling[label] = node_ufd.find_set(label);
        }
    }



}
}
