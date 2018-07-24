#pragma once
#include <boost/pending/disjoint_sets.hpp>
#include "xtensor/xtensor.hpp"


namespace affogato {
namespace segmentation {

    // the datastructure to hold the mutex edges for a single cluster and all clusters
    typedef std::unordered_set<uint32_t> MutexSet;
    typedef std::vector<MutexSet> MutexStorage;


    template<class UFD>
    inline bool check_mutex(const uint32_t u, const uint32_t rv,
                            UFD & ufd, const MutexStorage & mutexes) {
        // the mutex storages are symmetric, so we only need to check one of them
        const auto & mutex_u = mutexes[u];
        bool have_mutex = false;
        // we check for all representatives of mutex edges if
        // they are the same as the reperesentative of v
        for(const auto mu : mutex_u) {
            if(ufd.find_set(mu) == rv) {
                have_mutex = true;
                break;
            }
        }
        return have_mutex;
    }


    template<class UFD>
    inline void insert_mutex(const uint32_t u, const uint32_t v, const uint32_t rv,
                             UFD & ufd, MutexStorage & mutexes) {

        auto & mutex_u = mutexes[u];
        // if we don't have a mutex yet, insert it
        if(mutex_u.size() == 0) {
            mutex_u.insert(v);
        }

        // otherwise check if v is already in the mutexes
        // and filter the mutexes in the process
        else {

            bool have_mutex = false;
            std::unordered_set<uint32_t> mutex_representatives;

            // iterate over all current mutexes
            auto mutex_it = mutex_u.begin();
            while(mutex_it != mutex_u.end()) {
                const uint32_t rm = ufd.find_set(*mutex_it);

                // check if this mutex is already present in the list
                // if it is not, insert it, otherwise delete this mutex
                if(mutex_representatives.find(rm) == mutex_representatives.end()) {
                    mutex_representatives.insert(rm);
                    ++mutex_it;  // we don't erase, so we need to increase by hand
                } else {
                    mutex_it = mutex_u.erase(mutex_it);
                }

                // if we have not already found v as mutex, check for it
                if(!have_mutex) {
                    have_mutex = rv == rm;
                }
            }

            // insert the v mutex if it is not present
            if(!have_mutex) {
                // std::cout << "Inserting mutex " << u << " " << v << std::endl;
                mutex_u.insert(v);
            }
        }
    }


    template<class UFD>
    inline void merge_mutexes(const uint32_t u, const uint32_t v,
                              UFD & ufd, MutexStorage & mutexes) {
        auto & mutex_u = mutexes[u];
        auto & mutex_v = mutexes[v];

        // extract all representatives (which should be unique here)
        std::unordered_map<uint32_t, uint32_t> mutex_reps_u;
        for(const auto mu : mutex_u) {
            mutex_reps_u[ufd.find_set(mu)] = mu;
        }

        std::unordered_map<uint32_t, uint32_t> mutex_reps_v;
        for(const auto mv : mutex_v) {
            mutex_reps_v[ufd.find_set(mv)] = mv;
        }

        // merge u into v
        for(const auto mu : mutex_reps_u) {
            if(mutex_reps_v.find(mu.first) == mutex_reps_v.end()) {
                mutex_v.insert(mu.second);
            }
        }

        // merge v into u
        for(const auto mv : mutex_reps_v) {
            if(mutex_reps_u.find(mv.first) == mutex_reps_u.end()) {
                mutex_u.insert(mv.second);
            }
        }

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
                const uint32_t u = mutex_uvs(mutex_id, 0);
                const uint32_t v = mutex_uvs(mutex_id, 1);

                // find the current representatives
                const uint32_t ru = ufd.find_set(u);
                const uint32_t rv = ufd.find_set(v);

                // if the nodes are already connected, do nothing
                if(ru == rv) {
                    continue;
                }

                // otherwise, insert the mutex
                insert_mutex(u, v, rv, ufd, mutexes);
                insert_mutex(v, u, ru, ufd, mutexes);

            } else {

                // find the connected nodes
                const uint32_t u = uvs(edge_id, 0);
                const uint32_t v = uvs(edge_id, 1);

                // find the current representatives
                const uint32_t ru = ufd.find_set(u);
                const uint32_t rv = ufd.find_set(v);

                // if the nodes are already connected, do nothing
                if(ru == rv) {
                    continue;
                }

                // otherwise, check if we have an active constraint / mutex edge
                const bool have_mutex = check_mutex(u, rv, ufd, mutexes) || check_mutex(v, ru, ufd, mutexes);
                //const bool have_mutex = check_mutex_edge(u, v, mutexes);

                // only merge if we don't have a mutex
                if(!have_mutex) {
                    ufd.link(u, v);
                    merge_mutexes(u, v, ufd, mutexes);
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

            if (not valid_edges(edge_id)) continue;

            // check whether this edge is mutex via the edge offset
            const bool is_mutex = edge_id >= number_of_attractive_edges;

            // get nodes connected by edge of edge_id

            // const auto affCoord_ = xt::unravel_from_strides(edge_id, strides, layout);
            uint64_t u = edge_id % number_of_nodes;
            uint64_t v = u + offset_strides[edge_id / number_of_nodes];

            if(is_mutex) {

                // find the current representatives
                const uint64_t ru = ufd.find_set(u);
                const uint64_t rv = ufd.find_set(v);

                // if the nodes are already connected, do nothing
                if(ru == rv) {
                    continue;
                }

                // otherwise, insert the mutex
                insert_mutex(u, v, rv, ufd, mutexes);
                insert_mutex(v, u, ru, ufd, mutexes);

            } else {

                // find the current representatives
                const uint64_t ru = ufd.find_set(u);
                const uint64_t rv = ufd.find_set(v);

                // if the nodes are already connected, do nothing
                if(ru == rv) {
                    continue;
                }

                // otherwise, check if we have an active constraint / mutex edge
                const bool have_mutex = check_mutex(u, rv, ufd, mutexes) || check_mutex(v, ru, ufd, mutexes);
                //const bool have_mutex = check_mutex_edge(u, v, mutexes);

                // only merge if we don't have a mutex
                if(!have_mutex) {
                    ufd.link(u, v);
                    merge_mutexes(u, v, ufd, mutexes);
                }

            }
        }

        // get node labeling into output
        for(size_t label = 0; label < number_of_nodes; ++label) {
            node_labeling[label] = ufd.find_set(label);
        }
    }


}
}
