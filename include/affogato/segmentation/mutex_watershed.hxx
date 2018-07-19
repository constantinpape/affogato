#pragma once
#include <fstream>
#include <boost/pending/disjoint_sets.hpp>
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"


namespace affogato {
namespace segmentation {

    // the datastructure to hold the mutex edges for a single cluster and all clusters
    typedef std::unordered_set<uint64_t> MutexSet;
    typedef std::vector<MutexSet> MutexStorage;


    template<class UFD>
    inline bool check_mutex(const uint64_t u, const uint64_t rv,
                            UFD & ufd, const MutexStorage & mutexes, const bool verbose=false) {
        const auto & mutex_u = mutexes[u];
        bool have_mutex = false;

        if(verbose) {
            std::cout << "Checking mutex for node " << u << " against repr " << rv << std::endl;
        }

        // we check for all representatives of mutex edges if
        // they are the same as the reperesentative of v
        for(const auto mu : mutex_u) {
            if(verbose) {
                std::cout << mu << " " << ufd.find_set(mu) << std::endl;
            }
            if(ufd.find_set(mu) == rv) {
                have_mutex = true;
                break;
            }
        }
        return have_mutex;
    }

    template<class UFD>
    inline bool check_mutex2(const uint64_t u, const uint64_t v,
                             UFD & ufd, const MutexStorage & mutexes) {
        const auto & mutex_u = mutexes[u];
        const auto & mutex_v = mutexes[v];
        std::vector<uint64_t> reps_u;
        reps_u.reserve(mutex_u.size());
        std::vector<uint64_t> reps_v;
        reps_v.reserve(mutex_v.size());

        for(const auto mu: mutex_u) {
            reps_u.push_back(ufd.find_set(mu));
        }

        for(const auto mv: mutex_v) {
            reps_v.push_back(ufd.find_set(mv));
        }

        std::vector<uint64_t> intersect;
        std::set_intersection(reps_u.begin(), reps_u.end(),
                              reps_v.begin(), reps_v.end(),
                              std::back_inserter(intersect));
        return intersect.size() > 0;

    }


    // insert the node v into the mutexes of u, if we do not have a mutex node in u
    // with the same representative (rv) already
    template<class UFD>
    inline void insert_mutex(const uint64_t u, const uint64_t v, const uint64_t rv,
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
            std::unordered_set<uint64_t> mutex_representatives;

            // iterate over all current mutexes
            auto mutex_it = mutex_u.begin();
            while(mutex_it != mutex_u.end()) {
                const uint64_t rm = ufd.find_set(*mutex_it);

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
    inline void merge_mutexes(const uint64_t u, const uint64_t v,
                              UFD & ufd, MutexStorage & mutexes) {
        auto & mutex_u = mutexes[u];
        auto & mutex_v = mutexes[v];

        // extract all representatives (which should be unique here)
        std::unordered_map<uint64_t, uint64_t> mutex_reps_u;
        for(const auto mu : mutex_u) {
            mutex_reps_u[ufd.find_set(mu)] = mu;
        }

        std::unordered_map<uint64_t, uint64_t> mutex_reps_v;
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
                insert_mutex(u, v, rv, ufd, mutexes);
                insert_mutex(v, u, ru, ufd, mutexes);

            } else {

                // find the connected nodes
                const uint64_t u = uvs(edge_id, 0);
                const uint64_t v = uvs(edge_id, 1);

                // find the current representatives
                const uint64_t ru = ufd.find_set(u);
                const uint64_t rv = ufd.find_set(v);

                // if the nodes are already connected, do nothing
                if(ru == rv) {
                    continue;
                }

                // otherwise, check if we have an active constraint / mutex edge
                const bool have_mutex = check_mutex(u, rv, ufd, mutexes) || check_mutex(v, ru, ufd, mutexes);

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


    void fast_2d_set_valid_edges(xt::xarray<bool> & valid_edges,
                                 const std::vector<int> & image_shape,
                                 const std::vector<std::vector<int>> & offsets){
        xt::view(valid_edges, xt::range(0, image_shape[0]-1, 1L),
                              xt::range(0, image_shape[1]-1, 1L),
                              xt::all()) = 1;

        const size_t num_attractive_channels = 2;
        xt::view(valid_edges, xt::all(),
                              xt::all(),
                              xt::range(0, num_attractive_channels, int64_t(1))) = 1;

        for (uint64_t d = 0; d < offsets.size(); ++d) {
            if (offsets[d][0] > 0)
                xt::view(valid_edges, xt::range(image_shape[0]-1, image_shape[0]-offsets[d][0]-1, int64_t(-1)), xt::all(), d) = 0;
            else if (offsets[d][0] < 0)
                xt::view(valid_edges, xt::range(0., -offsets[d][0], 1), xt::all(), d) = 0;
            if (offsets[d][1] > 0)
                xt::view(valid_edges, xt::all(), xt::range(image_shape[1]-1, image_shape[1]-offsets[d][1]-1, int64_t(-1)), d) = 0;
            else if (offsets[d][1] < 0)
                xt::view(valid_edges, xt::all(), xt::range(0., -offsets[d][1], 1), d) = 0;
        }
    }


    /*
    void fast_3d_set_valid_edges(){

        xt::view(valid_edges, xt::range(0, image_shape(0)-1, 1L),
                              xt::range(0, image_shape(1)-1, 1L),
                              xt::range(0, image_shape(2)-1, 1L),
                              xt::all()) = 1;

        xt::view(valid_edges, xt::all(),
                         xt::all(),
                         xt::all(),
                         xt::range(0, num_attractive_channels, int64_t(1))) = 1;

        for (uint64_t d = 0; d < directions; ++d) {
            if (offsets(d, 0) > 0)
                xt::view(valid_edges, xt::range(image_shape(0)-1, image_shape(0)-offsets(d, 0)-1, int64_t(-1)), xt::all(), xt::all(), d) = 0;
            else if (offsets(d, 0) < 0)
                xt::view(valid_edges, xt::range(0., -offsets(d, 0), 1.), xt::all(), xt::all(), d) = 0;
            if (offsets(d, 1) > 0)
                xt::view(valid_edges, xt::all(), xt::range(image_shape(1)-1, image_shape(1)-offsets(d, 1)-1, int64_t(-1)), xt::all(), d) = 0;
            else if (offsets(d, 1) < 0)
                xt::view(valid_edges, xt::all(), xt::range(0., -offsets(d, 1), 1.), xt::all(), d) = 0;
            if (offsets(d, 2) > 0)
                xt::view(valid_edges, xt::all(), xt::all(), xt::range(image_shape(2)-1, image_shape(2)-offsets(d, 2)-1, int64_t(-1)), d) = 0;
            else if (offsets(d, 2) < 0)
                xt::view(valid_edges, xt::all(), xt::all(), xt::range(0., -offsets(d, 2), 1.), d) = 0;
        }
    }
    */


    // template<class WEIGHT_ARRAY, class NODE_ARRAY, class INDICATOR_ARRAY>
    template<class WEIGHT_ARRAY, class NODE_ARRAY>
    void compute_mws_segmentation(const size_t number_of_attractive_channels,
                                const std::vector<std::vector<int>> & offsets,
                                const std::vector<int> & image_shape,
                                const xt::xexpression<WEIGHT_ARRAY> & sorted_flat_indices_exp,
                                // const xt::xexpression<INDICATOR_ARRAY> & valid_edges_exp,
                                xt::xexpression<NODE_ARRAY> & node_labeling_exp) {

        // casts
        const auto & sorted_flat_indices = sorted_flat_indices_exp.derived_cast();
        // const auto & valid_edges = valid_edges_exp.derived_cast();
        auto & node_labeling = node_labeling_exp.derived_cast();

        const size_t number_of_nodes = node_labeling.size();
        const size_t number_of_attractive_edges = number_of_nodes * number_of_attractive_channels;
        const size_t number_of_offsets = offsets.size();
        const size_t ndims = offsets[0].size();

        typedef typename xt::xarray<bool>::shape_type Shape;
        Shape wshape = {image_shape[0], image_shape[1], offsets.size()};
        xt::xarray<bool> valid_edges = xt::zeros<bool>(wshape);
        fast_2d_set_valid_edges(valid_edges, image_shape, offsets);
        valid_edges.reshape({number_of_nodes, offsets.size()});

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
        for(uint64_t node = 0; node < number_of_nodes; ++node) {
            ufd.make_set(node);
        }

        MutexStorage mutexes(number_of_nodes);

        std::ofstream out("mutex_actions.txt", std::ofstream::out);

        // iterate over all edges
        for(const size_t edge_id : sorted_flat_indices) {

            out << "Edge: " << edge_id << std::endl;

            if(!valid_edges(edge_id % number_of_nodes, edge_id / number_of_nodes)){
                out << "is invalid" << std::endl;
                out << std::endl;
                continue;
            }

            // get nodes connected by edge of edge_id
            const uint64_t u = edge_id % number_of_nodes;
            const uint64_t v = u + offset_strides[edge_id / number_of_nodes];
            out << "connects " << u << " " << v << std::endl;

            // find the current representatives
            const uint64_t ru = ufd.find_set(u);
            const uint64_t rv = ufd.find_set(v);

            // if the nodes are already connected, do nothing
            if(ru == rv) {
                out << "already linked" << std::endl;
                out << std::endl;
                continue;
            }

            // check whether this edge is mutex via the edge offset
            const bool is_mutex = edge_id >= number_of_attractive_edges;

            if(is_mutex) {

                out << "is mutex" << std::endl;
                // otherwise, insert the mutex
                insert_mutex(u, v, rv, ufd, mutexes);
                insert_mutex(v, u, ru, ufd, mutexes);

            } else {

                // otherwise, check if we have an active constraint / mutex edge
                const bool verbose = (u == 7748) && (v == 7747);
                bool have_mutex = check_mutex(u, rv, ufd, mutexes, verbose) || check_mutex(v, ru, ufd, mutexes, verbose);
                // have_mutex = have_mutex || check_mutex2(u, v, ufd, mutexes);

                // only merge if we don't have a mutex
                if(!have_mutex) {
                    out << "have no mutex, merge" << std::endl;
                    ufd.link(u, v);
                    merge_mutexes(u, v, ufd, mutexes);
                } else {
                    out << "have mutex, no merge" << std::endl;
                }
            }
            out << std::endl;
        }

        // get node labeling into output
        for(size_t node = 0; node < number_of_nodes; ++node) {
            node_labeling[node] = ufd.find_set(node);
        }
    }


}
}
