#pragma once
#include "affogato/segmentation/mutex_watershed.hxx"

namespace affogato {
namespace learning {

    template<class WEIGHT_ARRAY, class INDEX_ARRAY, class INDICATOR_ARRAY, class LABEL_ARRAY, class GRADIENT_ARRAY>
    double compute_mutex_malis_gradient(const xt::xexpression<WEIGHT_ARRAY> & flat_weights_exp,
                                        const xt::xexpression<INDEX_ARRAY> & sorted_flat_indices_exp,
                                        const xt::xexpression<INDICATOR_ARRAY> & valid_edges_exp,
                                        const xt::xexpression<LABEL_ARRAY> & gt_labels_flat_exp,
                                        const std::vector<std::vector<int>> & offsets,
                                        const size_t number_of_attractive_channels,
                                        const std::vector<int> & image_shape,
                                        xt::xexpression<GRADIENT_ARRAY> & gradient_exp,
                                        xt::xexpression<LABEL_ARRAY> & labels_pos_exp,
                                        xt::xexpression<LABEL_ARRAY> & labels_neg_exp) {

        typedef typename GRADIENT_ARRAY::value_type GradType;

        // casts
        const auto & flat_weights = flat_weights_exp.derived_cast();
        const auto & sorted_flat_indices = sorted_flat_indices_exp.derived_cast();
        const auto & valid_edges = valid_edges_exp.derived_cast();
        const auto & gt_labels = gt_labels_flat_exp.derived_cast();
        auto & grads = gradient_exp.derived_cast();
        auto & labels_pos = labels_pos_exp.derived_cast();
        auto & labels_neg = labels_neg_exp.derived_cast();

        // determine number of nodes and attractive edges
        const size_t number_of_nodes = gt_labels.size();
        const size_t number_of_attractive_edges = number_of_nodes * number_of_attractive_channels;
        const size_t number_of_offsets = offsets.size();
        const size_t ndims = offsets[0].size();

        // determine the strides of the image
        std::vector<int64_t> array_stride(ndims);
        int64_t current_stride = 1;
        for (int i = ndims-1; i >= 0; --i){
            array_stride[i] = current_stride;
            current_stride *= image_shape[i];
        }

        // determine the strides of the offsets
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

        // data-structure storing mutex edges
        typedef std::vector<std::vector<uint64_t>> MutexStorage;
        MutexStorage mutexes(number_of_nodes);

        // additional malis datastructures
        // data structures for overlaps of nodes (= pixels) with gt labels
        // and sizes of gt segments
        std::vector<std::map<uint64_t, size_t>> overlaps(number_of_nodes);
        std::unordered_map<uint64_t, size_t> segment_sizes;

        // initialize sets, overlaps and find labeled pixels
        size_t number_of_labeled_nodes = 0, number_of_positive_pairs = 0;
        for(size_t node_index = 0; node_index < number_of_nodes; ++node_index) {
            const auto gt_id = gt_labels(node_index);
            if(gt_id != 0) {
                overlaps[node_index].insert(std::make_pair(gt_id, 1));
                ++segment_sizes[gt_id];
                ++number_of_labeled_nodes;
                number_of_positive_pairs += (segment_sizes[gt_id] - 1);
            }
            ufd.make_set(node_index);
        }

        // iterate over all edges
        double loss = 0;
        std::vector<uint64_t> neg_q;
        std::vector<std::pair<uint64_t, uint64_t>> merge_q;

        // positive pass
        for(const size_t edge_id : sorted_flat_indices) {

            if(!valid_edges(edge_id)){
                continue;
            }

            // check whether this edge is mutex via the edge offset
            const bool is_mutex = edge_id >= number_of_attractive_edges;

            // get nodes connected by edge of edge_id
            const uint64_t u = edge_id % number_of_nodes;
            const uint64_t v = u + offset_strides[edge_id / number_of_nodes];

            const auto lu = gt_labels[u];
            const auto lv = gt_labels[v];

            if(lu != lv) {
                if(lu != 0 || lv != 0) {
                    neg_q.push_back(edge_id);
                }
                continue;
            }

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
            if(segmentation::check_mutex(ru, rv, mutexes)) {
                continue;
            }

            if(is_mutex) {

                merge_q.push_back(std::make_pair(u, v));

                // insert the mutex edge into both mutex edge storages
                segmentation::insert_mutex(ru, rv, edge_id, mutexes);

                // compute gradient for the mutex edge
                // compute gradients for the merge
                GradType current_gradient = 0;
                const auto w = flat_weights[edge_id];
                // TODO sign of gradient ?!
                // const GradType grad = pos ? -w : 1 - w;
                const GradType grad = +w;

                // TODO double check the gradients
                // compute the number of node pairs that would be merged by this edge
                for(auto it_u = overlaps[ru].begin(); it_u != overlaps[ru].end(); ++it_u) {
                    for(auto it_v = overlaps[rv].begin(); it_v != overlaps[rv].end(); ++it_v) {
                        const size_t n_pair = it_u->second * it_v->second;
                        if(it_u->first == it_v->first) {
                            loss += grad * grad * n_pair;
                            current_gradient += grad * n_pair;
                        }
                    }
                }
                grads[edge_id] += current_gradient;

            } else {

                // otherwise merge and compute gradients
                ufd.link(u, v);
                // check  if we have to swap the roots
                if(ufd.find_set(ru) == rv) {
                    std::swap(ru, rv);
                }
                // merge mutexes from rv -> ru
                segmentation::merge_mutexes(rv, ru, mutexes);

                // compute gradients for the merge
                GradType current_gradient = 0;
                const auto w = flat_weights[edge_id];
                // const GradType grad = pos ? 1. - w : -w;
                const GradType grad = w - 1.;

                // compute the number of node pairs merged by this edge
                for(auto it_u = overlaps[ru].begin(); it_u != overlaps[ru].end(); ++it_u) {
                    for(auto it_v = overlaps[rv].begin(); it_v != overlaps[rv].end(); ++it_v) {
                        const size_t n_pair = it_u->second * it_v->second;
                        if(it_u->first == it_v->first) {
                            loss += grad * grad * n_pair;
                            current_gradient += grad * n_pair;
                        }
                    }
                }
                grads[edge_id] += current_gradient;

                auto & overlaps_u = overlaps[ru];
                auto & overlaps_v = overlaps[rv];
                auto it_v = overlaps_v.begin();
                while(it_v != overlaps_v.end()) {
                    auto it_u = overlaps_u.find(it_v->first);
                    if(it_u == overlaps_u.end()) {
                        overlaps_u.insert(std::make_pair(it_v->first, it_v->second));
                    } else {
                        it_u->second += it_v->second;
                    }
                    overlaps_v.erase(it_v);
                    ++it_v;
                }
            }
        }

        // get pos node labeling
        for(size_t label = 0; label < number_of_nodes; ++label) {
            labels_pos[label] = ufd.find_set(label);
        }

        // apply merge q
        for(const auto & merge_pair: merge_q) {
            ufd.link(merge_pair.first, merge_pair.second);
        }

        // clear mutex store
        // mutexes.clear();

        // negative pass
        for(const size_t edge_id : neg_q) {

            if(!valid_edges(edge_id)){
                continue;
            }

            // check whether this edge is mutex via the edge offset
            const bool is_mutex = edge_id >= number_of_attractive_edges;

            // get nodes connected by edge of edge_id
            const uint64_t u = edge_id % number_of_nodes;
            const uint64_t v = u + offset_strides[edge_id / number_of_nodes];

            const auto lu = gt_labels[u];
            const auto lv = gt_labels[v];

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
            if(segmentation::check_mutex(ru, rv, mutexes)) {
                continue;
            }

            if(is_mutex) {

                merge_q.push_back(std::make_pair(u, v));

                // insert the mutex edge into both mutex edge storages
                segmentation::insert_mutex(ru, rv, edge_id, mutexes);

                // compute gradient for the mutex edge
                // compute gradients for the merge
                GradType current_gradient = 0;
                const auto w = flat_weights[edge_id];
                // TODO sign of gradient ?!
                const GradType grad = w - 1.;

                // TODO double check the gradients
                // compute the number of node pairs that would be merged by this edge
                for(auto it_u = overlaps[ru].begin(); it_u != overlaps[ru].end(); ++it_u) {
                    for(auto it_v = overlaps[rv].begin(); it_v != overlaps[rv].end(); ++it_v) {
                        const size_t n_pair = it_u->second * it_v->second;
                        if(it_u->first != it_v->first) {
                            loss += grad * grad * n_pair;
                            current_gradient += grad * n_pair;
                        }
                    }
                }
                grads[edge_id] += current_gradient;

            } else {

                // otherwise merge and compute gradients
                ufd.link(u, v);
                // check  if we have to swap the roots
                if(ufd.find_set(ru) == rv) {
                    std::swap(ru, rv);
                }
                // merge mutexes from rv -> ru
                segmentation::merge_mutexes(rv, ru, mutexes);

                // compute gradients for the merge
                GradType current_gradient = 0;
                const auto w = flat_weights[edge_id];
                const GradType grad = +w;

                // compute the number of node pairs merged by this edge
                for(auto it_u = overlaps[ru].begin(); it_u != overlaps[ru].end(); ++it_u) {
                    for(auto it_v = overlaps[rv].begin(); it_v != overlaps[rv].end(); ++it_v) {
                        const size_t n_pair = it_u->second * it_v->second;
                        if(it_u->first != it_v->first) {
                            loss += grad * grad * n_pair;
                            current_gradient += grad * n_pair;
                        }
                    }
                }
                grads[edge_id] += current_gradient;

                auto & overlaps_u = overlaps[ru];
                auto & overlaps_v = overlaps[rv];
                auto it_v = overlaps_v.begin();
                while(it_v != overlaps_v.end()) {
                    auto it_u = overlaps_u.find(it_v->first);
                    if(it_u == overlaps_u.end()) {
                        overlaps_u.insert(std::make_pair(it_v->first, it_v->second));
                    } else {
                        it_u->second += it_v->second;
                    }
                    overlaps_v.erase(it_v);
                    ++it_v;
                }
            }
        }
        // get neg node labeling
        for(size_t label = 0; label < number_of_nodes; ++label) {
            labels_neg[label] = ufd.find_set(label);
        }

        return loss;
    }


    /*
    template<class WEIGHT_ARRAY, class INDEX_ARRAY, class INDICATOR_ARRAY, class LABEL_ARRAY, class GRADIENT_ARRAY>
    double constrained_mutex_malis(const xt::xexpression<WEIGHT_ARRAY> & flat_weights_exp,
                                   const xt::xexpression<INDEX_ARRAY> & sorted_flat_indices_exp,
                                   const xt::xexpression<INDICATOR_ARRAY> & valid_edges_exp,
                                   const xt::xexpression<LABEL_ARRAY> & gt_labels_flat_exp,
                                   const std::vector<std::vector<int>> & offsets,
                                   const size_t number_of_attractive_channels,
                                   const std::vector<int> & image_shape,
                                   xt::xexpression<GRADIENT_ARRAY> & gradient_exp) {

        const auto & flat_weights = flat_weights_exp.derived_cast();
        const auto & sorted_flat_indices = sorted_flat_indices_exp.derived_cast();
        const auto & valid_edges = valid_edges_exp.derived_cast();
        auto & gradients = gradient_exp.derived_cast();

        // run positive and negative pass
        const double loss_pos = compute_mutex_malis_gradient(flat_weights, sorted_flat_indices,
                                                             valid_edges, gt_labels_flat_exp,
                                                             offsets, number_of_attractive_channels,
                                                             image_shape, true, gradients);
        const double loss_neg = compute_mutex_malis_gradient(flat_weights, sorted_flat_indices,
                                                             valid_edges, gt_labels_flat_exp,
                                                             offsets, number_of_attractive_channels,
                                                             image_shape, false, gradients);

        // normalize and invert the gradients
        gradients /= -2.;
        return - (loss_pos + loss_neg) / 2.;
    }
    */


    template<class WEIGHT_ARRAY, class INDEX_ARRAY, class INDICATOR_ARRAY, class LABEL_ARRAY, class GRADIENT_ARRAY>
    double constrained_mutex_malis_debug(const xt::xexpression<WEIGHT_ARRAY> & flat_weights_exp,
                                         const xt::xexpression<INDEX_ARRAY> & sorted_flat_indices_exp,
                                         const xt::xexpression<INDICATOR_ARRAY> & valid_edges_exp,
                                         const xt::xexpression<LABEL_ARRAY> & gt_labels_flat_exp,
                                         const std::vector<std::vector<int>> & offsets,
                                         const size_t number_of_attractive_channels,
                                         const std::vector<int> & image_shape,
                                         xt::xexpression<GRADIENT_ARRAY> & gradient_exp,
                                         xt::xexpression<LABEL_ARRAY> & labels_pos_exp,
                                         xt::xexpression<LABEL_ARRAY> & labels_neg_exp) {

        const auto & flat_weights = flat_weights_exp.derived_cast();
        const auto & sorted_flat_indices = sorted_flat_indices_exp.derived_cast();
        const auto & valid_edges = valid_edges_exp.derived_cast();
        auto & gradients = gradient_exp.derived_cast();

        // run positive and negative pass
        const double loss = compute_mutex_malis_gradient(flat_weights, sorted_flat_indices,
                                                             valid_edges, gt_labels_flat_exp,
                                                             offsets, number_of_attractive_channels,
                                                             image_shape, gradients,
                                                             labels_pos_exp, labels_neg_exp);

        return loss;
    }

}
}
