#pragma once
#include "affogato/segmentation/mutex_watershed.hxx"

namespace affogato {
namespace learning {

    typedef typename std::vector<std::map<uint64_t, size_t>> OverlapType;

    template<class GRADTYPE, class GRADOUTTYPE>
    inline void compute_gradient(const uint64_t & ru,
                                 const uint64_t & rv,
                                 const OverlapType & overlaps,
                                 const bool invert,
                                 const bool same_region,
                                 double & loss,
                                 const GRADTYPE & edge_grad,
                                 GRADOUTTYPE & out_grad) {
        
        GRADOUTTYPE current_gradient = 0;
        GRADTYPE grad = edge_grad;
        // invert the gradient sign
        if (invert){
            grad = grad - 1.;
        }

        // early exit for more efficiency
        if (grad == 0.){
            return;
        }

        for(auto it_u = overlaps[ru].begin(); it_u != overlaps[ru].end(); ++it_u) {
            for(auto it_v = overlaps[rv].begin(); it_v != overlaps[rv].end(); ++it_v) {
                // if separate is true the error is proportional to 
                // all pairs with the same id
                // if separate is false we instead count all pairs with different ids
                if((it_u->first == it_v->first) == same_region) {
                    const size_t n_pair = it_u->second * it_v->second;
                    loss += grad * grad * 1.;//n_pair;
                    current_gradient += grad * n_pair;
                }
            }
        }

        out_grad += current_gradient;
    }

    inline void add_overlaps(const uint64_t & ru,
                             const uint64_t & rv,
                             OverlapType & overlaps) {
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

    template<class WEIGHT_ARRAY, class INDEX_ARRAY, class INDICATOR_ARRAY, class LABEL_ARRAY, class GRADIENT_ARRAY>
    double compute_mutex_malis_gradient(const xt::xexpression<WEIGHT_ARRAY> & flat_weights_exp,
                                        const xt::xexpression<INDEX_ARRAY> & sorted_flat_indices_exp,
                                        const xt::xexpression<INDICATOR_ARRAY> & valid_edges_exp,
                                        xt::xexpression<LABEL_ARRAY> & gt_labels_flat_exp,
                                        const std::vector<std::vector<int>> & offsets,
                                        const size_t number_of_attractive_channels,
                                        const std::vector<int> & image_shape,
                                        xt::xexpression<GRADIENT_ARRAY> & gradient_exp,
                                        xt::xexpression<LABEL_ARRAY> & labels_pos_exp,
                                        xt::xexpression<LABEL_ARRAY> & labels_neg_exp,
                                        const bool learn_in_ignore_label) {

        typedef typename GRADIENT_ARRAY::value_type GradType;

        // casts
        const auto & flat_weights = flat_weights_exp.derived_cast();
        const auto & sorted_flat_indices = sorted_flat_indices_exp.derived_cast();
        const auto & valid_edges = valid_edges_exp.derived_cast();
        auto & gt_labels = gt_labels_flat_exp.derived_cast();
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
        typedef boost::container::flat_set<uint64_t> SetType;
        typedef std::vector<SetType> MutexStorage;
        MutexStorage mutexes(number_of_nodes);

        // additional malis datastructures
        // data structures for overlaps of nodes (= pixels) with gt labels
        // and sizes of gt segments
        OverlapType overlaps(number_of_nodes);
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
        std::vector<uint64_t> ignore_label_merge_q;
        std::vector<uint64_t> ignore_label_mutex_q;
        std::vector<std::pair<uint64_t, uint64_t>> merge_q;
        std::vector<uint64_t> neg_q;

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

            // find the current representatives
            uint64_t ru = ufd.find_set(u);
            uint64_t rv = ufd.find_set(v);
            // if the nodes are already connected, do nothing
            if(ru == rv) {
                continue;
            }

            const auto lu = gt_labels[u];
            const auto lv = gt_labels[v];

            const bool touches_ignore_label = lu == 0 || lv == 0;

            if(learn_in_ignore_label && touches_ignore_label){
                if (is_mutex) ignore_label_mutex_q.push_back(edge_id);
                else          ignore_label_merge_q.push_back(edge_id);
                continue;
            }

            if(lu != lv) {
                if (!touches_ignore_label) {
                    neg_q.push_back(edge_id);
                }
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
                segmentation::insert_mutex(ru, rv, mutexes);
                compute_gradient(ru, rv, overlaps, false, true, loss, flat_weights[edge_id], grads[edge_id]);

            } else {

                // otherwise merge and compute gradients
                ufd.link(u, v);

                // check  if we have to swap the roots
                if(ufd.find_set(ru) == rv) {
                    std::swap(ru, rv);
                }
                // merge mutexes from rv -> ru
                segmentation::merge_mutexes(rv, ru, mutexes);
                compute_gradient(ru, rv, overlaps, true, true, loss, flat_weights[edge_id], grads[edge_id]);
                add_overlaps(ru, rv, overlaps);
            }
        }

        if (learn_in_ignore_label){

            // merge all nodes inside ignore regions 
            // this queue is empty if !learn_in_ignore_label
            for(const auto & edge_id: ignore_label_merge_q) {
                // get nodes connected by edge of edge_id
                const uint64_t u = edge_id % number_of_nodes;
                const uint64_t v = u + offset_strides[edge_id / number_of_nodes];

                // find the current representatives
                uint64_t ru = ufd.find_set(u);
                uint64_t rv = ufd.find_set(v);

                if(ru == rv) {
                    continue;
                }

                const auto lu = gt_labels[ru];
                const auto lv = gt_labels[rv];

                const bool on_boundary = lu != 0 && lv != 0 && lu != lv; 
                if (!on_boundary){
                    // merge
                    ufd.link(u, v);
                   // check  if we have to swap the roots
                    if(ufd.find_set(ru) == rv) {
                        std::swap(ru, rv);
                    }

                    if (gt_labels[ru] == 0){
                        gt_labels[ru] = lu == 0 ? lv : lu;
                    }
                    // merge mutexes from rv -> ru
                    segmentation::merge_mutexes(rv, ru, mutexes);
                    compute_gradient(ru, rv, overlaps, true, true, loss, flat_weights[edge_id], grads[edge_id]);
                    // the computed gradient can be zero if all considered pairs
                    // have only ignore labels
                    // therefore we set a minimum gradient with n_pair = 1 
                    if (grads[edge_id] == 0.){
                        grads[edge_id] = flat_weights[edge_id] - 1.;
                        loss += grads[edge_id] * grads[edge_id];
                    }
                    add_overlaps(ru, rv, overlaps);
                }
                else{
                    // this merge edge is on the boundary
                    // label nodes correctly
                    gt_labels[u] = gt_labels[ru];
                    gt_labels[v] = gt_labels[rv];
                    // insert edge into the sorted neg_q
                    for(int i = 0; i < neg_q.size(); ++i){
                        if (flat_weights[neg_q[i]] < flat_weights[edge_id]){
                            neg_q.insert(neg_q.begin()+i, edge_id);
                            break;
                        }
                    }
                }
            }


            // now the full image is labeled and we can treat the left over 
            // mutex edges properly
            // this queue is empty if !learn_in_ignore_label
            for(const auto & edge_id: ignore_label_mutex_q) {
                // get nodes connected by edge of edge_id
                const uint64_t u = edge_id % number_of_nodes;
                const uint64_t v = u + offset_strides[edge_id / number_of_nodes];

                // find the current representatives
                uint64_t ru = ufd.find_set(u);
                uint64_t rv = ufd.find_set(v);

                if(ru == rv) {
                    continue;
                }

                const auto lu = gt_labels[ru];
                const auto lv = gt_labels[rv];

                const bool on_boundary = lu != lv;
                if (!on_boundary){
                    merge_q.push_back(std::make_pair(u, v));
                    // // compute gradient for the mutex edge
                    segmentation::insert_mutex(ru, rv, mutexes);
                    compute_gradient(ru, rv, overlaps, false, true, loss, flat_weights[edge_id], grads[edge_id]);
                    // the computed gradient can be zero if all considered pairs
                    // have only ignore labels
                    // therefore we set a minimum gradient with n_pair = 1 
                    if (grads[edge_id] == 0.){
                        grads[edge_id] = flat_weights[edge_id];
                        loss += grads[edge_id] * grads[edge_id];
                    }
                }
                else{
                    // this is on the boundary
                    // write root information to label image and
                    // push edge to negative queue
                    gt_labels[u] = gt_labels[ru];
                    gt_labels[v] = gt_labels[rv];
                    // insert mutex edge into the sorted neg_q
                    for(int i = 0; i < neg_q.size(); ++i){
                        if (flat_weights[neg_q[i]] < flat_weights[edge_id]){
                            neg_q.insert(neg_q.begin()+i, edge_id);
                            break;
                        }
                    }
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
            uint64_t ru = ufd.find_set(merge_pair.first);
            uint64_t rv = ufd.find_set(merge_pair.second);
            add_overlaps(ru, rv, overlaps);
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

            // find the current representatives
            uint64_t ru = ufd.find_set(u);
            uint64_t rv = ufd.find_set(v);

            const auto lu = gt_labels[u];
            const auto lv = gt_labels[v];

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

                // insert the mutex edge into both mutex edge storages
                segmentation::insert_mutex(ru, rv, mutexes);
                compute_gradient(ru, rv, overlaps, true, false, loss, flat_weights[edge_id], grads[edge_id]);
            } else {

                // otherwise merge and compute gradients
                ufd.link(u, v);
                // check  if we have to swap the roots
                if(ufd.find_set(ru) == rv) {
                    std::swap(ru, rv);
                }
                // merge mutexes from rv -> ru
                segmentation::merge_mutexes(rv, ru, mutexes);
                compute_gradient(ru, rv, overlaps, false, false, loss, flat_weights[edge_id], grads[edge_id]);
                add_overlaps(ru, rv, overlaps);
            }
        }
        // get neg node labeling
        for(size_t label = 0; label < number_of_nodes; ++label) {
            labels_neg[label] = ufd.find_set(label);
        }

        return loss;
    }


    template<class WEIGHT_ARRAY, class INDEX_ARRAY, class INDICATOR_ARRAY, class LABEL_ARRAY, class GRADIENT_ARRAY>
    double constrained_mutex_malis_debug(const xt::xexpression<WEIGHT_ARRAY> & flat_weights_exp,
                                         const xt::xexpression<INDEX_ARRAY> & sorted_flat_indices_exp,
                                         const xt::xexpression<INDICATOR_ARRAY> & valid_edges_exp,
                                         xt::xexpression<LABEL_ARRAY> & gt_labels_flat_exp,
                                         const std::vector<std::vector<int>> & offsets,
                                         const size_t number_of_attractive_channels,
                                         const std::vector<int> & image_shape,
                                         xt::xexpression<GRADIENT_ARRAY> & gradient_exp,
                                         xt::xexpression<LABEL_ARRAY> & labels_pos_exp,
                                         xt::xexpression<LABEL_ARRAY> & labels_neg_exp,
                                         const bool learn_in_ignore_label) {

        const auto & flat_weights = flat_weights_exp.derived_cast();
        const auto & sorted_flat_indices = sorted_flat_indices_exp.derived_cast();
        const auto & valid_edges = valid_edges_exp.derived_cast();
        auto & gradients = gradient_exp.derived_cast();

        // run positive and negative pass
        const double loss = compute_mutex_malis_gradient(flat_weights, sorted_flat_indices,
                                                             valid_edges, gt_labels_flat_exp,
                                                             offsets, number_of_attractive_channels,
                                                             image_shape, gradients,
                                                             labels_pos_exp, labels_neg_exp,
                                                             learn_in_ignore_label);

        return loss;
    }

}
}
