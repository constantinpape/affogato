#pragma once

#include <deque>
#include <stdexcept>
#include <unordered_map>

#include <boost/pending/disjoint_sets.hpp>
#include <boost/functional/hash.hpp>
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"

#include "affogato/util.hxx"

namespace affogato {
namespace segmentation {


template<class EDGE_WEIGHTS, class NODE_WEIGHTS>
inline void get_node_weights(const xt::xexpression<EDGE_WEIGHTS> & edge_weights_exp,
                             const double lower_threshold,
                             xt::xexpression<NODE_WEIGHTS> & node_weights_exp,
                             const bool ignore_border = false) {

    typedef typename EDGE_WEIGHTS::value_type ValueType;
    const auto & edge_weights = edge_weights_exp.derived_cast();
    auto & node_weights = node_weights_exp.derived_cast();

    xt::xindex shape(node_weights.shape().begin(), node_weights.shape().end());
    const unsigned ndim = shape.size();

    // ValueType infinity = *std::max_element(edge_weights.begin(), edge_weights.end()) + .1;
    const ValueType infinity = 1.1;
    // iterate over all spatial coordinates. for each node, find all weigths that are connected to it
    util::for_each_coordinate(shape, [&](const xt::xindex coord){

        ValueType weight = 0;
        // get the weights connecting this node to other nodes
        for(unsigned d = 0; d < ndim; ++d) {
            // NOTE this assumes affinity offset conventions [[-1, 0], [0, -1]] etc.
            for(auto dir : {0, 1}) {
                // get the corresponding affi
                xt::xindex aff_coord(ndim + 1);
                std::copy(coord.begin(), coord.end(), aff_coord.begin() + 1);
                aff_coord[0] = d;
                aff_coord[d + 1] += dir;
                ValueType curr_weight;
                if(aff_coord[d + 1] > 0 && aff_coord[d + 1] < shape[d]) {
                    curr_weight = edge_weights[aff_coord];
                } else {
                    continue;
                }
                if(curr_weight > weight) {
                    weight = curr_weight;
                }
            }
        }
        node_weights[coord] = (weight > lower_threshold) ? infinity : weight;

        if(ignore_border) {
            bool at_border = false;
            for(unsigned d = 0; d < ndim; ++d) {
                if(coord[d] == 0 || coord[d] == shape[d] - 1) {
                    at_border = true;
                    break;
                }
            }
            if(at_border) {
                node_weights[coord] = infinity;
            }
        }
    });
}


template<class EDGE_WEIGHTS, class NODE_WEIGHTS, class LABELS>
inline uint64_t stream(const xt::xindex & start_coord,
                       const xt::xexpression<EDGE_WEIGHTS> & edge_weights_exp,
                       const xt::xexpression<NODE_WEIGHTS> & node_weights_exp,
                       const xt::xexpression<LABELS> & labels_exp,
                       const double upper_threshold,
                       std::vector<xt::xindex> & stream_coordinates
                       /*,std::ofstream debug*/ ) {

    typedef typename EDGE_WEIGHTS::value_type ValueType;
    // add pixel to the stream coordinates
    stream_coordinates.emplace_back(start_coord);
    const unsigned ndim = start_coord.size();

    const auto & edge_weights = edge_weights_exp.derived_cast();
    const auto & node_weights = node_weights_exp.derived_cast();
    const auto & labels = labels_exp.derived_cast();

    xt::xindex shape(labels.shape().begin(), labels.shape().end());

    // initialize pixel queue
    std::deque<xt::xindex> queue;
    queue.push_back(start_coord);

    const ValueType infinity = 1.1;

    while(!queue.empty()) {

        const auto & coord = queue.front();
        queue.pop_front();

        ValueType w_max = node_weights[coord];

        // iterate over the connection to neighbors for this coordinate
        for(unsigned d = 0; d < ndim; ++d) {
            // NOTE this assumes affinity offset conventions [[-1, 0], [0, -1]] etc.
            for(auto dir : {0, 1}) {
                // get the corresponding affinity coordinate and neighbor coordinate
                xt::xindex ngb_coord(ndim), aff_coord(ndim + 1);

                std::copy(coord.begin(), coord.end(), aff_coord.begin() + 1);
                aff_coord[0] = d;
                aff_coord[d + 1] += dir;

                std::copy(coord.begin(), coord.end(), ngb_coord.begin());
                ngb_coord[d] += (dir == 0) ? -1 : 1;

                // range check
                if(ngb_coord[d] <= 0 || ngb_coord[d] >= shape[d]) {
                    continue;
                }

                ValueType weight = edge_weights[aff_coord];
                weight = (weight > upper_threshold) ? infinity : weight;

                // only consider neighbor if its weight is equal to the nodes max-weight
                if(fabs(weight - w_max) > std::numeric_limits<ValueType>::epsilon()) {
                    continue;
                }

                // only consider neighbor if it's coordinate is not already in the stream
                if(std::find(stream_coordinates.begin(), stream_coordinates.end(), ngb_coord) != stream_coordinates.end()) {
                    continue;
                }

                // if we hit a labeled coordinate, return it's label
                if(labels[ngb_coord] != 0 ) {
                    return labels[ngb_coord];
                }

                // if the node weight of the considered pixel is smaller, we start depth first search from it
                else if(node_weights[ngb_coord] < w_max ) {
                    stream_coordinates.emplace_back(ngb_coord);
                    queue.clear();
                    queue.push_back(ngb_coord);
                    // break looping over the current neighbors and go to new pix
                    break;
                }
                else {
                    stream_coordinates.push_back(ngb_coord);
                    queue.push_back(ngb_coord);
                }
            }
        }
    }
    // return 0, if we have not found a labeled pixel in the stream
    return 0;
}


template<class EDGE_WEIGHTS, class NODE_WEIGHTS, class LABELS>
inline size_t run_zws(const xt::xexpression<EDGE_WEIGHTS> & edge_weights_exp,
                      const xt::xexpression<NODE_WEIGHTS> & node_weights_exp,
                      const double upper_threshold,
                      xt::xexpression<LABELS> & labels_exp,
                      bool const ignore_border=false) {
    typedef typename LABELS::value_type LabelType;
    LabelType next_label = 1;

    const auto & edge_weights = edge_weights_exp.derived_cast();
    const auto & node_weights = node_weights_exp.derived_cast();
    auto & labels = labels_exp.derived_cast();

    // ofstream for debug output
    //std::ofstream debug;
    //debug.open("run_wsgraph_debug.txt")
    //
    //
    const xt::xindex shape(node_weights.shape().begin(), node_weights.shape().end());
    const unsigned ndim = shape.size();

    util::for_each_coordinate(shape, [&](const xt::xindex & coord){
        if(ignore_border) {
            bool is_border = false;
            for(unsigned d = 0; d < ndim; ++d) {
                if(coord[d] == 0 || coord[d] == shape[d] - 1) {
                    is_border = true;
                    break;
                }
            }
            if(is_border) {
                return;
            }
        }

        auto label = labels[coord];
        // continue if the pixel is already labeled
        if(label != 0) {
            return;
        }

        // call stream -> finds the stream belonging to the current label and pixel coordinates belonging to the stream
        std::vector<xt::xindex> stream_coordinates;
        label = stream(coord, edge_weights, node_weights, labels,
                       upper_threshold, stream_coordinates); //debug

        // if stream returns 0, we found a new stream
        if(label == 0) {
            label = next_label++;
        }

        // write the new label to all coordinates in the stream
        for(const auto & coord : stream_coordinates) {
            labels[coord] = label;
        }

    });
    //debug.close();
    return next_label;
}


template<class EDGE_WEIGHTS, class LABELS, class LINK>
inline void get_region_weights(const xt::xexpression<EDGE_WEIGHTS> & edge_weights_exp,
                               const xt::xexpression<LABELS> & labels_exp,
                               const double upper_threshold,
                               std::unordered_map<LINK, typename EDGE_WEIGHTS::value_type, boost::hash<LINK>> & links) {

    typedef LINK Link;
    typedef typename EDGE_WEIGHTS::value_type ValueType;

    const ValueType infinity = 1.1;

    const auto & edge_weights = edge_weights_exp.derived_cast();
    const auto & labels = labels_exp.derived_cast();
    xt::xindex shape(labels.shape().begin(), labels.shape().end());
    const unsigned ndim = shape.size();

    util::for_each_coordinate(shape, [&](const xt::xindex & coord) {
        const auto label = labels[coord];
        for(unsigned d = 0; d < ndim ;++d) {

            // range check
            if(coord[d] > 0) {
                continue;
            }

            xt::xindex ngb_coord(coord);
            ngb_coord[d] -= 1;

            const auto ngb_label = labels[ngb_coord];
            if(ngb_label != label) {
                Link link(std::min(ngb_label, label),
                          std::max(ngb_label, label));
                xt::xindex aff_coord(ndim + 1);
                std::copy(coord.begin(), coord.end(), aff_coord.begin() + 1);
                aff_coord[0] = d;
                ValueType weight = edge_weights[aff_coord];
                weight = (weight > upper_threshold) ? infinity : weight;

                auto link_it = links.find(link);
                if(link_it != links.end()) {
                    link_it->second = std::max(weight, link_it->second);
                } else {
                    links.emplace(link, weight);
                }
            }
        }
    });
}


template<class LINK, class WEIGHT, class LABELS>
inline size_t apply_size_filter(const std::unordered_map<LINK, WEIGHT, boost::hash<LINK>> & region_weights,
                                const size_t n_labels,
                                const size_t size_threshold,
                                const double merge_threshold,
                                xt::xexpression<LABELS> & labels_exp) {
    typedef typename LABELS::value_type LabelType;
    auto & labels = labels_exp.derived_cast();

    // dump map into a vector and sort it by value
    std::vector<std::pair<LINK, WEIGHT>> rw_vec(region_weights.begin(),
                                                region_weights.end());

    std::sort(rw_vec.begin(), rw_vec.end(), [](const std::pair<LINK, WEIGHT> & a,
                                               const std::pair<LINK, WEIGHT> & b){return a.second < b.second;});

    xt::xindex shape(labels.shape().begin(), labels.shape().end());

    // find sizes of labels
    std::vector<size_t> sizes(n_labels);
    util::for_each_coordinate(shape, [&](const xt::xindex & coord){
        ++sizes[labels[coord]];
    });

    // make ufd
    std::vector<LabelType> ranks(n_labels);
    std::vector<LabelType> parents(n_labels);
    boost::disjoint_sets<LabelType*, LabelType*> ufd(&ranks[0], &parents[0]);
    for(LabelType label = 0; label < n_labels; ++label) {
        ufd.make_set(label);
    }

    // merge regions
    for(const auto & e_and_w : rw_vec) {

        // if we have reached the value threshold, we stop filtering
        if(e_and_w.second < merge_threshold) {
            break;
        }
        const auto & edge = e_and_w.first;

        const LabelType s1 = ufd.find_set(edge.first);
        const LabelType s2 = ufd.find_set(edge.second);

        // merge two regions, if at least one of them is below the size threshold
        if(s1 != s2 && (sizes[s1] < size_threshold || sizes[s2] < size_threshold)) {
            const size_t size = sizes[s1] + sizes[s2];
            sizes[s1] = 0;
            sizes[s2] = 0;
            ufd.link(s1, s2);
            sizes[ufd.find_set(s1)] = size;
        }
    }

    std::vector<LabelType> new_labels(n_labels);
    LabelType next_label = 1;
    for(LabelType label = 0; label < n_labels; ++label) {
        const auto repr = ufd.find_set(label);
        if(sizes[repr] < size_threshold) {
            new_labels[label] = 0;
        } else {
            new_labels[label] = next_label++;
        }
    }

    // write the new labels
    util::for_each_coordinate(shape, [&](const xt::xindex & coord){
        labels[coord] = new_labels[labels[coord]];
    });

    return next_label;
}


// z-watershed, implementation based on
// http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4564470&tag=1
// and http://dspace.mit.edu/handle/1721.1/66820
template<class WEIGHTS, class LABELS>
size_t compute_zws_segmentation(const xt::xexpression<WEIGHTS> & edge_weights_exp,
							    const double lower_threshold,
        					    const double upper_threshold,
							    const double merge_threshold,
        					    const size_t size_threshold,
                                xt::xexpression<LABELS> & labels_exp) {

    typedef typename WEIGHTS::value_type ValueType;
    typedef typename LABELS::value_type LabelType;
    if(lower_threshold > upper_threshold) {
        throw std::runtime_error("Thresholds inverted!");
    }

    const auto & edge_weights = edge_weights_exp.derived_cast();
    // initialize the node weights, which are one dimension less than the edge weights
    typedef typename xt::xarray<ValueType>::shape_type ShapeType;
    ShapeType shape(edge_weights.shape().begin() + 1, edge_weights.shape().end());
    xt::xarray<ValueType> node_weights = xt::zeros<ValueType>(shape);

    // std::cout << "computing node weights from edge weights" << std::endl;
    get_node_weights(edge_weights, lower_threshold, node_weights);

    // threshold_edge_weights(edge_weights, upper_threshold);
    // std::cout << "run zws" << std::endl;
    size_t n_labels = run_zws(edge_weights, node_weights, upper_threshold, labels_exp);
    // std::cout << n_labels << " labels after zws" << std::endl;

    // if we don't do merging or size filtering, we can return here already
    if(merge_threshold == 0 && size_threshold == 0) {
        return n_labels;
    }

    typedef std::pair<LabelType, LabelType> Link;
    std::unordered_map<Link, ValueType, boost::hash<Link>> region_weights;
    // std::cout << "get region weights" << std::endl;
    get_region_weights(edge_weights, labels_exp, upper_threshold, region_weights);

    // std::cout << "apply size filter" << std::endl;
    n_labels = apply_size_filter(region_weights, n_labels,
                                 size_threshold, merge_threshold,
                                 labels_exp);

    return n_labels;
}


}
}
