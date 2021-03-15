#include <random>
#include <emscripten/bind.h>

#include "affogato/segmentation/mutex_watershed.hxx"

using namespace emscripten;


//
// helper functions
//


template<class EDGES>
void get_valid_edges(
        EDGES & edges,
        const std::vector<int> & strides,
        const bool randomize_strides,
        const size_t number_of_attractive_edges
        ) {
    
    const int stride_product = std::accumulate(strides.begin(), strides.end(), 1, std::multiplies<int>());
    if(stride_product == 1) {
        return;
    }

    if(randomize_strides) {
        const double p_keep = 1. / static_cast<double>(stride_product); 
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(0., 1.);
        auto draw = std::bind(distribution, generator);

        for(size_t edge_id = 0; edge_id < edges.size(); ++edge_id) {
            if(edge_id >= number_of_attractive_edges) {  // is this a mutex edge?
                if(draw() > p_keep) {
                    edges[edge_id] = false;
                }
            }
        }
    } else {
        // This is a bit more complicated here ...
        throw std::logic_error("Masking from strides is not implemented yet");
    }
}


template<class EDGES, class MASK>
void get_valid_edges_with_mask(
        EDGES & edges,
        MASK & mask,
        const std::vector<int> & strides,
        const bool randomize_strides,
        const size_t number_of_attractive_edges
        ) {
    
    const int stride_product = std::accumulate(strides.begin(), strides.end(), 1, std::multiplies<int>());
    if(stride_product == 1) {
        return;
    }

    if(randomize_strides) {
        const double p_keep = 1. / static_cast<double>(stride_product); 
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(0., 1.);
        auto draw = std::bind(distribution, generator);

        const size_t number_of_nodes = mask.size();
        size_t node_id;
        for(size_t edge_id = 0; edge_id < edges.size(); ++edge_id) {
            node_id = edge_id % number_of_nodes;
            if(!mask[node_id]) {
                edges[edge_id] = false;
                continue;
            }

            if(edge_id >= number_of_attractive_edges) {  // is this a mutex edge?
                if(draw() > p_keep) {
                    edges[edge_id] = false;
                }
            }
        }
    } else {
        // This is a bit more complicated here ...
        throw std::logic_error("Masking from strides is not implemented yet");
    }
}


template<class INDICES, class OFFSETS,
         class WEIGHTS, class OFFSETS_FLAT>
void get_indices_and_offsets(
    INDICES & indices,
    OFFSETS & offsets,
    const WEIGHTS & weights,
    const OFFSETS_FLAT & offsets_flat,
    const unsigned dim
) {

    // argsort ALL edges
    // we sort in ascending order based on the weights
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](const size_t a, const size_t b){
        return weights[a] > weights[b];
    });

    // copy from flat to actual offsets
    for(int chan_id = 0; chan_id < offsets.size(); ++chan_id) {
        std::vector<int> this_offset = {offsets_flat[dim*chan_id],
                                        offsets_flat[dim*chan_id + 1]};
        offsets[chan_id] = this_offset;
    }
}


//
// default mws implementation (without mask)
//


// Arguments:
// weights - flattened network predictions
// offsets_flat - affinity offsets (flat vector)
// strides - stride vector
// image_shape - shape of the image
// randomize strides - whether to randomize the strides
// Returns:
// vector with node labels
std::vector<int> mws_js_impl(
    const std::vector<float> & weights,
    const std::vector<int> & offsets_flat,
    const std::vector<int> & strides,
    const std::vector<int> & image_shape,
    const bool randomize_strides
) {
    
    const unsigned dim = image_shape.size();
    const int64_t number_of_channels = offsets_flat.size() / dim;
    const int64_t number_of_nodes = std::accumulate(image_shape.begin(), image_shape.end(), 1L, std::multiplies<int64_t>());
    const int64_t number_of_edges = number_of_channels * number_of_nodes;
    
    // Number of attractive channels is hard-coded to the number of dimensions
    const int number_of_attractive_channels = dim;
    const size_t number_of_attractive_edges = number_of_nodes * number_of_attractive_channels;

    std::vector<size_t> indices(number_of_edges);
    std::vector<std::vector<int>> offsets(number_of_channels);
    get_indices_and_offsets(indices, offsets, weights, offsets_flat, dim);
    const auto sorted_indices = xt::adapt(indices, {number_of_edges});

    xt::xtensor<bool, 1> valid_edges = xt::ones<bool>({number_of_edges});
    get_valid_edges(valid_edges, strides, randomize_strides, number_of_attractive_edges);

    std::vector<int> node_labels(number_of_nodes);
    std::vector<size_t> node_shape = {static_cast<size_t>(number_of_nodes)};
    auto node_labeling = xt::adapt(node_labels, node_shape);
    affogato::segmentation::compute_mws_segmentation(sorted_indices,
                                                     valid_edges,
                                                     offsets,
                                                     number_of_attractive_channels,
                                                     image_shape,
                                                     node_labeling);
    return node_labels;
}


std::vector<int> mws_js_2d(
    const std::vector<float> & weights,
    const std::vector<int> & offsets_flat,
    const std::vector<int> & strides,
    const int h,
    const int w,
    const bool randomize_strides
) {
    const std::vector<int> image_shape({h, w}); 
    return mws_js_impl(weights, offsets_flat, strides, image_shape, randomize_strides);
}


std::vector<int> mws_js_3d(
    const std::vector<float> & weights,
    const std::vector<int> & offsets_flat,
    const std::vector<int> & strides,
    const int d,
    const int h,
    const int w,
    const bool randomize_strides
) {
    const std::vector<int> image_shape({d, h, w}); 
    return mws_js_impl(weights, offsets_flat, strides, image_shape, randomize_strides);
}


//
// mws implementation with mask
//


// Arguments:
// weights - flattened network predictions
// mask - flattened mask (only spatial!)
// offsets_flat - affinity offsets (flat vector)
// strides - stride vector
// image_shape - shape of the image
// randomize strides - whether to randomize the strides
// Returns:
// vector with node labels
std::vector<int> mws_masked_js_impl(
    const std::vector<float> & weights,
    const std::vector<int> & mask,
    const std::vector<int> & offsets_flat,
    const std::vector<int> & strides,
    const std::vector<int> & image_shape,
    const bool randomize_strides
) {
    
    const unsigned dim = image_shape.size();
    const int64_t number_of_channels = offsets_flat.size() / dim;
    const int64_t number_of_nodes = std::accumulate(image_shape.begin(), image_shape.end(), 1L, std::multiplies<int64_t>());
    if(mask.size() != number_of_nodes) {
        throw std::invalid_argument("Mask has wrong size");
    }
    const int64_t number_of_edges = number_of_channels * number_of_nodes;
    
    // Number of attractive channels is hard-coded to the number of dimensions
    const int number_of_attractive_channels = dim;
    const size_t number_of_attractive_edges = number_of_nodes * number_of_attractive_channels;

    std::vector<size_t> indices(number_of_edges);
    std::vector<std::vector<int>> offsets(number_of_channels);
    get_indices_and_offsets(indices, offsets, weights, offsets_flat, dim);
    const auto sorted_indices = xt::adapt(indices, {number_of_edges});

    xt::xtensor<bool, 1> valid_edges = xt::ones<bool>({number_of_edges});
    get_valid_edges_with_mask(valid_edges, mask, strides, randomize_strides, number_of_attractive_edges);

    std::vector<int> node_labels(number_of_nodes);
    std::vector<size_t> node_shape = {static_cast<size_t>(number_of_nodes)};
    auto node_labeling = xt::adapt(node_labels, node_shape);
    affogato::segmentation::compute_mws_segmentation(sorted_indices,
                                                     valid_edges,
                                                     offsets,
                                                     number_of_attractive_channels,
                                                     image_shape,
                                                     node_labeling);
    return node_labels;
}


std::vector<int> mws_masked_js_2d(
    const std::vector<float> & weights,
    const std::vector<int> & mask,
    const std::vector<int> & offsets_flat,
    const std::vector<int> & strides,
    const int h,
    const int w,
    const bool randomize_strides
) {
    const std::vector<int> image_shape({h, w}); 
    return mws_masked_js_impl(weights, mask, offsets_flat, strides, image_shape, randomize_strides);
}


std::vector<int> mws_masked_js_3d(
    const std::vector<float> & weights,
    const std::vector<int> & mask,
    const std::vector<int> & offsets_flat,
    const std::vector<int> & strides,
    const int d,
    const int h,
    const int w,
    const bool randomize_strides
) {
    const std::vector<int> image_shape({d, h, w}); 
    return mws_masked_js_impl(weights, mask, offsets_flat, strides, image_shape, randomize_strides);
}


//
// expose vector types to java script
//


std::vector<float> create_float_vector(const std::size_t n_weights) {
    std::vector<float> weights(n_weights);
    return weights;
}

std::vector<int> create_int_vector(const std::size_t n_elems) {
    std::vector<int> vec(n_elems);
    return vec;
}

/*
std::vector<bool> create_bool_vector(const std::size_t n_elems) {
    std::vector<bool> vec(n_elems);
    return vec;
}
*/


//
// the actual bindings
//


EMSCRIPTEN_BINDINGS(my_module) {
    function("mutex_watershed_2d", &mws_js_2d);
    function("mutex_watershed_3d", &mws_js_3d);

    function("mutex_watershed_masked_2d", &mws_masked_js_2d);
    function("mutex_watershed_masked_3d", &mws_masked_js_3d);

    function("create_float_vector", &create_float_vector);
    function("create_int_vector", &create_int_vector);
    // function("create_bool_vector", &create_bool_vector);
    
    register_vector<float>("FloatVector");
    register_vector<int>("IntVector");
    // NOTE this does not compile, so for now we use int vectors for the mask
    // register_vector<bool>("BoolVector");
}
