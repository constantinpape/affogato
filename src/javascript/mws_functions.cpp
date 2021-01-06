#include <emscripten/bind.h>

#include "affogato/segmentation/mutex_watershed.hxx"

using namespace emscripten;


// Arguments:
// weights - flattened network predictions
// offsets - affinity offsets
// image_shape - 
// number_of_attractive_edges - 
// Returns:
// vector with node labels
std::vector<int> mws_js_2d(
    const std::vector<float> & weights,
    const std::vector<int> & offsets_flat,
    const int number_of_channels,
    const int h,
    const int w
) {
    
    const int64_t number_of_nodes = h * w;
    const int64_t number_of_edges = number_of_channels * number_of_nodes;
    // TODO hw or wh?
    const std::vector<int> image_shape({h, w}); 
    
    std::vector<std::vector<int>> offsets(number_of_channels);
    int chan_offset = 0;
    for(int chan_id = 0; chan_id < number_of_channels; ++chan_id, chan_offset+=2) {
        std::vector<int> this_offset = {offsets_flat[chan_offset],
                                        offsets_flat[chan_offset + 1]};
        offsets[chan_id] = this_offset;
    }

    // Number of attractive channels is hard-coded to 2
    const int number_of_attractive_channels = 2;

    // argsort ALL edges
    // we sort in ascending order
    std::vector<size_t> indices(number_of_edges);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](const size_t a, const size_t b){
        return weights[a] > weights[b];
    });

    const auto sorted_indices = xt::adapt(indices, {number_of_edges});

    // TODO the valid edge array should also be exposed
    xt::xtensor<bool, 1> valid_edges = xt::ones<bool>({number_of_edges});

    std::vector<int> node_labels(number_of_nodes);

    // xt::pytensor<uint64_t, 1> node_labeling = xt::zeros<uint64_t>({number_of_nodes});
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


// TODO implement random init
std::vector<float> create_float_vector(const std::size_t n_weights, const bool initialise_random) {
    std::vector<float> weights(n_weights);
    if(initialise_random) {

    }
    return weights;
}


std::vector<int> create_int_vector(const std::size_t n_elems) {
    std::vector<int> vec(n_elems);
    return vec;
}


EMSCRIPTEN_BINDINGS(my_module) {
    function("mutex_watershed_2d", &mws_js_2d);
    function("create_float_vector", &create_float_vector);
    function("create_int_vector", &create_int_vector);
    
    register_vector<float>("FloatVector");
    register_vector<int>("IntVector");
}
