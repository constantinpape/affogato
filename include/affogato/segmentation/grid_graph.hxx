#pragma once

#include <random>
#include "xtensor/xtensor.hpp"
#include "affogato/util.hxx"


namespace affogato {
namespace segmentation {


    class MWSGridGraph {

    public:
        typedef std::pair<uint64_t, uint64_t> EdgeType;
        typedef std::vector<std::size_t> OffsetType;

        template<class AFFS>
        MWSGridGraph(const AFFS & affs, const std::vector<OffsetType> & offsets,
                     const std::vector<std::size_t> & strides, const bool randomize_strides) :
        _aff_shape(affs.shape().begin(), affs.shape().end()),
        _shape(_aff_shape.begin() + 1, _aff_shape.end()),
        _ndim(_shape.size()) {
            // initializers
            init_strides();
            init_nn(affs, offsets);
            init_lr(affs, offsets, strides, randomize_strides);
        }

        template<class AFFS, class MASK>
        MWSGridGraph(const AFFS & affs, const MASK & mask, const std::vector<OffsetType> & offsets,
                     const std::vector<std::size_t> & strides, const bool randomize_strides) :
        _aff_shape(affs.shape().begin(), affs.shape().end()),
        _shape(_aff_shape.begin() + 1, _aff_shape.end()),
        _ndim(_shape.size()) {
            // initializers
            init_strides();
            init_mask(mask);
            init_nn(affs, offsets);
            init_lr(affs, offsets, strides, randomize_strides);
        }

        inline uint64_t get_node(const xt::xindex & coord) const {
            uint64_t node = 0;
            for(unsigned d = 0; d < _ndim; ++d) {
                node += coord[d] * _strides[d];
            }
            return node;
        }

        const std::vector<EdgeType> & uv_ids() const {
            return _uv_ids;
        }

        const std::vector<EdgeType> & lr_uv_ids() const {
            return _lr_uv_ids;
        }

        const std::vector<float> & weights() const {
            return _weights;
        }

        const std::vector<float> & lr_weights() const {
            return _lr_weights;
        }

        // TODO we need to generalize this to support more than one time-step back ! than labels
        // would be 1 dim bigger than _dim
        template<class AFFS, class LABELS>
        void get_causal_edges(const AFFS & affs, const LABELS & labels, const std::vector<OffsetType> & offsets,
                              std::vector<EdgeType> & uv_ids, std::vector<float> & weights) const {
            // get iteration shape
            auto iter_shape = _aff_shape;
            iter_shape[0] = offsets.size();
            xt::xindex coord(_ndim), prev_coord(_ndim);
            // iterate over the causal offsets
            util::for_each_coordinate(iter_shape, [&](const xt::xindex & aff_coord){
                // set the spatial coordinates
                std::copy(aff_coord.begin() + 1, aff_coord.end(), coord.begin());
                prev_coord = coord;

                // set the spatial coords from the offsets
                const auto & offset = offsets[aff_coord[0]];

                // TODO for now we assume implicitely that offset[0] = -1 and that labels.ndim == _ndim
                // need to generalize this to arbitrary causal offsets

                // check that we are in range
                bool out_of_range = false;
                for(unsigned d = 0; d < _ndim; ++d) {
                    prev_coord[d] += offset[d + 1];
                    if(prev_coord[d] < 0 || prev_coord[d] >= _shape[d]) {
                        out_of_range = true;
                        break;
                    }
                }
                if(out_of_range) {
                    return;
                }

                // insert the edge and weight corresponding to this grid connection
                uint64_t u = get_node(coord);
                uint64_t v = labels[prev_coord];

                // check if v is masked
                if(v == 0) {
                    return;
                }

                // if we have a mask,
                // check if any of the nodes is masked
                if(_masked_nodes.size()) {
                    if(_masked_nodes[u]) {
                        return;
                    }
                }

                if(u > v) {
                    std::swap(u, v);
                }

                uv_ids.emplace_back(u, v);
                weights.emplace_back(affs[aff_coord]);
            });
        }

    private:

        void init_strides() {
            _strides.resize(_ndim);
            _strides[_ndim - 1] = 1;
            for(unsigned d = _ndim - 2; d >= 0; --d) {
                _strides[d] = _shape[d + 1] * _strides[d + 1];
            }
        }

        template<class MASK>
        void init_mask(const MASK & mask) {
            const std::size_t n_nodes = std::accumulate(_shape.begin(), _shape.end(),
                                                        1, std::multiplies<std::size_t>());
            _masked_nodes.resize(n_nodes);
            util::for_each_coordinate(_shape, [&](const xt::xindex & coord){
                _masked_nodes[get_node(coord)] = mask[coord] == 0;
            });
        }

        template<class AFFS>
        void init_nn(const AFFS & affs, const std::vector<OffsetType> & offsets) {

            // get neareset neighbor shape
            auto nn_shape = _aff_shape;
            nn_shape[0] = _ndim;
            // iterate over nearest neighbors, extract edges and weights
            xt::xindex coord(_ndim), ngb_coord(_ndim);
            util::for_each_coordinate(nn_shape, [&](const xt::xindex & aff_coord){
                // set the spatial coordinates
                std::copy(aff_coord.begin() + 1, aff_coord.end(), coord.begin());
                ngb_coord = coord;

                // set the spatial coords from the offsets
                const auto & offset = offsets[aff_coord[0]];

                // check that we are in range
                bool out_of_range = false;
                for(unsigned d = 0; d < _ndim; ++d) {
                    ngb_coord[d] += offset[d];
                    if(ngb_coord[d] < 0 || ngb_coord[d] >= _shape[d]) {
                        out_of_range = true;
                        break;
                    }
                }
                if(out_of_range) {
                    return;
                }

                // insert the edge and weight corresponding to this grid connection
                uint64_t u = get_node(coord);
                uint64_t v = get_node(ngb_coord);

                // if we have a mask,
                // check if any of the nodes is masked
                if(_masked_nodes.size()) {
                    if(_masked_nodes[u] || _masked_nodes[v]) {
                        return;
                    }
                }

                if(u > v) {
                    std::swap(u, v);
                }

                _uv_ids.emplace_back(u, v);
                _weights.emplace_back(affs[aff_coord]);
            });
        }

        template<class AFFS>
        void init_lr(const AFFS & affs, const std::vector<OffsetType> & offsets,
                     const std::vector<std::size_t> & strides, const bool randomize_strides) {

            // check if we have strides
            const std::size_t stride_product = std::accumulate(strides.begin(), strides.end(), 1,
                                                               std::multiplies<std::size_t>());
            const bool have_strides = stride_product > 1;

            // call appropriate initializer
            if(have_strides && randomize_strides) {
                // compute fraction of lr edges that we keep
                const double lr_fraction = 1. / stride_product;
                init_lr(affs, offsets, lr_fraction);
            }
            else if(have_strides) {
                init_lr(affs, offsets, strides);
            }
            else {
                init_lr(affs, offsets);
            }
        }

        template<class AFFS>
        void init_lr(const AFFS & affs, const std::vector<OffsetType> & offsets) {
            // iterate over lr  neighbors, extract edges and weights
            xt::xindex coord(_ndim), ngb_coord(_ndim);
            util::for_each_coordinate(_aff_shape, [&](const xt::xindex & aff_coord){

                // skip nn edges
                if(aff_coord[0] < _ndim) {
                    return;
                }

                // set the spatial coordinates
                std::copy(aff_coord.begin() + 1, aff_coord.end(), coord.begin());
                ngb_coord = coord;

                // set the spatial coords from the offsets
                const auto & offset = offsets[aff_coord[0]];

                // check that we are in range
                bool out_of_range = false;
                for(unsigned d = 0; d < _ndim; ++d) {
                    ngb_coord[d] += offset[d];
                    if(ngb_coord[d] < 0 || ngb_coord[d] >= _shape[d]) {
                        out_of_range = true;
                        break;
                    }
                }
                if(out_of_range) {
                    return;
                }

                // insert the edge and weight corresponding to this grid connection
                uint64_t u = get_node(coord);
                uint64_t v = get_node(ngb_coord);

                // if we have a mask,
                // check if any of the nodes is masked
                if(_masked_nodes.size()) {
                    if(_masked_nodes[u] || _masked_nodes[v]) {
                        return;
                    }
                }

                if(u > v) {
                    std::swap(u, v);
                }

                _lr_uv_ids.emplace_back(u, v);
                _lr_weights.emplace_back(affs[aff_coord]);
            });
        }

        template<class AFFS>
        void init_lr(const AFFS & affs, const std::vector<OffsetType> & offsets,
                     const double lr_fraction) {

            //
            std::default_random_engine generator;
            std::uniform_real_distribution<float> distribution;
            auto draw = std::bind(distribution, generator);

            // iterate over lr  neighbors, extract edges and weights
            xt::xindex coord(_ndim), ngb_coord(_ndim);
            util::for_each_coordinate(_aff_shape, [&](const xt::xindex & aff_coord){

                // skip nn edges
                if(aff_coord[0] < _ndim) {
                    return;
                }

                // draw random number to check if we keep this edge
                if(draw() < lr_fraction) {
                    return;
                }

                // set the spatial coordinates
                std::copy(aff_coord.begin() + 1, aff_coord.end(), coord.begin());
                ngb_coord = coord;

                // set the spatial coords from the offsets
                const auto & offset = offsets[aff_coord[0]];

                // check that we are in range
                bool out_of_range = false;
                for(unsigned d = 0; d < _ndim; ++d) {
                    ngb_coord[d] += offset[d];
                    if(ngb_coord[d] < 0 || ngb_coord[d] >= _shape[d]) {
                        out_of_range = true;
                        break;
                    }
                }
                if(out_of_range) {
                    return;
                }

                // insert the edge and weight corresponding to this grid connection
                uint64_t u = get_node(coord);
                uint64_t v = get_node(ngb_coord);

                // if we have a mask,
                // check if any of the nodes is masked
                if(_masked_nodes.size()) {
                    if(_masked_nodes[u] || _masked_nodes[v]) {
                        return;
                    }
                }

                if(u > v) {
                    std::swap(u, v);
                }

                _lr_uv_ids.emplace_back(u, v);
                _lr_weights.emplace_back(affs[aff_coord]);
            });
        }

        template<class AFFS>
        void init_lr(const AFFS & affs, const std::vector<OffsetType> & offsets,
                     const std::vector<std::size_t> & strides) {

            // iterate over lr  neighbors, extract edges and weights
            xt::xindex coord(_ndim), ngb_coord(_ndim);
            util::for_each_coordinate(_aff_shape, [&](const xt::xindex & aff_coord){

                // skip nn edges
                if(aff_coord[0] < _ndim) {
                    return;
                }

                // set the spatial coordinates
                std::copy(aff_coord.begin() + 1, aff_coord.end(), coord.begin());
                ngb_coord = coord;

                bool valid_stride = true;
                for(unsigned d = 0; d < _ndim; ++d) {
                    if(coord[d] % strides[d] != 0) {
                        valid_stride = false;
                        break;
                    }
                }

                // check if we keep this stride
                if(!valid_stride) {
                    return;
                }

                // set the spatial coords from the offsets
                const auto & offset = offsets[aff_coord[0]];

                // check that we are in range
                bool out_of_range = false;
                for(unsigned d = 0; d < _ndim; ++d) {
                    ngb_coord[d] += offset[d];
                    if(ngb_coord[d] < 0 || ngb_coord[d] >= _shape[d]) {
                        out_of_range = true;
                        break;
                    }
                }
                if(out_of_range) {
                    return;
                }

                // insert the edge and weight corresponding to this grid connection
                uint64_t u = get_node(coord);
                uint64_t v = get_node(ngb_coord);

                // if we have a mask,
                // check if any of the nodes is masked
                if(_masked_nodes.size()) {
                    if(_masked_nodes[u] || _masked_nodes[v]) {
                        return;
                    }
                }

                if(u > v) {
                    std::swap(u, v);
                }

                _lr_uv_ids.emplace_back(u, v);
                _lr_weights.emplace_back(affs[aff_coord]);
            });
        }


    private:
        xt::xindex _aff_shape;
        xt::xindex _shape;
        unsigned _ndim;
        xt::xindex _strides;
        //
        std::vector<EdgeType> _uv_ids;
        std::vector<EdgeType> _lr_uv_ids;
        std::vector<float> _weights;
        std::vector<float> _lr_weights;
        std::vector<bool> _masked_nodes;
    };


}
}