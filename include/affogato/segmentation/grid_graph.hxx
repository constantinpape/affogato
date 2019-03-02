#pragma once

#include <random>
#include "xtensor/xtensor.hpp"
#include "affogato/util.hxx"


namespace affogato {
namespace segmentation {


    class MWSGridGraph {

    public:
        typedef std::pair<uint64_t, uint64_t> EdgeType;
        typedef std::vector<int> OffsetType;

        MWSGridGraph(const std::vector<std::size_t> & shape): _shape(shape.begin(), shape.end()),
                                                              _ndim(shape.size()),
                                                              _n_nodes(std::accumulate(shape.begin(),
                                                                                       shape.end(),
                                                                                       1, std::multiplies<std::size_t>())){
            init_strides();
        }

        template<class MASK>
        inline void set_mask(const MASK & mask) {
            init_mask(mask);
        }

        inline void clear_mask() {
            _masked_nodes.clear();
        }

        template<class AFFS>
        inline void compute_weights_and_nh_from_affs(const AFFS & affs, const std::vector<OffsetType> & offsets,
                                                     const std::vector<std::size_t> & strides, const bool randomize_strides) {
            // clear
            _uv_ids.clear();
            _lr_uv_ids.clear();
            _weights.clear();
            _lr_weights.clear();
            // compute
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

        std::size_t n_nodes() const {
            return _n_nodes;
        }

        // TODO we need to generalize this to support more than one time-step back ! than labels
        // would be 1 dim bigger than _dim
        template<class AFFS, class LABELS>
        void get_causal_edges(const AFFS & affs, const LABELS & labels, const std::vector<OffsetType> & offsets,
                              std::vector<EdgeType> & uv_ids, std::vector<float> & weights) const {
            // get iteration shape
            auto iter_shape = affs.shape();
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

                uv_ids.emplace_back(u, v);
                weights.emplace_back(affs[aff_coord]);
            });
        }

        template<class AFFS>
        void init_general_nn(const AFFS & affs, const std::vector<OffsetType> & offsets) {

            // get neareset neighbor shape
            auto nn_shape = affs.shape();
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
        void init_general_lr(const AFFS & affs, const std::vector<OffsetType> & offsets,
                     const std::vector<std::size_t> & strides, const bool randomize_strides) {

            // check if we have strides
            const std::size_t stride_product = std::accumulate(strides.begin(), strides.end(), 1,
                                                               std::multiplies<std::size_t>());
            const bool have_strides = stride_product > 1;

            // call appropriate initializer
            if(have_strides && randomize_strides) {
                // compute fraction of lr edges that we keep
                const double lr_fraction = 1. / stride_product;
                std::cout << "2222222222" << std::endl;
                init_general_lr(affs, offsets, lr_fraction);
            }
            else if(have_strides) {
                std::logic_error("Function not yet implemented");
            }
            else {
                std::logic_error("Function not yet implemented");
            }
        }


        template<class AFFS>
        void init_general_lr(const AFFS & affs, const std::vector<OffsetType> & offsets,
                     const double lr_fraction) {

            //
            std::default_random_engine generator;
            std::uniform_real_distribution<float> distribution;
            auto draw = std::bind(distribution, generator);

            // iterate over lr  neighbors, extract edges and weights
            const auto & aff_shape = affs.shape();
            xt::xindex coord(_ndim), ngb_coord(_ndim);
            util::for_each_coordinate(aff_shape, [&](const xt::xindex & aff_coord){

                // draw random number to check if we keep this edge
                if(draw() < lr_fraction) {
                    return;
                }

                // set the spatial coords from the offsets
                const auto & offset = offsets[aff_coord[0]];
                
                // set the spatial coordinates
                std::copy(aff_coord.begin() + 1, aff_coord.end(), coord.begin());
                ngb_coord = coord;

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

        void init_strides() {
            _strides.resize(_ndim);
            _strides[_ndim - 1] = 1;
            for(int d = _ndim - 2; d >= 0; --d) {
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
            auto nn_shape = affs.shape();
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
                std::cout << "1111111" << std::endl;
                init_lr(affs, offsets, lr_fraction);
            }
            else if(have_strides) {
                std::cout << "1111120" << std::endl;
                init_lr(affs, offsets, strides);
            }
            else {
                std::cout << "1111130" << std::endl;
                init_lr(affs, offsets);
            }
        }

        template<class AFFS>
        void init_lr(const AFFS & affs, const std::vector<OffsetType> & offsets) {
            // iterate over lr  neighbors, extract edges and weights
            xt::xindex coord(_ndim), ngb_coord(_ndim);
            const auto & aff_shape = affs.shape();
            util::for_each_coordinate(aff_shape, [&](const xt::xindex & aff_coord){

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
            const auto & aff_shape = affs.shape();
            xt::xindex coord(_ndim), ngb_coord(_ndim);
            util::for_each_coordinate(aff_shape, [&](const xt::xindex & aff_coord){

                // skip nn edges
                if(aff_coord[0] < _ndim) {
                    std::cout << aff_coord[0] << " , " << aff_coord[1]  << "= aff_coord[0] < _ndim = " << _ndim << std::endl;
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
            const auto & aff_shape = affs.shape();
            xt::xindex coord(_ndim), ngb_coord(_ndim);
            util::for_each_coordinate(aff_shape, [&](const xt::xindex & aff_coord){

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
        xt::xindex _shape;
        unsigned _ndim;
        std::size_t _n_nodes;
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
