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

        template<class SEEDS>
        inline void set_seeds(const SEEDS & seeds) {
            init_seeds(seeds);
        }

        inline void clear_seeds() {
            _seeded_nodes.clear();
        }

        // TODO implement
        // initialize the nearest neighbor graph
        inline void init_nn_graph() {

        }

        template<class AFFS>
        inline void compute_nh_and_weights(const AFFS & affs, const std::vector<OffsetType> & offsets,
                                           const std::vector<std::size_t> & strides, const bool randomize_strides,
                                           std::vector<std::pair<uint64_t, uint64_t>> & uv_ids, std::vector<float> & weights) const {
            // check if we have strides
            const std::size_t stride_product = std::accumulate(strides.begin(), strides.end(), 1,
                                                               std::multiplies<std::size_t>());
            const bool have_strides = stride_product > 1;

            // call appropriate initializer
            if(have_strides && randomize_strides) {
                // compute fraction of lr edges that we keep
                const double fraction = 1. / stride_product;
                compute_nh_and_weights_impl(affs, offsets, fraction,
                                            uv_ids, weights);
            }
            else if(have_strides) {
                compute_nh_and_weights_impl(affs, offsets, strides,
                                            uv_ids, weights);
            }
            else {
                compute_nh_and_weights_impl(affs, offsets,
                                            uv_ids, weights);
            }

        }

        inline uint64_t get_node(const xt::xindex & coord) const {
            uint64_t node = 0;
            for(unsigned d = 0; d < _ndim; ++d) {
                node += coord[d] * _strides[d];
            }
            return node;
        }

        inline uint64_t get_node(const std::vector<int64_t> & coord) const {
            uint64_t node = 0;
            for(unsigned d = 0; d < _ndim; ++d) {
                node += coord[d] * _strides[d];
            }
            return node;
        }

        inline std::vector<int64_t> get_coordinate(const uint64_t node) const {
            std::vector<int64_t> coord(_ndim);
            uint64_t index = node;
            for(auto d = 0; d < _ndim; ++d) {
                coord[d] = index / _strides[d];
                index -= coord[d] * _strides[d];
            }
            return coord;
        }

        std::size_t n_nodes() const {
            return _n_nodes;
        }

        unsigned ndim() const {
            return _ndim;
        }

        // TODO we need to generalize this to support more than one time-step back ! than labels
        // would be 1 dim bigger than _dim
        template<class AFFS, class LABELS>
        void get_causal_edges(const AFFS & affs, const LABELS & labels,
                              const std::vector<OffsetType> & offsets,
                              std::vector<EdgeType> & uv_ids,
                              std::vector<float> & weights) const {
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
            _masked_nodes.resize(_n_nodes);
            util::for_each_coordinate(_shape, [&](const xt::xindex & coord){
                _masked_nodes[get_node(coord)] = mask[coord] == 0;
            });
        }

        template<class SEEDS>
        void init_seeds(const SEEDS & seeds) {
            _seeded_nodes.resize(_n_nodes);
            util::for_each_coordinate(_shape, [&](const xt::xindex & coord){
                const uint64_t seed_id = seeds[coord];
                if(seed_id == 0) {
                    return;
                }
                if(seed_id < _n_nodes) {
                    std::cout << seed_id << " / " << _n_nodes << std::endl;
                    throw std::runtime_error("Seeds need to have offset");
                }
                _seeded_nodes[get_node(coord)] = seed_id;
            });
        }


        template<class AFFS>
        inline void insertEdge(const AFFS & affs, const xt::xindex & aff_coord, const OffsetType & offset,
                               std::vector<std::pair<uint64_t, uint64_t>> & uv_ids, std::vector<float> & weights) const{
            // initialize the spatial coordinates
            xt::xindex coord(_ndim);
            std::copy(aff_coord.begin() + 1, aff_coord.end(), coord.begin());
            xt::xindex ngb_coord = coord;

            // update the ngb-coord and check that we are in range
            for(unsigned d = 0; d < _ndim; ++d) {
                ngb_coord[d] += offset[d];
                if(ngb_coord[d] < 0 || ngb_coord[d] >= _shape[d]) {
                    return;
                }
            }

            uint64_t u = get_node(coord);
            uint64_t v = get_node(ngb_coord);

            // if we have a mask,
            // check if any of the nodes is masked
            if(_masked_nodes.size()) {
                if(_masked_nodes[u] || _masked_nodes[v]) {
                    return;
                }
            }

            // if we have seeds, check if any of the nodes
            // is seeded
            bool haveSeed = false;
            if(_seeded_nodes.size()) {
                if(_seeded_nodes[u]) {
                    haveSeed = true;
                    u = _seeded_nodes[u];
                }
                if(_seeded_nodes[v]) {
                    haveSeed = true;
                    v = _seeded_nodes[v];
                }
            }

            if(u > v) {
                std::swap(u, v);
            }

            // if we have a seed, we need to check if we had this edge already
            if(haveSeed) {
                auto uv = std::make_pair(u, v);
                auto uv_it = std::find(uv_ids.begin(), uv_ids.end(), uv);
                // we had this edge already -> update edge strength and skip
                if(uv_it != uv_ids.end()) {
                    const std::size_t edgeId = std::distance(uv_ids.begin(), uv_it);
                    weights[edgeId] = std::max(affs[aff_coord], weights[edgeId]);
                    return;
                }
            }
            uv_ids.emplace_back(u, v);
            weights.emplace_back(affs[aff_coord]);
        }


        template<class AFFS>
        void compute_nh_and_weights_impl(const AFFS & affs, const std::vector<OffsetType> & offsets,
                                         std::vector<std::pair<uint64_t, uint64_t>> & uv_ids, std::vector<float> & weights) const {
            const auto & aff_shape = affs.shape();
            // iterate over  neighbors, extract edges and weights
            util::for_each_coordinate(aff_shape, [&](const xt::xindex & aff_coord){
                // set the ngb coord from the offsets
                const auto & offset = offsets[aff_coord[0]];
                // insert the edge and weight corresponding to this grid connection
                insertEdge(affs, aff_coord, offset, uv_ids, weights);
            });
        }

        template<class AFFS>
        void compute_nh_and_weights_impl(const AFFS & affs, const std::vector<OffsetType> & offsets,
                                         const double fraction, std::vector<std::pair<uint64_t, uint64_t>> & uv_ids,
                                         std::vector<float> & weights) const {

            //
            std::default_random_engine generator;
            std::uniform_real_distribution<float> distribution;
            auto draw = std::bind(distribution, generator);

            // iterate over lr  neighbors, extract edges and weights
            const auto & aff_shape = affs.shape();
            util::for_each_coordinate(aff_shape, [&](const xt::xindex & aff_coord){

                // keep edge if random number is below fraction
                if(draw() > fraction) {
                    return;
                }

                // set the ngb coord from the offsets
                const auto & offset = offsets[aff_coord[0]];
                // insert the edge and weight corresponding to this grid connection
                insertEdge(affs, aff_coord, offset, uv_ids, weights);
            });
        }

        template<class AFFS>
        void compute_nh_and_weights_impl(const AFFS & affs, const std::vector<OffsetType> & offsets,
                                         const std::vector<std::size_t> & strides, std::vector<std::pair<uint64_t, uint64_t>> & uv_ids,
                                         std::vector<float> & weights) const {

            // iterate over lr  neighbors, extract edges and weights
            const auto & aff_shape = affs.shape();
            xt::xindex coord(_ndim), ngb_coord(_ndim);
            util::for_each_coordinate(aff_shape, [&](const xt::xindex & aff_coord){

                for(unsigned d = 1; d < _ndim + 1; ++d) {
                    if(aff_coord[d] % strides[d - 1] != 0) {
                        return;
                    }
                }

                // set the ngb coord from the offsets
                const auto & offset = offsets[aff_coord[0]];
                // insert the edge and weight corresponding to this grid connection
                insertEdge(affs, aff_coord, offset, uv_ids, weights);
            });
        }


    private:
        xt::xindex _shape;
        unsigned _ndim;
        std::size_t _n_nodes;
        xt::xindex _strides;
        std::vector<bool> _masked_nodes;
        std::vector<uint64_t> _seeded_nodes;
        // TODO need a datastructure for the nn graph edges
        // std::vector<EdgeType> _uv_ids;
    };


}
}
