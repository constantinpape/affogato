#pragma once

#include <random>
#include <boost/functional/hash.hpp>
#include "xtensor/xtensor.hpp"
#include "affogato/util.hxx"


namespace affogato {
namespace segmentation {


    class MWSGridGraph {

    public:
        typedef std::pair<uint64_t, uint64_t> EdgeType;
        typedef std::vector<int> OffsetType;
        typedef std::unordered_map<std::pair<uint64_t, uint64_t>,
                                   float,
                                   boost::hash<std::pair<uint64_t, uint64_t>>
                                  > SeedStateType;

        MWSGridGraph(const std::vector<std::size_t> & shape): _shape(shape.begin(), shape.end()),
                                                              _ndim(shape.size()),
                                                              _n_nodes(std::accumulate(shape.begin(),
                                                                                       shape.end(),
                                                                                       1, std::multiplies<std::size_t>())),
                                                              _same_seed_weight(0.),
                                                              _different_seed_weight(0.){
            init_strides();
        }

        //
        // mask functionality
        //

        template<class MASK>
        inline void set_mask(const MASK & mask) {
            init_mask(mask);
        }

        inline void clear_mask() {
            _masked_nodes.clear();
        }

        //
        // seed functionality
        //

        // TODO allow for ROI
        template<class SEEDS>
        void update_seeds(const SEEDS & seeds) {
            util::for_each_coordinate(_shape, [&](const xt::xindex & coord){
                const uint64_t seed_id = seeds[coord];
                // 0 means un-seeded
                if(seed_id == 0) {
                    return;
                }
                const uint64_t this_node = get_node(coord);
                _seed_ids.insert(seed_id);

                // set this node to the representative node
                _seeded_nodes[this_node] = seed_id;
            });
        }

        inline void clear_seeds() {
            _seeded_nodes.clear();
        }

        inline float get_same_seed_weight() const {
            return _same_seed_weight;
        }

        inline void set_same_seed_weight(const float weight) {
            _same_seed_weight = weight;
        }

        inline float get_different_seed_weight() const {
            return _different_seed_weight;
        }

        inline void set_different_seed_weight(const float weight) {
            _different_seed_weight = weight;
        }

        //
        //
        //

        // STATE = std::unordered_map<std::pair<uint64_t, uint64_t>, std::pair<float, bool>>
        // + boost hash function for pair
        template<class AFFS, class SEG, class STATE>
        inline void compute_state_for_segmentation(const AFFS & affs, const SEG & seg,
                                                   const std::vector<OffsetType> & offsets,
                                                   const unsigned n_attactive_channels,
                                                   const bool ignore_label,
                                                   STATE & state) const {

            typedef typename SEG::value_type SegType;

            const auto & aff_shape = affs.shape();
            xt::xindex coord(_ndim), ngb_coord(_ndim);
            // Iterate over the affinities and extract the segmentation state
            util::for_each_coordinate(aff_shape, [&](const xt::xindex & aff_coord) {

                // get the spatial coords
                std::copy(aff_coord.begin() + 1, aff_coord.end(), coord.begin());
                ngb_coord = coord;

                // update the ngb-coord and check that we are in range
                const auto & offset = offsets[aff_coord[0]];
                for(unsigned d = 0; d < _ndim; ++d) {
                    ngb_coord[d] += offset[d];
                    if(ngb_coord[d] < 0 || ngb_coord[d] >= _shape[d]) {
                        return;
                    }
                }

                // get the seg ids
                SegType u = seg[coord];
                SegType v = seg[coord];

                // skip if we have an ignore label
                if((u == 0 || v == 0) && ignore_label) {
                    return;
                }

                // TODO should we also skip masked nodes here ?
                // if(_masked_nodes.size()) {
                //  if(_maksed_nodes[u] || _masked_nodes[v]) {
                //    return;
                //   }
                // }

                if(u > v) {
                    std::swap(u, v);
                }

                const auto uv = std::make_pair(u, v);
                const auto edge_it = state.find(uv);

                const bool is_attractive = aff_coord[0] < n_attactive_channels;
                const float weight = affs[aff_coord];

                // check if we had this edge already
                if(edge_it != state.end()) { // yes -> update the edge state
                    // TODO support other update than max
                    auto & edge_state = edge_it->second;
                    if(weight > edge_state.first) {
                        edge_state.first = weight;
                        edge_state.second = is_attractive;
                    }
                }
                else { // no -> insert new edge state
                    state.emplace(uv, std::make_pair(weight, is_attractive));
                }
            });
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

        std::size_t n_seeds() const {
            return _seed_ids.size();
        }

        // FIXME
        template<class NODE_LABELS, class SEED_ASSIGNMENTS>
        inline void get_seed_assignments_from_node_labels(const NODE_LABELS & node_labels,
                                                          SEED_ASSIGNMENTS & seed_assignments) const {
            /*
            std::size_t seed_index = 0;
            for(const uint64_t seed_id : _seed_lut) {
                const uint64_t seed_id = seed_it.first;
                const uint64_t representative_node = seed_it.second;
                seed_assignments(seed_index, 0) = seed_id;
                seed_assignments(seed_index, 1) = node_labels(representative_node);
                ++seed_index;
            }
            */
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

        template<class AFFS>
        inline void insert_edge(const AFFS & affs, const xt::xindex & aff_coord, const OffsetType & offset,
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

            if(u > v) {
                std::swap(u, v);
            }
            uv_ids.emplace_back(u, v);
            weights.emplace_back(affs[aff_coord]);
        }


        template<class AFFS>
        void compute_nh_and_weights_impl(const AFFS & affs, const std::vector<OffsetType> & offsets,
                                         std::vector<std::pair<uint64_t, uint64_t>> & uv_ids, std::vector<float> & weights) const {
            const auto & aff_shape = affs.shape();
            // first, compute the weights from the affinities with the given offset pattern
            util::for_each_coordinate(aff_shape, [&](const xt::xindex & aff_coord){
                // set the ngb coord from the offsets
                const auto & offset = offsets[aff_coord[0]];
                // insert the edge and weight corresponding to this grid connection
                insert_edge(affs, aff_coord, offset, uv_ids, weights);
            });

            // if we have seeds, insert all edges between the seeds
            if(_seeded_nodes.size()) {
                update_edges_from_seeds(uv_ids, weights);
            }
        }

        template<class AFFS>
        void compute_nh_and_weights_impl(const AFFS & affs, const std::vector<OffsetType> & offsets,
                                         const double fraction, std::vector<std::pair<uint64_t, uint64_t>> & uv_ids,
                                         std::vector<float> & weights) const {

            const auto & aff_shape = affs.shape();
            // random number generator
            std::default_random_engine generator;
            std::uniform_real_distribution<float> distribution;
            auto draw = std::bind(distribution, generator);

            // first, compute the weights from the affinities with the given offset pattern
            util::for_each_coordinate(aff_shape, [&](const xt::xindex & aff_coord){
                // keep edge if random number is below fraction
                if(draw() > fraction) {
                    return;
                }
                // set the ngb coord from the offsets
                const auto & offset = offsets[aff_coord[0]];
                // insert the edge and weight corresponding to this grid connection
                insert_edge(affs, aff_coord, offset, uv_ids, weights);
            });

            // if we have seeds, insert all edges between the seeds
            if(_seeded_nodes.size()) {
                update_edges_from_seeds(uv_ids, weights);
            }
        }

        template<class AFFS>
        void compute_nh_and_weights_impl(const AFFS & affs, const std::vector<OffsetType> & offsets,
                                         const std::vector<std::size_t> & strides, std::vector<std::pair<uint64_t, uint64_t>> & uv_ids,
                                         std::vector<float> & weights) const {
            const auto & aff_shape = affs.shape();

            util::for_each_coordinate(aff_shape, [&](const xt::xindex & aff_coord){
                // check if this coordinate is in the strides
                for(unsigned d = 1; d < _ndim + 1; ++d) {
                    if(aff_coord[d] % strides[d - 1] != 0) {
                        return;
                    }
                }
                // set the ngb coord from the offsets
                const auto & offset = offsets[aff_coord[0]];
                // insert the edge and weight corresponding to this grid connection
                insert_edge(affs, aff_coord, offset, uv_ids, weights);
            });

            // if we have seeds, insert all edges between the seeds
            if(_seeded_nodes.size()) {
                update_edges_from_seeds(uv_ids, weights);
            }
        }

        inline void update_edges_from_seeds(std::vector<std::pair<uint64_t, uint64_t>> & uv_ids,
                                            std::vector<float> & weights) const {
            // get the seeded node ids from the _seeded_nodes
            std::vector<uint64_t> node_ids(_seeded_nodes.size());
            std::size_t id = 0;
            for(const auto & seedPair : _seeded_nodes) {
                node_ids[id] = seedPair.first;
                ++id;
            }

            // iterate over the cartesian product of the seeded ids and insert
            // edges according to the seed states
            for(std::size_t i = 0; i < node_ids.size(); ++i) {
                for(std::size_t j = i + 1; j < node_ids.size(); ++j) {
                    //
                    uint64_t u = node_ids[i];
                    uint64_t v = node_ids[j];
                    if(u > v) {
                        std::swap(u, v);
                    }

                    const uint64_t sU = _seeded_nodes.at(u);
                    const uint64_t sV = _seeded_nodes.at(v);
                    // are we in the same or in different seeds
                    const float w = (sU == sV) ? _same_seed_weight : _different_seed_weight;

                    // check if we have this edge already
                    auto uv = std::make_pair(u, v);
                    auto edge_it = std::find(uv_ids.begin(), uv_ids.end(), uv);
                    if(edge_it != uv_ids.end()) {
                        weights[std::distance(uv_ids.begin(), edge_it)] = w;
                    } else {
                        uv_ids.emplace_back(u, v);
                        weights.emplace_back(w);
                    }
                }
            }
        }


    private:
        xt::xindex _shape;
        unsigned _ndim;
        std::size_t _n_nodes;
        float _same_seed_weight;
        float _different_seed_weight;
        xt::xindex _strides;

        // data structures for masks
        std::vector<bool> _masked_nodes;
        // data structures for seeds
        std::unordered_map<uint64_t, uint64_t> _seeded_nodes;
        std::set<uint64_t> _seed_ids;
    };


}
}
