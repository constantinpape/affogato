#pragma once
#include <unordered_map>
#include <vector>
#include "xtensor/xtensor.hpp"
#include "affogato/util.hxx"


namespace affogato {
namespace affinities {

    // typedefs
    typedef std::unordered_map<uint64_t, size_t> Histogram;
    typedef std::vector<Histogram> HistogramStorage;


    template<class LABELS, class COORD>
    inline size_t compute_histogram(const LABELS & labels,
                                    const COORD & block_begin,
                                    const COORD & block_end,
                                    Histogram & out) {
        typedef typename LABELS::value_type LabelType;
        size_t n_pixels = 0;
        util::for_each_coordinate(block_begin, block_end, [&](const xt::xindex & coord){
            const LabelType label = labels[coord];
            auto out_it = out.find(label);
            if(out_it == out.end()) {
                out.insert(out_it, std::make_pair(label, 1));
            } else {
                ++out_it->second;
            }
            ++n_pixels;
        });
        return n_pixels;
    }


    template<class LABELS, class COORD>
    inline size_t compute_histogram_with_ignore_label(const LABELS & labels,
                                                      const COORD & block_begin,
                                                      const COORD & block_end,
                                                      Histogram & out,
                                                      const uint64_t ignore_label) {
        typedef typename LABELS::value_type LabelType;
        size_t n_pixels = 0;
        util::for_each_coordinate(block_begin, block_end, [&](const xt::xindex & coord){
            const uint64_t label = labels[coord];
            if(label == ignore_label) {
                return;
            }

            auto out_it = out.find(label);
            if(out_it == out.end()) {
                out.insert(out_it, std::make_pair(label, 1));
            } else {
                ++out_it->second;
            }
            ++n_pixels;
        });
        return n_pixels;
    }


    inline double compute_single_affinity(const Histogram & histo, const Histogram & ngb_histo,
                                        const double norm) {
        double aff = 0.;
        for(auto it = histo.begin(); it != histo.end(); ++it){
            auto ngbIt = ngb_histo.find(it->first);
            if (ngbIt != ngb_histo.end()) {
                aff += it->second * ngbIt->second;
            }
        }
        return aff / norm;
    }


    inline size_t get_block_index(const xt::xindex & block_coord,  const xt::xindex & block_strides) {
        size_t block_id = 0;
        for(unsigned d = 0; d < block_coord.size(); ++d) {
            block_id += block_coord[d] * block_strides[d];
        }
        return block_id;
    }



    template<class LABEL, class AFFS, class MASK>
    void compute_multiscale_affinities(const xt::xexpression<LABEL> & labels_exp,
                                       const std::vector<int> & block_shape,
                                       xt::xexpression<AFFS> & affs_exp,
                                       xt::xexpression<MASK> & mask_exp,
                                       const bool have_ignore_label=false,
                                       const uint64_t ignore_label=0) {

        const auto & labels = labels_exp.derived_cast();
        auto & affs = affs_exp.derived_cast();
        auto & mask = mask_exp.derived_cast();

        // compute the block sizes
        const unsigned ndim = labels.dimension();
        xt::xindex shape(ndim), blocks_per_axis(ndim);
        for(unsigned d = 0; d < ndim; ++d) {
            blocks_per_axis[d] = affs.shape()[d + 1];
            shape[d] = labels.shape()[d];
        }

        // compute the total number of blocks
        const size_t n_blocks = std::accumulate(blocks_per_axis.begin(), blocks_per_axis.end(), 1,
                                                std::multiplies<size_t>());

        // compute the block strides, e.g. {blocks_per_axis[1] * blocks_per_axis[2], blocks_per_axis[2], 1}
        // for 3 D
        xt::xindex block_strides(ndim);
        for(int d = ndim - 1; d >= 0; --d) {
            block_strides[d] = (d == ndim - 1) ? 1 : block_strides[d + 1] * blocks_per_axis[d + 1];
        }

        // compute the histograms for each block
        HistogramStorage histograms(n_blocks);
        std::vector<size_t> block_sizes(n_blocks, 1);
        util::for_each_coordinate(blocks_per_axis, [&](const xt::xindex & block_coord){
            // get the 1d block index
            const size_t block_id = get_block_index(block_coord, block_strides);
            // get the begin and end coordinates of this block
            xt::xindex block_begin(ndim), block_end(ndim);
            for(unsigned d = 0; d < ndim; ++d) {
                block_begin[d] = block_coord[d] * block_shape[d];
                block_end[d] = std::min((block_coord[d] + 1) * block_shape[d], shape[d]);
            }

            size_t & block_size = block_sizes[block_id];
             if(have_ignore_label){
                 block_size = compute_histogram_with_ignore_label(labels, block_begin, block_end,
                                                                  histograms[block_id], ignore_label);
             }
             else {
                 block_size = compute_histogram(labels, block_begin, block_end, histograms[block_id]);
             }

        });

        // compute the affinties
        util::for_each_coordinate(blocks_per_axis, [&](const xt::xindex & block_coord){

            const size_t block_id = get_block_index(block_coord, block_strides);
            const auto & histo = histograms[block_id];
            const size_t block_size = block_sizes[block_id];

            xt::xindex aff_coord(ndim + 1);
            std::copy(block_coord.begin(), block_coord.end(), aff_coord.begin() + 1);
            for(unsigned d = 0; d < ndim; ++d) {
                aff_coord[0] = d;
                if(block_coord[d] > 0) {
                    xt::xindex ngb_coord = block_coord;
                    --ngb_coord[d];
                    const size_t ngb_id = get_block_index(ngb_coord, block_strides);
                    const auto & ngb_histo = histograms[ngb_id];
                    const double norm = block_size * block_sizes[ngb_id];
                    if(norm > 0) {
                        affs[aff_coord] = compute_single_affinity(histo, ngb_histo, norm);
                        mask[aff_coord] = 1;
                    } else {
                        mask[aff_coord] = 0;
                    }
                } else {
                    mask[aff_coord] = 0;
                }
            }
        });
    }
}
}
