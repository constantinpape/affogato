#pragma once
#include <unordered_map>
#include <vector>

#include "xtensor/xtensor.hpp"
#include "affogato/util.hxx"


namespace affogato {
namespace affinities {

    template<class LABELS, class AFFS, class MASK>
    void compute_affinities(const xt::xexpression<LABELS> & labels_exp,
                            const std::vector<std::vector<int>> & offsets,
                            xt::xexpression<AFFS> & affs_exp,
                            xt::xexpression<MASK> & mask_exp,
                            const bool have_ignore_label=false,
                            const uint64_t ignore_label=0) {
        const auto & labels = labels_exp.derived_cast();
        auto & affs = affs_exp.derived_cast();
        auto & mask = mask_exp.derived_cast();

        typedef typename AFFS::value_type AffinityType;
        typedef typename LABELS::value_type LabelType;

        // get the number of dimensions, and the shapes of
        // labels and affinities
        const unsigned ndim = labels.dimension();
        const xt::xindex shape(labels.shape().begin(), labels.shape().end());
        // check that dimensions agree
        if(ndim + 1 != affs.dimension()) {
            throw std::runtime_error("Dimensions of labels and affinities do not agree.");
        }
        // affinity shape
        const xt::xindex aff_shape(affs.shape().begin(), affs.shape().end());
        // TODO check shapes !

        const int64_t n_channels = offsets.size();
        // check that number of channels do agree
        if(n_channels != aff_shape[0]) {
            throw std::runtime_error("Number of channels in affinities and offsets do not agree.");
        }

        // initialize coordinates
        xt::xindex coord(ndim), ngb_coord(ndim);

        // get the affinity and mask value for each affinity coordinate
        util::for_each_coordinate(aff_shape, [&](const xt::xindex & aff_coord){

            // set the spatial coordinates
            std::copy(aff_coord.begin() + 1, aff_coord.end(), coord.begin());
            ngb_coord = coord;
            // set the spatial coords from the offsets
            const auto & offset = offsets[aff_coord[0]];

            bool out_of_range = false;
            for(unsigned d = 0; d < ndim; ++d) {
                ngb_coord[d] += offset[d];
                if(ngb_coord[d] < 0 || ngb_coord[d] >= shape[d]) {
                    out_of_range = true;
                    mask[aff_coord] = 0;
                    break;
                }
            }

            if(out_of_range) {
                return;
            }

            const LabelType label = labels[coord];
            const LabelType label_ngb = labels[ngb_coord];

            if(have_ignore_label) {
                if(label == ignore_label || label_ngb == ignore_label) {
                    mask[aff_coord] = 0;
                    return;
                }
            }

            affs[aff_coord] = static_cast<AffinityType>(label == label_ngb);
            mask[aff_coord] = 1;
        });
    }

    template<class LABELS, class AFFS, class MASK>
    void compute_affinities_with_glia(const xt::xexpression<LABELS> & labels_exp,
                            const std::vector<std::vector<int>> & offsets,
                            xt::xexpression<AFFS> & affs_exp,
                            xt::xexpression<MASK> & mask_exp,
                            const bool have_ignore_label=false,
                            const bool have_boundary_label=false,
                            const bool have_glia_label=false,
                            const uint64_t ignore_label=0,
                            const int64_t boundary_label=-1,
                            const int64_t glia_label=-2
                            ) {
        const auto & labels = labels_exp.derived_cast();
        auto & affs = affs_exp.derived_cast();
        auto & mask = mask_exp.derived_cast();

        typedef typename AFFS::value_type AffinityType;
        typedef typename LABELS::value_type LabelType;

        // get the number of dimensions, and the shapes of
        // labels and affinities
        const unsigned ndim = labels.dimension();
        const xt::xindex shape(labels.shape().begin(), labels.shape().end());
        // check that dimensions agree
        if(ndim + 1 != affs.dimension()) {
            throw std::runtime_error("Dimensions of labels and affinities do not agree.");
        }
        // affinity shape
        const xt::xindex aff_shape(affs.shape().begin(), affs.shape().end());
        // TODO check shapes !

        const int64_t n_channels = offsets.size();
        // check that number of channels do agree
        if(n_channels != aff_shape[0]) {
            throw std::runtime_error("Number of channels in affinities and offsets do not agree.");
        }

        // initialize coordinates
        xt::xindex coord(ndim), ngb_coord(ndim);

        // get the affinity and mask value for each affinity coordinate
        util::for_each_coordinate(aff_shape, [&](const xt::xindex & aff_coord){

            // set the spatial coordinates
            std::copy(aff_coord.begin() + 1, aff_coord.end(), coord.begin());
            ngb_coord = coord;
            // set the spatial coords from the offsets
            const auto & offset = offsets[aff_coord[0]];

            bool out_of_range = false;
            for(unsigned d = 0; d < ndim; ++d) {
                ngb_coord[d] += offset[d];
                if(ngb_coord[d] < 0 || ngb_coord[d] >= shape[d]) {
                    out_of_range = true;
                    mask[aff_coord] = 0;
                    break;
                }
            }

            if(out_of_range) {
                return;
            }

            const LabelType label = labels[coord];
            const LabelType label_ngb = labels[ngb_coord];

            if(have_ignore_label) {
                if(label == ignore_label || label_ngb == ignore_label) {
                    mask[aff_coord] = 0;
                    return;
                }
            }
            if(have_glia_label) {
                if(label == glia_label && label_ngb == glia_label) {
                    mask[aff_coord] = 0;
                    return;
                }
            }
            if(have_boundary_label) {
                if(label == boundary_label || label_ngb == boundary_label) {
                    affs[aff_coord] = static_cast<AffinityType>(false);
                    mask[aff_coord] = 1;
                    return;
                }
            }
            affs[aff_coord] = static_cast<AffinityType>(label == label_ngb);
            mask[aff_coord] = 1;


        });
    }


}
}
