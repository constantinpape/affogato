#pragma once

#include <algorithm>
#include <unordered_map>
#include <map>

#include "boost/pending/disjoint_sets.hpp"
#include "xtensor.hpp"
#include "xtensor/views/xstrided_view.hpp"
#include "affogato/util.hxx"


namespace affogato {
namespace learning {

    template<class AFFS, class GT, class GRADS>
    double malis_gradient(const xt::xexpression<AFFS> & affs_exp,
                          const xt::xexpression<GT> & gt_exp,
                          xt::xexpression<GRADS> & grads_exp,
                          const std::vector<std::vector<int>> & offsets,
                          const bool pos) {
        const auto & affs = affs_exp.derived_cast();
        const auto & gt = gt_exp.derived_cast();
        auto & grads = grads_exp.derived_cast();

        typedef typename GT::value_type LabelType;
        typedef typename AFFS::value_type AffType;
        typedef typename GRADS::value_type GradType;
        // TODO check shapes
        const auto & shape = gt.shape();
        const auto & gt_strides = gt.strides();
        const unsigned nDim = gt.dimension();

        // nodes and edges in the affinity graph
        const size_t nNodes = gt.size();
        const size_t nEdges = affs.size();

        // make union find for the mst
        std::vector<LabelType> rank(nNodes);
        std::vector<LabelType> parent(nNodes);
        boost::disjoint_sets<LabelType*, LabelType*> sets(&rank[0], &parent[0]);

        // data structures for overlaps of nodes (= pixels) with gt labels
        // and sizes of gt segments
        std::vector<std::map<LabelType, size_t>> overlaps(nNodes);
        std::unordered_map<LabelType, size_t> segmentSizes;

        // initialize sets, overlaps and find labeled pixels
        size_t nNodesLabeled = 0, nPairPos = 0, nodeIndex = 0;
        util::for_each_coordinate(shape, [&](const xt::xindex & coord){
            const auto gtId = gt[coord];
            if(gtId != 0) {
                overlaps[nodeIndex].insert(std::make_pair(gtId, 1));
                ++segmentSizes[gtId];
                ++nNodesLabeled;
                nPairPos += (segmentSizes[gtId] - 1);
            }
            sets.make_set(nodeIndex);
            ++nodeIndex;
        });

        // compute normalisation
        const size_t nPairNorm = pos ? nPairPos : nNodesLabeled * (nNodesLabeled - 1) / 2 - nPairPos;
        if(nPairNorm == 0) {
            throw std::runtime_error("Normalization is zero!");
        }

        // sort the edges by affinity strength
        const auto flatView = xt::flatten(affs); // flat view
        std::vector<size_t> pqueue(nEdges);
        std::iota(pqueue.begin(), pqueue.end(), 0);

        // sort in increasing order
        std::sort(pqueue.begin(), pqueue.end(), [&flatView](const size_t ind1,
                                                            const size_t ind2){
            return flatView(ind1) > flatView(ind2);
        });

        // run kruskal and calculate mals gradeint for each edge in the spanning tree
        // initialize values
        const auto & strides = affs.strides();
        const auto & layout = affs.layout();
        LabelType setU, setV, nodeU, nodeV;
        // coordinates for affinities and gt
        xt::xindex affCoord(nDim + 1);
        xt::xindex gtCoordU(nDim), gtCoordV(nDim);

        double loss = 0;
        // iterate over the queue
        for(const auto edgeId : pqueue) {
            // translate edge id to coordinate
            const auto affCoord_ = xt::unravel_from_strides(edgeId, strides, layout);
            std::copy(affCoord_.begin(), affCoord_.end(), affCoord.begin());

            // get offset and spatial coordinates
            const auto & offset = offsets[affCoord[0]];
            // range check
            bool inRange = true;
            for(unsigned d = 0; d < nDim; ++d) {
                gtCoordU[d] = affCoord[d + 1];
                gtCoordV[d] = affCoord[d + 1] + offset[d];
                if(gtCoordV[d] < 0 || gtCoordV[d] >= shape[d]) {
                    inRange = false;
                    break;
                }
            }

            if(!inRange) {
                continue;
            }

            // get the spatial node index
            LabelType nodeU = 0, nodeV = 0;
            for(unsigned d = 0; d < nDim; ++d) {
                nodeU += gtCoordU[d] * gt_strides[d];
                nodeV += gtCoordV[d] * gt_strides[d];
            }

            // get the representatives of the nodes
            LabelType setU = sets.find_set(nodeU);
            LabelType setV = sets.find_set(nodeV);

            // check if the nodes are not merged yet
            if(setU != setV) {

                // merge nodes
                sets.link(nodeU, nodeV);
                // sets.link(setU, setV);

                // initialize values for gradient calculation
                GradType currentGradient = 0;
                const AffType aff = affs[affCoord];
                const GradType grad = pos ? 1. - aff : -aff;

                // compute the number of node pairs merged by this edge
                for(auto itU = overlaps[setU].begin(); itU != overlaps[setU].end(); ++itU) {
                    for(auto itV = overlaps[setV].begin(); itV != overlaps[setV].end(); ++itV) {
                        const size_t nPair = itU->second * itV->second;
                        if(pos && (itU->first == itV->first)) {
                            loss += grad * grad * nPair;
                            currentGradient += grad * nPair;
                        }

                        if(!pos && (itU->first != itV->first)) {
                            loss += grad * grad * nPair;
                            currentGradient += grad * nPair;
                        }
                    }
                }
                grads[affCoord] += currentGradient / nPairNorm;

                if(sets.find_set(setU) == setV) {
                    std::swap(setU, setV);
                }

                auto & overlapsU = overlaps[setU];
                auto & overlapsV = overlaps[setV];
                auto itV = overlapsV.begin();
                while(itV != overlapsV.end()) {
                    auto itU = overlapsU.find(itV->first);
                    if(itU == overlapsU.end()) {
                        overlapsU.insert(std::make_pair(itV->first, itV->second));
                    } else {
                        itU->second += itV->second;
                    }
                    overlapsV.erase(itV);
                    ++itV;
                }
            }
        }
        return loss / nPairNorm;
    }


    // for constrained malis, we compute the gradients in 2 passes:
    // in the positive pass, ...
    // in the negative pass, ...
    template<class AFFS, class GT, class GRADS>
    double constrained_malis(const xt::xexpression<AFFS> & affs_exp,
                             const xt::xexpression<GT> & gt_exp,
                             xt::xexpression<GRADS> & grads_exp,
                             const std::vector<std::vector<int>> & offsets) {
        typedef typename AFFS::value_type AffType;
        typedef typename GT::value_type LabelType;

        const auto & affs = affs_exp.derived_cast();
        const auto & gt = gt_exp.derived_cast();
        auto & grads = grads_exp.derived_cast();

        const auto & shape = gt.shape();
        const unsigned ndim = shape.size();
        // init affinities for the positive and negative pass
        const auto & aff_shape = affs.shape();
        typedef typename xt::xarray<AffType>::shape_type ShapeType;
        ShapeType out_shape(aff_shape.begin(), aff_shape.end());
        xt::xarray<AffType> affs_pos(out_shape);
        xt::xarray<AffType> affs_neg(out_shape);

        // initialize spatial coordinates
        xt::xindex coord_u(ndim), coord_v(ndim);

        // iterate over all affinities and get values for negative and positive pass
        // positive pass: aff = min(aff, gt_affinity)
        // negative pass: aff = max(aff, gt_affinity)
        // we can compute the gt affinity (which is either 0 or 1, implicitly)
        util::for_each_coordinate(aff_shape, [&](const xt::xindex & aff_coord){
            const auto aff = affs[aff_coord];
            const auto & offset = offsets[aff_coord[0]];

            // get the spatial coordinates
            // range check
            bool inRange = true;
            for(unsigned d = 0; d < ndim; ++d) {
                coord_u[d] = aff_coord[d + 1];
                coord_v[d] = aff_coord[d + 1] + offset[d];
                if(coord_v[d] < 0 || coord_v[d] >= shape[d]) {
                    inRange = false;
                    break;
                }
            }

            if(!inRange) {
                return;
            }

            const LabelType l_u = gt[coord_u];
            const LabelType l_v = gt[coord_v];

            // check whether this is a connecting or separating
            // affinity edge
            if(l_u != l_v || l_u == 0 || l_v == 0) {
                // seperating (or ignore) edge -> gt affinity is 0
                // -> aff_pos = min(aff, gt_aff) = 0
                affs_pos[aff_coord] = 0.;
                affs_neg[aff_coord] = aff;
            } else {
                // connecting edge -> gt affinity is 1
                // -> aff_neg = max(aff, gt_aff) = 1
                affs_pos[aff_coord] = aff;
                affs_neg[aff_coord] = 1.;
            }

        });

        // run positive and negative pass
        const double loss_pos = malis_gradient(affs_pos, gt, grads, offsets, true);
        const double loss_neg = malis_gradient(affs_neg, gt, grads, offsets, false);

        // normalize and invert the gradients
        grads /= -2.;

        return - (loss_pos + loss_neg) / 2.;
    }

}
}
