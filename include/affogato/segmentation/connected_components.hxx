#include "boost/pending/disjoint_sets.hpp"
#include "xtensor/xtensor.hpp"
#include "affogato/util.hxx"


namespace affogato {
namespace segmentation {


    template<class AFFS, class LABELS>
    inline size_t connected_components(const xt::xexpression<AFFS> & affinities_exp,
                                       xt::xexpression<LABELS> & labels_exp,
                                       const float threshold) {

        typedef typename LABELS::value_type LabelType;
        const auto & affs = affinities_exp.derived_cast();
        auto & labels = labels_exp.derived_cast();

        const auto & shape = labels.shape();
        const unsigned dim = shape.size();

        // create and initialise union find
        const size_t n_nodes = labels.size();
        std::vector<LabelType> rank(n_nodes);
        std::vector<LabelType> parent(n_nodes);
        boost::disjoint_sets<LabelType*, LabelType*> sets(&rank[0], &parent[0]);
        for(LabelType node_id = 0; node_id < n_nodes; ++node_id) {
            sets.make_set(node_id);
        }

        // First pass:
        // iterate over each coordinate and create new label at coordinate
        // or assign representative of the neighbor label
        LabelType current_label = 0;
        util::for_each_coordinate(shape, [&](const xt::xindex & coord){

            // get the spatial part of the affinity coordinate
            xt::xindex aff_coord(affs.dimension());
            std::copy(coord.begin(), coord.end(), aff_coord.begin() + 1);

            // iterate over all neighbors with smaller coordiates
            // (corresponding to affinity neighbors) and collect the labels
            // if neighbors that are connected
            std::set<LabelType> ngb_labels;
            for(unsigned d = 0; d < dim; ++d) {
                xt::xindex ngb_coord = coord;
                ngb_coord[d] -= 1;
                // perform range check
                if(ngb_coord[d] < 0 || ngb_coord[d] >= shape[d]) {
                    continue;  // continue if out of range
                }
                // set the proper dimension in the affinity coordinate
                aff_coord[0] = d;
                // check if the neighbor is connected and appen its label if so
                if(affs[aff_coord] > threshold) {
                    ngb_labels.insert(labels[ngb_coord]);
                }
            }

            // check if we are connected to any of the neighbors
            // and if the neighbor labels need to be merged
            if(ngb_labels.size() == 0) {
                // no connection -> make new label @ current pixel
                labels[coord] = ++current_label;
            } else if (ngb_labels.size() == 1) {
                // only single label -> we assign its representative to the current pixel
                labels[coord] = sets.find_set(*ngb_labels.begin());
            } else {
                // multiple labels -> we merge them and assign representative to the current pixel
                std::vector<LabelType> tmp_labels(ngb_labels.begin(), ngb_labels.end());
                for(unsigned ii = 1; ii < tmp_labels.size(); ++ii) {
                    sets.link(tmp_labels[ii - 1], tmp_labels[ii]);
                }
                labels[coord] = sets.find_set(tmp_labels[0]);
            }
        });

        // Second pass:
        // Assign representative to each pixel
        util::for_each_coordinate(shape, [&](const xt::xindex & coord){
            labels[coord] = sets.find_set(labels[coord]);
        });

        // FIXME this is not necessarily the correct max value !!!
        return current_label;
    }
}
}
