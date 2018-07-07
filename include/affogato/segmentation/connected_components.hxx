#include "xtensor/xtensor.hpp"
#include "affogato/util.hxx"


namespace affogato {
namespace segmentation {

    // run bfs connecting all nodes that are linked
    // by edges with affinities larger than threshold
    // we assume that affinities have the offsets:
    // (2d)
    // [-1, 0]
    // [0, -1]
    // (3d)
    // [-1, 0, 0]
    // [0, -1, 0]
    // [0, 0, -1]
    template<class AFFS, class LABELS>
    inline void bfs(const xt::xexpression<AFFS> & affinities_exp,
                    xt::xexpression<LABELS> & labels_exp,
                    const xt::xindex & coord,
                    const typename LABELS::value_type current_label,
                    const float threshold) {
        const auto & affs = affinities_exp.derived_cast();
        auto & labels = labels_exp.derived_cast();
        // label the current pixel
        labels[coord] = current_label;

        const auto & shape = labels.shape();
        const unsigned dim = shape.size();
        // initialise the affinity coordinate at the current pixel position
        xt::xindex aff_coord(affs.dimension());
        std::copy(coord.begin(), coord.end(), aff_coord.begin() + 1);
        // iterate over the adjacent pixels
        for(unsigned d = 0; d < dim; ++d) {
            // get the affinity of the edge to
            // this adjacent pixel
            aff_coord[0] = d;
            const auto aff = affs[aff_coord];
            // check whether the pixels are connected
            // according to the threshold TODO < or >
            if(aff < threshold) {
                continue;  // continue if not connected
            }
            // get the coordinate of adjacent pixel
            xt::xindex next_coord = coord;
            --next_coord[d];
            // check if the pixel is out of range
            if(next_coord[d] < 0 || next_coord[d] > shape[d]) {
                continue;  // continue if out of range
            }
            // continue bfs from adjacent node
            bfs(affs, labels, next_coord, current_label, threshold);
        }
    }


    // compute connected components based on affinities
    template<class AFFS, class LABELS>
    inline size_t connected_components(const xt::xexpression<AFFS> & affinities_exp,
                                       xt::xexpression<LABELS> & labels_exp,
                                       const float threshold){
        //
        typedef typename LABELS::value_type LabelType;
        const auto affs = affinities_exp.derived_cast();
        auto & labels = labels_exp.derived_cast();

        //
        LabelType current_label = 1;
        xt::xindex shape(labels.shape().begin(), labels.shape().end());
        // iterate over the nodes (pixels), run bfs for each node
        // to label all connected nodes
        util::for_each_coordinate(shape, [&](const xt::xindex & coord){
            // don't do anything if this label is already labeled
            if(labels[coord] != 0) {
                return;
            }

            // run bfs beginning from the current node (pixel)
            bfs(affs, labels, coord, current_label, threshold);
            // increase the label
            ++current_label;
        });

        // return the max label
        return static_cast<size_t>(current_label - 1);
    }

}
}
