#pragma once
#include <unordered_map>
#include <vector>
#include <cmath>

#include "xtensor/xtensor.hpp"
#include "affogato/util.hxx"


namespace affogato {
namespace affinities {


    template<class VALUES, class DISTANCES, class F>
    void compute_embedding_distances_impl(const xt::xexpression<VALUES> & values_exp,
                                          const std::vector<std::vector<int>> & offsets,
                                          xt::xexpression<DISTANCES> & distances_exp,
                                          F && norm) {
        const auto & values = values_exp.derived_cast();
        auto & distances = distances_exp.derived_cast();

        typedef typename DISTANCES::value_type DistanceType;
        typedef typename VALUES::value_type ValueType;

        // get the number of dimensions, and the shapes of
        // values and distances
        const unsigned ndim = values.dimension() - 1;
        const xt::xindex shape(values.shape().begin() + 1, values.shape().end());
        // check that dimensions agree
        if(ndim + 1 != distances.dimension()) {
            throw std::runtime_error("Dimensions of values and distances do not agree.");
        }
        // distances shape
        const xt::xindex dist_shape(distances.shape().begin(), distances.shape().end());

        const int64_t n_channels = offsets.size();
        // check that number of channels do agree
        if(n_channels != dist_shape[0]) {
            throw std::runtime_error("Number of channels in distances and offsets do not agree.");
        }

        // initialize coordinates
        xt::xindex coord(ndim), ngb_coord(ndim);

        util::for_each_coordinate(dist_shape, [&](const xt::xindex & dist_coord){

            // set the spatial coordinates
            std::copy(dist_coord.begin() + 1, dist_coord.end(), coord.begin());
            ngb_coord = coord;
            // set the spatial coords from the offsets
            const auto & offset = offsets[dist_coord[0]];

            bool out_of_range = false;
            for(unsigned d = 0; d < ndim; ++d) {
                ngb_coord[d] += offset[d];
                if(ngb_coord[d] < 0 || ngb_coord[d] >= shape[d]) {
                    out_of_range = true;
                    break;
                }
            }
            if(out_of_range) {
                return;
            }

            // compute distance in embedding space via functor for different norms
            distances[dist_coord] = norm(values, coord, ngb_coord);
        });
    }

}
}
