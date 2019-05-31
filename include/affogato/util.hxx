#pragma once

#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xstrided_view.hpp"


// overload ostream operator for xindex
inline std::ostream & operator << (std::ostream & os, const xt::xindex & coord) {
    os << "xindex(";
    for(const auto & cc: coord) {
        os << " " << cc;
    }
    os << " )";
    return os;
}


namespace affogato {
namespace util {

    //
    // for each coordinate:
    // iterate over single shape
    //

    template<typename COORD, typename F>
    inline void for_each_coordinate_c(const COORD & shape, F && f) {
        const int dim = shape.size();
        xt::xindex coord(dim);
        std::fill(coord.begin(), coord.end(), 0);

        // C-Order: last dimension is the fastest moving one
        for(int d = dim - 1; d >= 0;) {
            f(coord);
            for(d = dim - 1; d >= 0; --d) {
                ++coord[d];
                if(coord[d] < shape[d]) {
                    break;
                } else {
                    coord[d] = 0;
                }
            }
        }
    }


    template<typename COORD, typename F>
    inline void for_each_coordinate_f(const COORD & shape, F && f) {
        const int dim = shape.size();
        xt::xindex coord(dim);
        std::fill(coord.begin(), coord.end(), 0);

        // F-Order: last dimension is the fastest moving one
        for(int d = 0; d < dim;) {
            f(coord);
            for(d = 0; d < dim; ++d) {
                ++coord[d];
                if(coord[d] < shape[d]) {
                    break;
                } else {
                    coord[d] = 0;
                }
            }
        }
    }


    template<typename COORD, typename F>
    inline void for_each_coordinate(const COORD & shape, F && f, const bool c_order=true) {
        if(c_order) {
            for_each_coordinate_c(shape, f);
        } else {
            for_each_coordinate_f(shape, f);
        }
    }


    //
    // for each coordinate:
    // iterate between begin and end
    //


    template<typename COORD, typename F>
    inline void for_each_coordinate_c(const COORD & begin, const COORD & end, F && f) {
        const int dim = begin.size();
        xt::xindex coord(dim);
        std::copy(begin.begin(), begin.end(), coord.begin());

        // C-Order: last dimension is the fastest moving one
        for(int d = dim - 1; d >= 0;) {
            f(coord);
            for(d = dim - 1; d >= 0; --d) {
                ++coord[d];
                if(coord[d] < end[d]) {
                    break;
                } else {
                    coord[d] = 0;
                }
            }
        }
    }


    template<typename COORD, typename F>
    inline void for_each_coordinate_f(const COORD & begin, const COORD & end, F && f) {
        const int dim = begin.size();
        xt::xindex coord(dim);
        std::copy(begin.begin(), begin.end(), coord.begin());

        // F-Order: last dimension is the fastest moving one
        for(int d = 0; d < dim;) {
            f(coord);
            for(d = 0; d < dim; ++d) {
                ++coord[d];
                if(coord[d] < end[d]) {
                    break;
                } else {
                    coord[d] = 0;
                }
            }
        }
    }


    template<typename COORD, typename F>
    inline void for_each_coordinate(const COORD & begin, const COORD & end, F && f, const bool c_order=true) {
        if(c_order) {
            for_each_coordinate_c(begin, end, f);
        } else {
            for_each_coordinate_f(begin, end, f);
        }
    }

    template<typename UFD, typename NUM, typename NODE_ARRAY>
    inline uint64_t export_consecutive_labels(UFD & ufd, const NUM & number_of_nodes, NODE_ARRAY & node_labeling) {
        std::unordered_map<uint64_t, uint64_t> labelmap;
        for(size_t label = 0; label < number_of_nodes; ++label) {
            const uint64_t root = ufd.find_set(label);
            auto iter = labelmap.find(root);
            if (iter != labelmap.end()){
                node_labeling[label] = iter->second;
            } else {
                uint64_t newlabel = labelmap.size() + 1;
                labelmap[root] = newlabel;
                node_labeling[label] = newlabel;
            }
        }
        return labelmap.size();
    }

}
}
