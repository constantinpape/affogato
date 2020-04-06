#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyarray.hpp"

#include "affogato/affinities/affinities.hxx"
#include "affogato/affinities/multiscale_affinities.hxx"
#include "affogato/affinities/embedding_distances.hxx"

namespace py = pybind11;

namespace affogato {

    template<typename T>
    void export_affinities_T(py::module & m) {
        m.def("compute_affinities_impl_", [](const xt::pyarray <T> &labels,
                                            const std::vector <std::vector<int>> &offsets,
                                            const xt::pyarray <uint8_t> &boundary_mask,
                                            const xt::pyarray <uint8_t> &glia_mask,
                                            const bool have_ignore_label,
                                            const T ignore_label
              ) {
                  // compute the out shape
                  typedef typename xt::pyarray<float>::shape_type ShapeType;
                  const auto &shape = labels.shape();
                  const unsigned ndim = labels.dimension();
                  ShapeType out_shape(ndim + 1);
                  out_shape[0] = offsets.size();
                  for (unsigned d = 0; d < ndim; ++d) {
                      out_shape[d + 1] = shape[d];
                  }

                  // allocate the output
                  xt::pyarray<float> affs = xt::zeros<float>(out_shape);
                  xt::pyarray <uint8_t> mask = xt::zeros<uint8_t>(out_shape);
                  {
                      py::gil_scoped_release allowThreads;
                      affinities::compute_affinities(labels, offsets,
                                                     affs, mask,
                                                     boundary_mask,
                                                     glia_mask,
                                                     have_ignore_label,
                                                     ignore_label);
                  }
                  return std::make_pair(affs, mask);
              }, py::arg("labels").noconvert(),
              py::arg("offset"),
              py::arg("boundary_mask").noconvert(),
              py::arg("glia_mask").noconvert(),
              py::arg("have_ignore_label") = false,
              py::arg("ignore_label") = 0);
    }

    // for now we only export L2 norm
    template<typename T>
    void export_embedding_distances_T(py::module & m) {

        // l2 norm
        m.def("compute_embedding_distances_l2", [](const xt::pyarray<T> & values,
                                                   const std::vector<std::vector<int>> & offsets) {
            // compute the out shape
            typedef typename xt::pyarray<float>::shape_type ShapeType;
            const auto & shape = values.shape();
            const unsigned ndim = values.dimension() - 1;
            ShapeType out_shape(ndim + 1);
            out_shape[0] = offsets.size();
            for(unsigned d = 1; d < ndim + 1; ++d) {
                out_shape[d] = shape[d];
            }

            auto l2norm = [](const xt::pyarray<float> & values,
                             const xt::xindex & coord1,
                             const xt::xindex & coord2) {
                const int64_t ndim = values.dimension();
                // full coordinate
                xt::xindex coordA(ndim), coordB(ndim);
                // copy spatial
                std::copy(coord1.begin(), coord1.end(), coordA.begin() + 1);
                std::copy(coord2.begin(), coord2.end(), coordB.begin() + 1);
                float ret = 0;

                const int64_t n_channels = values.shape()[0];
                for(int c = 0; c < n_channels; ++c) {
                    coordA[0] = c;
                    coordB[0] = c;
                    ret += pow((values[coordA] - values[coordB]), 2.);
                }
                return sqrt(ret);
            };

            // allocate the output
            xt::pyarray<float> distances = xt::zeros<float>(out_shape);
            {
                py::gil_scoped_release allowThreads;
                affinities::compute_embedding_distances_impl(values, offsets, distances, l2norm);
            }
            return distances;
        }, py::arg("values").noconvert(), py::arg("offset"));

        // NOTE: we assume positive embeddings here, otherwise we could
        // negative cosine similarity, which would result in return values > 1
        // cosine distance
        m.def("compute_embedding_distances_cos", [](const xt::pyarray<T> & values,
                                                    const std::vector<std::vector<int>> & offsets) {
            // compute the out shape
            typedef typename xt::pyarray<float>::shape_type ShapeType;
            const auto & shape = values.shape();
            const unsigned ndim = values.dimension() - 1;
            ShapeType out_shape(ndim + 1);
            out_shape[0] = offsets.size();
            for(unsigned d = 1; d < ndim + 1; ++d) {
                out_shape[d] = shape[d];
            }

            auto cos_dist = [](const xt::pyarray<float> & values,
                               const xt::xindex & coord1,
                               const xt::xindex & coord2) {
                const int64_t ndim = values.dimension();
                // full coordinate
                xt::xindex coordA(ndim), coordB(ndim);
                // copy spatial
                std::copy(coord1.begin(), coord1.end(), coordA.begin() + 1);
                std::copy(coord2.begin(), coord2.end(), coordB.begin() + 1);

                double dot = 0;
                double normA = 0;
                double normB = 0;

                const int64_t n_channels = values.shape()[0];
                for(int c = 0; c < n_channels; ++c) {
                    coordA[0] = c;
                    coordB[0] = c;
                    const T valA = values[coordA];
                    const T valB = values[coordB];
                    dot += valA * valB;
                    normA += valA * valA;
                    normB += valB * valB;
                }
                normA = sqrt(normA);
                normB = sqrt(normB);
                float ret = 1. - dot / (normA * normB);
                return ret;
            };

            // allocate the output
            xt::pyarray<float> distances = xt::zeros<float>(out_shape);
            {
                py::gil_scoped_release allowThreads;
                affinities::compute_embedding_distances_impl(values, offsets, distances, cos_dist);
            }
            return distances;
        }, py::arg("values").noconvert(), py::arg("offset"));
    }


    //
    template<typename T>
    void export_multiscale_affinities_T(py::module & m) {
        m.def("compute_multiscale_affinities", [](const xt::pyarray<T> & labels,
                                                  const std::vector<int> & block_shape,
                                                  const bool have_ignore_label,
                                                  const T ignore_label) {
                // compute the out shape
                typedef typename xt::pyarray<float>::shape_type ShapeType;
                const auto & shape = labels.shape();
                const unsigned ndim = shape.size();
                ShapeType out_shape(ndim + 1);
                out_shape[0] = ndim;
                for(unsigned d = 0; d < ndim; ++d) {
                    // integer division should do the right thing in all cases
                    out_shape[d + 1] = (shape[d] % block_shape[d]) ? shape[d] / block_shape[d] + 1 : shape[d] / block_shape[d];
                }

                // allocate the output
                xt::pyarray<float> affs = xt::zeros<float>(out_shape);
                xt::pyarray<uint8_t> mask = xt::zeros<float>(out_shape);
                {
                    py::gil_scoped_release allowThreads;
                    affinities::compute_multiscale_affinities(labels, block_shape,
                                                              affs, mask,
                                                              have_ignore_label, ignore_label);
                }
                return std::make_pair(affs, mask);
            }, py::arg("labels"),
               py::arg("block_shape"),
               py::arg("have_ignore_label")=false,
               py::arg("ignore_label")=0);
    }
}

PYBIND11_MODULE(_affinities, m)
{
    xt::import_numpy();
    m.doc() = "affinity module of affogato";

    using namespace affogato;

    // bool export to compute affinity transitions to masks
    export_affinities_T<bool>(m);
    export_affinities_T<uint64_t>(m);
    export_affinities_T<int64_t>(m);

    export_multiscale_affinities_T<uint64_t>(m);
    export_multiscale_affinities_T<int64_t>(m);

    export_embedding_distances_T<float>(m);
}
