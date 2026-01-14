// Pybind11 interface for `annotation` module
// Organized by module header in the order imported.

#include "wp_utils.h"
#include "libactionet.hpp"

namespace py = pybind11;

// marker_stats ========================================================================================================

py::dict compute_feature_specificity_sparse(py::object S, py::array_t<int> labels,
                                             int thread_no = 0) {
    arma::sp_mat S_sp = scipy_to_arma_sparse(S);

    auto labels_buf = labels.request();
    auto labels_ptr = static_cast<int*>(labels_buf.ptr);
    arma::uvec labels_vec(labels_buf.size);
    for (size_t i = 0; i < labels_buf.size; ++i) {
        labels_vec(i) = labels_ptr[i];
    }

    arma::field<arma::mat> res = actionet::computeFeatureSpecificity(S_sp, labels_vec, thread_no);

    py::dict out;
    out["average_profile"] = arma_mat_to_numpy(res(0));
    out["upper_significance"] = arma_mat_to_numpy(res(1));
    out["lower_significance"] = arma_mat_to_numpy(res(2));
    return out;
}

// =====================================================================================================================

void init_annotation(py::module_ &m) {
    m.def("compute_feature_specificity_sparse", &compute_feature_specificity_sparse,
          "Compute feature specificity",
          py::arg("S"), py::arg("labels"), py::arg("thread_no") = 0);
}
