// Pybind11 interface for `annotation` module
// Organized by module header in the order imported.

#include "wp_utils.h"
#include "libactionet.hpp"

namespace py = pybind11;

// marker_stats ========================================================================================================

py::array_t<double> compute_feature_stats(py::object G, py::object S, py::object X, int norm_method = 2,
                                            double alpha = 0.85, int max_it = 5, bool approx = false,
                                            int thread_no = 0, bool ignore_baseline = false) {
    arma::sp_mat G_sp = scipy_to_arma_sparse(G);
    arma::sp_mat S_sp = scipy_to_arma_sparse(S);
    arma::sp_mat X_sp = scipy_to_arma_sparse(X);

    arma::mat stats = actionet::computeFeatureStats(G_sp, S_sp, X_sp, norm_method, alpha, max_it,
                                                     approx, thread_no, ignore_baseline);

    return arma_mat_to_numpy(stats);
}

py::array_t<double> compute_feature_stats_vision(py::object G, py::object S, py::object X, int norm_method = 2,
                                                   double alpha = 0.85, int max_it = 5, bool approx = false,
                                                   int thread_no = 0) {
    arma::sp_mat G_sp = scipy_to_arma_sparse(G);
    arma::sp_mat S_sp = scipy_to_arma_sparse(S);
    arma::sp_mat X_sp = scipy_to_arma_sparse(X);

    arma::mat stats = actionet::computeFeatureStatsVision(G_sp, S_sp, X_sp, norm_method,
                                                           alpha, max_it, approx, thread_no);

    return arma_mat_to_numpy(stats);
}

// specificity =========================================================================================================

py::dict archetype_feature_specificity_sparse(py::object S, py::array_t<double> H, int thread_no = 0) {
    arma::sp_mat S_sp = scipy_to_arma_sparse(S);
    arma::mat H_mat = numpy_to_arma_mat(H);

    arma::field<arma::mat> res = actionet::computeFeatureSpecificity(S_sp, H_mat, thread_no);

    py::dict out;
    out["archetypes"] = arma_mat_to_numpy(res(0));
    out["upper_significance"] = arma_mat_to_numpy(res(1));
    out["lower_significance"] = arma_mat_to_numpy(res(2));
    return out;
}

py::dict archetype_feature_specificity_dense(py::array_t<double> S, py::array_t<double> H, int thread_no = 0) {
    arma::mat S_mat = numpy_to_arma_mat(S);
    arma::mat H_mat = numpy_to_arma_mat(H);

    arma::field<arma::mat> res = actionet::computeFeatureSpecificity(S_mat, H_mat, thread_no);

    py::dict out;
    out["archetypes"] = arma_mat_to_numpy(res(0));
    out["upper_significance"] = arma_mat_to_numpy(res(1));
    out["lower_significance"] = arma_mat_to_numpy(res(2));
    return out;
}

py::dict compute_feature_specificity_sparse(py::object S, py::array_t<int> labels, int thread_no = 0) {
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

py::dict compute_feature_specificity_dense(py::array_t<double> S, py::array_t<int> labels, int thread_no = 0) {
    arma::mat S_mat = numpy_to_arma_mat(S);

    auto labels_buf = labels.request();
    auto labels_ptr = static_cast<int*>(labels_buf.ptr);
    arma::uvec labels_vec(labels_buf.size);
    for (size_t i = 0; i < labels_buf.size; ++i) {
        labels_vec(i) = labels_ptr[i];
    }

    arma::field<arma::mat> res = actionet::computeFeatureSpecificity(S_mat, labels_vec, thread_no);

    py::dict out;
    out["average_profile"] = arma_mat_to_numpy(res(0));
    out["upper_significance"] = arma_mat_to_numpy(res(1));
    out["lower_significance"] = arma_mat_to_numpy(res(2));
    return out;
}

// =====================================================================================================================

void init_annotation(py::module_ &m) {
    // marker_stats
    m.def("compute_feature_stats", &compute_feature_stats,
          "Compute feature statistics",
          py::arg("G"), py::arg("S"), py::arg("X"), py::arg("norm_method") = 2,
          py::arg("alpha") = 0.85, py::arg("max_it") = 5, py::arg("approx") = false,
          py::arg("thread_no") = 0, py::arg("ignore_baseline") = false);

    m.def("compute_feature_stats_vision", &compute_feature_stats_vision,
          "Compute feature statistics (VISION method)",
          py::arg("G"), py::arg("S"), py::arg("X"), py::arg("norm_method") = 2,
          py::arg("alpha") = 0.85, py::arg("max_it") = 5, py::arg("approx") = false,
          py::arg("thread_no") = 0);

    // specificity
    m.def("archetype_feature_specificity_sparse", &archetype_feature_specificity_sparse,
          "Compute archetype feature specificity (sparse)",
          py::arg("S"), py::arg("H"), py::arg("thread_no") = 0);

    m.def("archetype_feature_specificity_dense", &archetype_feature_specificity_dense,
          "Compute archetype feature specificity (dense)",
          py::arg("S"), py::arg("H"), py::arg("thread_no") = 0);

    m.def("compute_feature_specificity_sparse", &compute_feature_specificity_sparse,
          "Compute feature specificity (sparse)",
          py::arg("S"), py::arg("labels"), py::arg("thread_no") = 0);

    m.def("compute_feature_specificity_dense", &compute_feature_specificity_dense,
          "Compute feature specificity (dense)",
          py::arg("S"), py::arg("labels"), py::arg("thread_no") = 0);
}
