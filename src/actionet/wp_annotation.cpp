// Pybind11 interface for `annotation` module
// Organized by module header in the order imported.

#include "wp_utils.h"
#include "libactionet.hpp"
#include "io/backed_h5ad/backed_sparse_matrix_operator.hpp"
#include "io/backed_h5ad/backed_dense_matrix_operator.hpp"

namespace py = pybind11;

// marker_stats ========================================================================================================

py::array_t<double> compute_feature_stats(py::object G, py::object S, py::object X, int norm_method = 2,
                                            double alpha = 0.85, int max_it = 5, bool approx = false,
                                            int thread_no = 0, bool ignore_baseline = false) {
    arma::sp_mat G_sp = scipy_to_arma_sparse(G);
    arma::sp_mat S_sp = scipy_to_arma_sparse(S);
    arma::sp_mat X_sp = scipy_to_arma_sparse(X);

    arma::mat stats;
    {
        py::gil_scoped_release release;
        stats = actionet::computeFeatureStats(G_sp, S_sp, X_sp, norm_method, alpha, max_it,
                                               approx, thread_no, ignore_baseline);
    }

    return arma_mat_to_numpy(stats);
}

py::array_t<double> compute_feature_stats_vision(py::object G, py::object S, py::object X, int norm_method = 2,
                                                   double alpha = 0.85, int max_it = 5, bool approx = false,
                                                   int thread_no = 0) {
    arma::sp_mat G_sp = scipy_to_arma_sparse(G);
    arma::sp_mat S_sp = scipy_to_arma_sparse(S);
    arma::sp_mat X_sp = scipy_to_arma_sparse(X);

    arma::mat stats;
    {
        py::gil_scoped_release release;
        stats = actionet::computeFeatureStatsVision(G_sp, S_sp, X_sp, norm_method,
                                                     alpha, max_it, approx, thread_no);
    }

    return arma_mat_to_numpy(stats);
}

py::array_t<double> compute_feature_stats_vision_from_stats(
        py::object G,
        py::array_t<double> stats_arr,
        py::array_t<double> mu_arr,
        py::array_t<double> sigma_sq_arr,
        py::object X,
        int norm_method = 2,
        double alpha = 0.85, int max_it = 5, bool approx = false,
        int thread_no = 0) {
    arma::sp_mat G_sp = scipy_to_arma_sparse(G);
    arma::mat stats_mat = numpy_to_arma_mat(stats_arr);
    arma::vec mu_vec = numpy_to_arma_vec(mu_arr);
    arma::vec sigma_sq_vec = numpy_to_arma_vec(sigma_sq_arr);
    arma::sp_mat X_sp = scipy_to_arma_sparse(X);

    arma::mat result;
    {
        py::gil_scoped_release release;
        result = actionet::computeFeatureStatsVisionFromStats(
            G_sp, stats_mat, mu_vec, sigma_sq_vec, X_sp,
            norm_method, alpha, max_it, approx, thread_no);
    }

    return arma_mat_to_numpy(result);
}

// specificity =========================================================================================================

py::dict archetype_feature_specificity_sparse(py::object S, py::array_t<double> H, int thread_no = 0) {
    arma::sp_mat S_sp = scipy_to_arma_sparse(S);
    arma::mat H_mat = numpy_to_arma_mat(H);

    arma::field<arma::mat> res;
    {
        py::gil_scoped_release release;
        res = actionet::computeFeatureSpecificity(S_sp, H_mat, thread_no);
    }

    py::dict out;
    out["archetypes"] = arma_mat_to_numpy(res(0));
    out["upper_significance"] = arma_mat_to_numpy(res(1));
    out["lower_significance"] = arma_mat_to_numpy(res(2));
    return out;
}

py::dict archetype_feature_specificity_dense(py::array_t<double> S, py::array_t<double> H, int thread_no = 0) {
    arma::mat S_mat = numpy_to_arma_mat(S);
    arma::mat H_mat = numpy_to_arma_mat(H);

    arma::field<arma::mat> res;
    {
        py::gil_scoped_release release;
        res = actionet::computeFeatureSpecificity(S_mat, H_mat, thread_no);
    }

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

    arma::field<arma::mat> res;
    {
        py::gil_scoped_release release;
        res = actionet::computeFeatureSpecificity(S_sp, labels_vec, thread_no);
    }

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

    arma::field<arma::mat> res;
    {
        py::gil_scoped_release release;
        res = actionet::computeFeatureSpecificity(S_mat, labels_vec, thread_no);
    }

    py::dict out;
    out["average_profile"] = arma_mat_to_numpy(res(0));
    out["upper_significance"] = arma_mat_to_numpy(res(1));
    out["lower_significance"] = arma_mat_to_numpy(res(2));
    return out;
}

// =====================================================================================================================

py::dict archetype_feature_specificity_backed_operator(
    std::shared_ptr<actionet::BackedSparseMatrixOperator> op,
    py::array_t<double> H, int thread_no = 0) {
    if (!op) {
        throw std::runtime_error("archetype_feature_specificity_backed_operator: operator is null");
    }
    arma::mat H_mat = numpy_to_arma_mat(H);

    py::gil_scoped_release release;
    arma::field<arma::mat> res = actionet::computeFeatureSpecificity(*op, H_mat, thread_no);
    py::gil_scoped_acquire acquire;

    py::dict out;
    out["archetypes"]          = arma_mat_to_numpy(res(0));
    out["upper_significance"]  = arma_mat_to_numpy(res(1));
    out["lower_significance"]  = arma_mat_to_numpy(res(2));
    return out;
}

py::dict compute_feature_specificity_backed_operator(
    std::shared_ptr<actionet::BackedSparseMatrixOperator> op,
    py::array_t<int> labels, int thread_no = 0) {
    if (!op) {
        throw std::runtime_error("compute_feature_specificity_backed_operator: operator is null");
    }

    auto labels_buf = labels.request();
    auto labels_ptr = static_cast<int*>(labels_buf.ptr);
    arma::uvec labels_vec(static_cast<size_t>(labels_buf.size));
    for (size_t i = 0; i < static_cast<size_t>(labels_buf.size); ++i) {
        labels_vec(i) = static_cast<arma::uword>(labels_ptr[i]);
    }

    py::gil_scoped_release release;
    arma::field<arma::mat> res = actionet::computeFeatureSpecificity(*op, labels_vec, thread_no);
    py::gil_scoped_acquire acquire;

    py::dict out;
    out["average_profile"]     = arma_mat_to_numpy(res(0));
    out["upper_significance"]  = arma_mat_to_numpy(res(1));
    out["lower_significance"]  = arma_mat_to_numpy(res(2));
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

    m.def("compute_feature_stats_vision_from_stats", &compute_feature_stats_vision_from_stats,
          "VISION standardization + diffusion from pre-computed per-row stats",
          py::arg("G"), py::arg("stats"), py::arg("mu"), py::arg("sigma_sq"),
          py::arg("X"), py::arg("norm_method") = 2,
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

    // backed operator specificity (sparse)
    m.def("archetype_feature_specificity_backed_operator", &archetype_feature_specificity_backed_operator,
          "Compute archetype feature specificity from HDF5-backed sparse matrix",
          py::arg("op"), py::arg("H"), py::arg("thread_no") = 0);

    m.def("compute_feature_specificity_backed_operator", &compute_feature_specificity_backed_operator,
          "Compute cluster feature specificity from HDF5-backed sparse matrix",
          py::arg("op"), py::arg("labels"), py::arg("thread_no") = 0);

    // backed operator specificity (dense)
    m.def("archetype_feature_specificity_backed_dense_operator",
          [](std::shared_ptr<actionet::MatrixOperator> op_base,
             py::array_t<double> H, int thread_no) -> py::dict {
              if (!op_base) {
                  throw std::runtime_error("archetype_feature_specificity_backed_dense_operator: operator is null");
              }
              auto* op = dynamic_cast<actionet::BackedDenseMatrixOperator*>(op_base.get());
              if (!op) {
                  throw std::runtime_error("archetype_feature_specificity_backed_dense_operator: operator is not a dense operator");
              }
              arma::mat H_mat = numpy_to_arma_mat(H);

              py::gil_scoped_release release;
              arma::field<arma::mat> res = actionet::computeFeatureSpecificity(*op, H_mat, thread_no);
              py::gil_scoped_acquire acquire;

              py::dict out;
              out["archetypes"]         = arma_mat_to_numpy(res(0));
              out["upper_significance"] = arma_mat_to_numpy(res(1));
              out["lower_significance"] = arma_mat_to_numpy(res(2));
              return out;
          },
          "Compute archetype feature specificity from HDF5-backed dense matrix",
          py::arg("op"), py::arg("H"), py::arg("thread_no") = 0);

    m.def("compute_feature_specificity_backed_dense_operator",
          [](std::shared_ptr<actionet::MatrixOperator> op_base,
             py::array_t<int> labels, int thread_no) -> py::dict {
              if (!op_base) {
                  throw std::runtime_error("compute_feature_specificity_backed_dense_operator: operator is null");
              }
              auto* op = dynamic_cast<actionet::BackedDenseMatrixOperator*>(op_base.get());
              if (!op) {
                  throw std::runtime_error("compute_feature_specificity_backed_dense_operator: operator is not a dense operator");
              }

              auto labels_buf = labels.request();
              auto labels_ptr = static_cast<int*>(labels_buf.ptr);
              arma::uvec labels_vec(static_cast<size_t>(labels_buf.size));
              for (size_t i = 0; i < static_cast<size_t>(labels_buf.size); ++i) {
                  labels_vec(i) = static_cast<arma::uword>(labels_ptr[i]);
              }

              py::gil_scoped_release release;
              arma::field<arma::mat> res = actionet::computeFeatureSpecificity(*op, labels_vec, thread_no);
              py::gil_scoped_acquire acquire;

              py::dict out;
              out["average_profile"]    = arma_mat_to_numpy(res(0));
              out["upper_significance"] = arma_mat_to_numpy(res(1));
              out["lower_significance"] = arma_mat_to_numpy(res(2));
              return out;
          },
          "Compute cluster feature specificity from HDF5-backed dense matrix",
          py::arg("op"), py::arg("labels"), py::arg("thread_no") = 0);

    // backed operator vision marker stats
    m.def("compute_feature_stats_vision_backed_operator",
          [](std::shared_ptr<actionet::MatrixOperator> op_base,
             py::object G_obj, py::object X_obj,
             int norm_method, double alpha, int max_it,
             bool approx, int thread_no) -> py::array_t<double> {
              if (!op_base) {
                  throw std::runtime_error("compute_feature_stats_vision_backed_operator: operator is null");
              }
              arma::sp_mat G_sp = scipy_to_arma_sparse(G_obj);
              arma::sp_mat X_sp = scipy_to_arma_sparse(X_obj);

              auto* sparse_op = dynamic_cast<actionet::BackedSparseMatrixOperator*>(op_base.get());
              auto* dense_op = dynamic_cast<actionet::BackedDenseMatrixOperator*>(op_base.get());

              arma::mat result;
              {
                  py::gil_scoped_release release;
                  if (sparse_op) {
                      result = actionet::computeFeatureStatsVision(
                          *sparse_op, G_sp, X_sp, norm_method, alpha, max_it, approx, thread_no);
                  } else if (dense_op) {
                      result = actionet::computeFeatureStatsVision(
                          *dense_op, G_sp, X_sp, norm_method, alpha, max_it, approx, thread_no);
                  } else {
                      throw std::runtime_error("compute_feature_stats_vision_backed_operator: unsupported operator type");
                  }
              }
              return arma_mat_to_numpy(result);
          },
          "Compute VISION marker statistics from HDF5-backed matrix",
          py::arg("op"), py::arg("G"), py::arg("X"),
          py::arg("norm_method") = 2, py::arg("alpha") = 0.85,
          py::arg("max_it") = 5, py::arg("approx") = false,
          py::arg("thread_no") = 0);
}
