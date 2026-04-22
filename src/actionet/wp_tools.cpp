// Pybind11 interface for `tools` module
// Organized by module header in the order imported.

#include "wp_utils.h"
#include "libactionet.hpp"
#include "io/backed_h5ad/backed_sparse_matrix_operator.hpp"
#include "io/backed_h5ad/backed_dense_matrix_operator.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace py = pybind11;

namespace {
    py::array_t<int64_t> arma_uvec_to_numpy_int64(const arma::uvec& vec) {
        py::array_t<int64_t> arr(vec.n_elem);
        auto buf = arr.request();
        auto* ptr = static_cast<int64_t*>(buf.ptr);
        for (arma::uword i = 0; i < vec.n_elem; ++i) {
            ptr[i] = static_cast<int64_t>(vec(i));
        }
        return arr;
    }

    py::array_t<int> arma_ivec_to_numpy_int(const arma::ivec& vec) {
        py::array_t<int> arr(vec.n_elem);
        auto buf = arr.request();
        auto* ptr = static_cast<int*>(buf.ptr);
        for (arma::uword i = 0; i < vec.n_elem; ++i) {
            ptr[i] = static_cast<int>(vec(i));
        }
        return arr;
    }

    /// Populate a GuideGMMFitParams struct from Python-side keyword arguments.
    actionet::GuideGMMFitParams make_guide_fit_params(
        int min_points,
        double min_counts,
        int n_init,
        int max_iter,
        double tol,
        double variance_floor,
        bool apply_log10p1,
        int seed,
        int n_threads,
        int backed_chunk_guides
    ) {
        actionet::GuideGMMFitParams params;
        params.min_points = static_cast<arma::uword>(std::max(min_points, 0));
        params.min_counts = min_counts;
        params.n_init = static_cast<arma::uword>(std::max(n_init, 1));
        params.max_iter = static_cast<arma::uword>(std::max(max_iter, 1));
        params.tol = tol;
        params.variance_floor = variance_floor;
        params.apply_log10p1 = apply_log10p1;
        params.seed = seed;
        params.n_threads = n_threads;
        params.backed_chunk_guides = static_cast<arma::uword>(std::max(backed_chunk_guides, 1));
        return params;
    }

    /// Convert a C++ GuideGMMFitResult into a Python dict of numpy arrays,
    /// including a nested status_codes dict for interpreting the status column.
    py::dict guide_fit_to_dict(const actionet::GuideGMMFitResult& fits) {
        py::dict out;
        out["weights"] = arma_mat_to_numpy(fits.weights);
        out["means"] = arma_mat_to_numpy(fits.means);
        out["sigma"] = arma_vec_to_numpy(fits.sigma);
        out["log_likelihood"] = arma_vec_to_numpy(fits.log_likelihood);
        out["n_points"] = arma_uvec_to_numpy_int64(fits.n_points);
        out["status"] = arma_ivec_to_numpy_int(fits.status);

        py::dict status_codes;
        status_codes["ok"] = static_cast<int>(actionet::GUIDE_GMM_OK);
        status_codes["insufficient_points"] = static_cast<int>(actionet::GUIDE_GMM_INSUFFICIENT_POINTS);
        status_codes["degenerate"] = static_cast<int>(actionet::GUIDE_GMM_DEGENERATE);
        status_codes["numerical_failure"] = static_cast<int>(actionet::GUIDE_GMM_NUMERICAL_FAILURE);
        out["status_codes"] = status_codes;
        return out;
    }

    /// Reconstruct a GuideGMMFitResult from the component numpy arrays returned
    /// by the Python layer (used by derive/sweep threshold binding functions).
    actionet::GuideGMMFitResult fit_from_arrays(
        py::array_t<double> weights_arr,
        py::array_t<double> means_arr,
        py::array_t<double> sigma_arr,
        py::array_t<int> status_arr
    ) {
        actionet::GuideGMMFitResult fits;
        fits.weights = numpy_to_arma_mat(weights_arr);
        fits.means = numpy_to_arma_mat(means_arr);
        fits.sigma = numpy_to_arma_vec(sigma_arr);

        if (fits.weights.n_cols != 2 || fits.means.n_cols != 2) {
            throw std::runtime_error("weights and means must be shaped (n_guides, 2)");
        }
        if (fits.weights.n_rows != fits.means.n_rows || fits.weights.n_rows != fits.sigma.n_elem) {
            throw std::runtime_error("weights/means/sigma guide dimensions must match");
        }

        py::buffer_info status_buf = status_arr.request();
        if (status_buf.ndim != 1) {
            throw std::runtime_error("status must be a 1D array");
        }
        if (static_cast<arma::uword>(status_buf.shape[0]) != fits.weights.n_rows) {
            throw std::runtime_error("status length must match number of guides");
        }

        fits.status.set_size(fits.weights.n_rows);
        auto* status_ptr = static_cast<int*>(status_buf.ptr);
        for (arma::uword i = 0; i < fits.status.n_elem; ++i) {
            fits.status(i) = status_ptr[i];
        }

        fits.log_likelihood = arma::vec(fits.weights.n_rows, arma::fill::zeros);
        fits.n_points = arma::uvec(fits.weights.n_rows, arma::fill::zeros);
        return fits;
    }
} // namespace

// autocorrelation =====================================================================================================

py::dict autocorrelation_moran(py::object G, py::array_t<double> scores, int normalization_method = 1,
                                int perm_no = 30, int thread_no = 0) {
    arma::sp_mat G_sp = scipy_to_arma_sparse(G);
    arma::mat scores_mat = numpy_to_arma_mat(scores);

    arma::field<arma::vec> out;
    {
        py::gil_scoped_release release;
        out = actionet::autocorrelation_Moran(G_sp, scores_mat, normalization_method, perm_no, thread_no);
    }

    py::dict res;
    res["Moran_I"] = arma_vec_to_numpy(out[0]);
    res["zscore"] = arma_vec_to_numpy(out[1]);
    res["mu"] = arma_vec_to_numpy(out[2]);
    res["sigma"] = arma_vec_to_numpy(out[3]);

    return res;
}

py::dict autocorrelation_geary(py::object G, py::array_t<double> scores, int normalization_method = 1,
                                int perm_no = 30, int thread_no = 0) {
    arma::sp_mat G_sp = scipy_to_arma_sparse(G);
    arma::mat scores_mat = numpy_to_arma_mat(scores);

    arma::field<arma::vec> out;
    {
        py::gil_scoped_release release;
        out = actionet::autocorrelation_Geary(G_sp, scores_mat, normalization_method, perm_no, thread_no);
    }

    py::dict res;
    res["Geary_C"] = arma_vec_to_numpy(out[0]);
    res["zscore"] = arma_vec_to_numpy(out[1]);
    res["mu"] = arma_vec_to_numpy(out[2]);
    res["sigma"] = arma_vec_to_numpy(out[3]);

    return res;
}

// enrichment ==========================================================================================================

py::array_t<double> compute_graph_label_enrichment(py::object G, py::array_t<double> scores, int thread_no = 0) {
    arma::sp_mat G_sp = scipy_to_arma_sparse(G);
    arma::mat scores_mat = numpy_to_arma_mat(scores);

    arma::mat logPvals;
    {
        py::gil_scoped_release release;
        logPvals = actionet::computeGraphLabelEnrichment(G_sp, scores_mat, thread_no);
    }

    return arma_mat_to_numpy(logPvals);
}

py::dict assess_enrichment(py::array_t<double> scores, py::object associations, int thread_no = 0) {
    arma::mat scores_mat = numpy_to_arma_mat(scores);
    arma::sp_mat assoc_sp = scipy_to_arma_sparse(associations);

    arma::field<arma::mat> res;
    {
        py::gil_scoped_release release;
        res = actionet::assess_enrichment(scores_mat, assoc_sp, thread_no);
    }

    py::dict out;
    out["logPvals"] = arma_mat_to_numpy(res(0));
    out["thresholds"] = arma_mat_to_numpy(res(1));

    return out;
}

// matrix_aggregate ====================================================================================================

py::array_t<double> compute_grouped_sums_sparse(py::object S, py::array_t<double> sample_assignments, int axis = 0) {
    arma::sp_mat S_sp = scipy_to_arma_sparse(S);
    arma::vec assignments_vec = numpy_to_arma_vec(sample_assignments);

    arma::mat pb;
    {
        py::gil_scoped_release release;
        pb = actionet::computeGroupedSums<arma::sp_mat, arma::mat>(S_sp, assignments_vec, axis);
    }
    return arma_mat_to_numpy(pb);
}

py::object compute_grouped_sums_sparse2(py::object S, py::array_t<double> sample_assignments, int axis = 0) {
    arma::sp_mat S_sp = scipy_to_arma_sparse(S);
    arma::vec assignments_vec = numpy_to_arma_vec(sample_assignments);

    arma::sp_mat pb;
    {
        py::gil_scoped_release release;
        pb = actionet::computeGroupedSums<arma::sp_mat, arma::sp_mat>(S_sp, assignments_vec, axis);
    }
    return arma_sparse_to_scipy(pb);
}

py::array_t<double> compute_grouped_sums_dense(py::array_t<double> S, py::array_t<double> sample_assignments, int axis = 0) {
    arma::mat S_mat = numpy_to_arma_mat(S);
    arma::vec assignments_vec = numpy_to_arma_vec(sample_assignments);

    arma::mat pb;
    {
        py::gil_scoped_release release;
        pb = actionet::computeGroupedSums<arma::mat, arma::mat>(S_mat, assignments_vec, axis);
    }
    return arma_mat_to_numpy(pb);
}

py::array_t<double> compute_grouped_means_sparse(py::object S, py::array_t<double> sample_assignments, int axis = 0) {
    arma::sp_mat S_sp = scipy_to_arma_sparse(S);
    arma::vec assignments_vec = numpy_to_arma_vec(sample_assignments);

    arma::mat pb;
    {
        py::gil_scoped_release release;
        pb = actionet::computeGroupedMeans<arma::sp_mat, arma::mat>(S_sp, assignments_vec, axis);
    }
    return arma_mat_to_numpy(pb);
}

py::object compute_grouped_means_sparse2(py::object S, py::array_t<double> sample_assignments, int axis = 0) {
    arma::sp_mat S_sp = scipy_to_arma_sparse(S);
    arma::vec assignments_vec = numpy_to_arma_vec(sample_assignments);

    arma::sp_mat pb;
    {
        py::gil_scoped_release release;
        pb = actionet::computeGroupedMeans<arma::sp_mat, arma::sp_mat>(S_sp, assignments_vec, axis);
    }
    return arma_sparse_to_scipy(pb);
}

py::array_t<double> compute_grouped_means_dense(py::array_t<double> S, py::array_t<double> sample_assignments, int axis = 0) {
    arma::mat S_mat = numpy_to_arma_mat(S);
    arma::vec assignments_vec = numpy_to_arma_vec(sample_assignments);

    arma::mat pb;
    {
        py::gil_scoped_release release;
        pb = actionet::computeGroupedMeans<arma::mat, arma::mat>(S_mat, assignments_vec, axis);
    }
    return arma_mat_to_numpy(pb);
}

py::array_t<double> compute_grouped_vars_sparse(py::object S, py::array_t<double> sample_assignments, int axis = 0) {
    arma::sp_mat S_sp = scipy_to_arma_sparse(S);
    arma::vec assignments_vec = numpy_to_arma_vec(sample_assignments);

    arma::mat pb;
    {
        py::gil_scoped_release release;
        pb = actionet::computeGroupedVars<arma::sp_mat, arma::mat>(S_sp, assignments_vec, axis);
    }
    return arma_mat_to_numpy(pb);
}

py::object compute_grouped_vars_sparse2(py::object S, py::array_t<double> sample_assignments, int axis = 0) {
    arma::sp_mat S_sp = scipy_to_arma_sparse(S);
    arma::vec assignments_vec = numpy_to_arma_vec(sample_assignments);

    arma::sp_mat pb;
    {
        py::gil_scoped_release release;
        pb = actionet::computeGroupedVars<arma::sp_mat, arma::sp_mat>(S_sp, assignments_vec, axis);
    }
    return arma_sparse_to_scipy(pb);
}

py::array_t<double> compute_grouped_vars_dense(py::array_t<double> S, py::array_t<double> sample_assignments, int axis = 0) {
    arma::mat S_mat = numpy_to_arma_mat(S);
    arma::vec assignments_vec = numpy_to_arma_vec(sample_assignments);

    arma::mat pb;
    {
        py::gil_scoped_release release;
        pb = actionet::computeGroupedVars<arma::mat, arma::mat>(S_mat, assignments_vec, axis);
    }
    return arma_mat_to_numpy(pb);
}

// matrix_transform ====================================================================================================

py::object normalize_matrix_sparse(py::object X, unsigned int p = 1, unsigned int dim = 0) {
    arma::sp_mat X_sp = scipy_to_arma_sparse(X);
    arma::sp_mat Xn;
    {
        py::gil_scoped_release release;
        Xn = actionet::normalizeMatrix(X_sp, p, dim);
    }
    return arma_sparse_to_scipy(Xn);
}

py::array_t<double> normalize_matrix_dense(py::array_t<double> X, unsigned int p = 1, unsigned int dim = 0) {
    arma::mat X_mat = numpy_to_arma_mat(X);
    arma::mat Xn;
    {
        py::gil_scoped_release release;
        Xn = actionet::normalizeMatrix(X_mat, p, dim);
    }
    return arma_mat_to_numpy(Xn);
}

py::array_t<double> scale_matrix_dense(py::array_t<double> X, py::array_t<double> v, unsigned int dim = 0) {
    arma::mat X_mat = numpy_to_arma_mat(X);
    arma::vec v_vec = numpy_to_arma_vec(v);
    arma::mat Xs;
    {
        py::gil_scoped_release release;
        Xs = actionet::scaleMatrix(X_mat, v_vec, dim);
    }
    return arma_mat_to_numpy(Xs);
}

py::object scale_matrix_sparse(py::object X, py::array_t<double> v, unsigned int dim = 0) {
    arma::sp_mat X_sp = scipy_to_arma_sparse(X);
    arma::vec v_vec = numpy_to_arma_vec(v);
    arma::sp_mat Xs;
    {
        py::gil_scoped_release release;
        Xs = actionet::scaleMatrix(X_sp, v_vec, dim);
    }
    return arma_sparse_to_scipy(Xs);
}

py::object normalize_graph(py::object G, int norm_method = 0) {
    arma::sp_mat G_sp = scipy_to_arma_sparse(G);
    {
        py::gil_scoped_release release;
        actionet::normalizeGraph(G_sp, norm_method);
    }
    return arma_sparse_to_scipy(G_sp);
}

// mwm =================================================================================================================

py::array_t<double> mwm_hungarian(py::array_t<double> G) {
    arma::mat G_mat = numpy_to_arma_mat(G);
    arma::mat G_matched;
    {
        py::gil_scoped_release release;
        G_matched = actionet::MWM_hungarian(G_mat);
    }
    return arma_mat_to_numpy(G_matched);
}

py::array_t<int> mwm_rank1(py::array_t<double> u, py::array_t<double> v,
                             double u_threshold = 0, double v_threshold = 0) {
    arma::vec u_vec = numpy_to_arma_vec(u);
    arma::vec v_vec = numpy_to_arma_vec(v);

    arma::umat pairs;
    {
        py::gil_scoped_release release;
        pairs = actionet::MWM_rank1(u_vec, v_vec, u_threshold, v_threshold);
    }

    // Return 0-indexed pairs for Python
    std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(pairs.n_rows), 2};
    auto result = py::array_t<int>(shape);
    auto buf = result.request();
    int* ptr = static_cast<int*>(buf.ptr);
    for (size_t i = 0; i < pairs.n_rows; ++i) {
        ptr[i * 2] = static_cast<int>(pairs(i, 0));
        ptr[i * 2 + 1] = static_cast<int>(pairs(i, 1));
    }

    return result;
}


// xicor ===============================================================================================================

py::array_t<double> xicor(py::array_t<double> xvec, py::array_t<double> yvec,
                           bool compute_pval = true, int seed = 0) {
    arma::vec x_vec = numpy_to_arma_vec(xvec);
    arma::vec y_vec = numpy_to_arma_vec(yvec);

    arma::vec res;
    {
        py::gil_scoped_release release;
        res = actionet::xicor(std::move(x_vec), std::move(y_vec), compute_pval, seed);
    }

    return arma_vec_to_numpy(res);
}

py::dict xicor_matrix(py::array_t<double> X, py::array_t<double> Y,
                       bool compute_pval = true, int seed = 0, int thread_no = 0) {
    arma::mat X_mat = numpy_to_arma_mat(X);
    arma::mat Y_mat = numpy_to_arma_mat(Y);

    arma::field<arma::mat> out;
    {
        py::gil_scoped_release release;
        out = actionet::XICOR(X_mat, Y_mat, compute_pval, seed, thread_no);
    }

    py::dict res;
    res["XI"] = arma_mat_to_numpy(out(0));
    res["Z"] = arma_mat_to_numpy(out(1));

    return res;
}

// guide_calling ======================================================================================================

/// Fit per-guide GMMs on an in-memory scipy sparse matrix (GIL released).
py::dict fit_guides_gmm_sparse(py::object X,
                               int min_points = 5,
                               double min_counts = 10.0,
                               int n_init = 8,
                               int max_iter = 200,
                               double tol = 1e-6,
                               double variance_floor = 1e-3,
                               bool apply_log10p1 = true,
                               int seed = 0,
                               int n_threads = 0) {
    arma::sp_mat X_sp = scipy_to_arma_sparse(X);
    actionet::GuideGMMFitParams params = make_guide_fit_params(
        min_points,
        min_counts,
        n_init,
        max_iter,
        tol,
        variance_floor,
        apply_log10p1,
        seed,
        n_threads,
        256
    );

    actionet::GuideGMMFitResult fits;
    {
        py::gil_scoped_release release;
        fits = actionet::fitGuidesSharedVarianceGMM(X_sp, params);
    }
    return guide_fit_to_dict(fits);
}

/// Fit per-guide GMMs via a backed MatrixOperator (chunked HDF5, GIL released).
py::dict fit_guides_gmm_backed_operator(std::shared_ptr<actionet::MatrixOperator> op_base,
                                        int min_points = 5,
                                        double min_counts = 10.0,
                                        int n_init = 8,
                                        int max_iter = 200,
                                        double tol = 1e-6,
                                        double variance_floor = 1e-3,
                                        bool apply_log10p1 = true,
                                        int seed = 0,
                                        int n_threads = 0,
                                        int backed_chunk_guides = 256) {
    if (!op_base) {
        throw std::runtime_error("fit_guides_gmm_backed_operator: operator is null");
    }

    actionet::GuideGMMFitParams params = make_guide_fit_params(
        min_points,
        min_counts,
        n_init,
        max_iter,
        tol,
        variance_floor,
        apply_log10p1,
        seed,
        n_threads,
        backed_chunk_guides
    );

    actionet::GuideGMMFitResult fits;
    {
        py::gil_scoped_release release;
        if (auto* sparse_op = dynamic_cast<actionet::BackedSparseMatrixOperator*>(op_base.get())) {
            fits = actionet::fitGuidesSharedVarianceGMM(*sparse_op, params);
        } else if (auto* dense_op = dynamic_cast<actionet::BackedDenseMatrixOperator*>(op_base.get())) {
            fits = actionet::fitGuidesSharedVarianceGMM(*dense_op, params);
        } else {
            throw std::runtime_error("fit_guides_gmm_backed_operator: unsupported operator type");
        }
    }
    return guide_fit_to_dict(fits);
}

/// Derive quantile-based thresholds from pre-fitted GMM arrays.
py::dict derive_guide_thresholds_quantile(py::array_t<double> weights,
                                          py::array_t<double> means,
                                          py::array_t<double> sigma,
                                          py::array_t<int> status,
                                          double bg_quantile = 0.99,
                                          double fg_quantile = 0.01) {
    actionet::GuideGMMFitResult fits = fit_from_arrays(weights, means, sigma, status);
    actionet::GuideThresholdResult thresholds;
    {
        py::gil_scoped_release release;
        thresholds = actionet::deriveGuideThresholdsQuantile(fits, bg_quantile, fg_quantile);
    }

    py::dict out;
    out["background"] = arma_vec_to_numpy(thresholds.background);
    out["foreground"] = arma_vec_to_numpy(thresholds.foreground);
    return out;
}

/// Derive equal-density intersection thresholds from pre-fitted GMM arrays.
py::dict derive_guide_thresholds_equal_density(py::array_t<double> weights,
                                               py::array_t<double> means,
                                               py::array_t<double> sigma,
                                               py::array_t<int> status) {
    actionet::GuideGMMFitResult fits = fit_from_arrays(weights, means, sigma, status);
    actionet::GuideThresholdResult thresholds;
    {
        py::gil_scoped_release release;
        thresholds = actionet::deriveGuideThresholdsEqualDensity(fits);
    }

    py::dict out;
    out["background"] = arma_vec_to_numpy(thresholds.background);
    out["foreground"] = arma_vec_to_numpy(thresholds.foreground);
    return out;
}

/// Derive valley (minimum density) thresholds from pre-fitted GMM arrays.
py::dict derive_guide_thresholds_valley(py::array_t<double> weights,
                                        py::array_t<double> means,
                                        py::array_t<double> sigma,
                                        py::array_t<int> status,
                                        int grid_size = 256) {
    actionet::GuideGMMFitResult fits = fit_from_arrays(weights, means, sigma, status);
    actionet::GuideThresholdResult thresholds;
    {
        py::gil_scoped_release release;
        thresholds = actionet::deriveGuideThresholdsValley(
            fits,
            static_cast<arma::uword>(std::max(grid_size, 3))
        );
    }

    py::dict out;
    out["background"] = arma_vec_to_numpy(thresholds.background);
    out["foreground"] = arma_vec_to_numpy(thresholds.foreground);
    return out;
}

/// Sweep quantile thresholds across bg/fg quantile grids.
py::dict sweep_guide_thresholds_quantile(py::array_t<double> weights,
                                         py::array_t<double> means,
                                         py::array_t<double> sigma,
                                         py::array_t<int> status,
                                         py::array_t<double> bg_quantiles,
                                         py::array_t<double> fg_quantiles) {
    actionet::GuideGMMFitResult fits = fit_from_arrays(weights, means, sigma, status);
    arma::vec bg_q = numpy_to_arma_vec(bg_quantiles);
    arma::vec fg_q = numpy_to_arma_vec(fg_quantiles);

    actionet::GuideThresholdSweepResult thresholds;
    {
        py::gil_scoped_release release;
        thresholds = actionet::sweepGuideThresholdsQuantile(fits, bg_q, fg_q);
    }

    py::dict out;
    out["background"] = arma_mat_to_numpy(thresholds.background);
    out["foreground"] = arma_mat_to_numpy(thresholds.foreground);
    return out;
}

/// Apply per-guide thresholds to an in-memory scipy sparse matrix.
/// Returns background and foreground binary indicator matrices.
py::dict apply_guide_thresholds_sparse(py::object X,
                                       py::array_t<double> background_thresholds,
                                       py::array_t<double> foreground_thresholds) {
    arma::sp_mat X_sp = scipy_to_arma_sparse(X);
    arma::vec bg = numpy_to_arma_vec(background_thresholds);
    arma::vec fg = numpy_to_arma_vec(foreground_thresholds);

    arma::field<arma::sp_mat> out;
    {
        py::gil_scoped_release release;
        out = actionet::applyGuideThresholds(X_sp, bg, fg);
    }

    py::dict result;
    result["background"] = arma_sparse_to_scipy(out(0));
    result["foreground"] = arma_sparse_to_scipy(out(1));
    return result;
}

/// Apply per-guide thresholds to a backed MatrixOperator (chunked HDF5).
py::dict apply_guide_thresholds_backed_operator(std::shared_ptr<actionet::MatrixOperator> op_base,
                                                py::array_t<double> background_thresholds,
                                                py::array_t<double> foreground_thresholds,
                                                int chunk_guides = 256) {
    if (!op_base) {
        throw std::runtime_error("apply_guide_thresholds_backed_operator: operator is null");
    }

    arma::vec bg = numpy_to_arma_vec(background_thresholds);
    arma::vec fg = numpy_to_arma_vec(foreground_thresholds);
    arma::field<arma::sp_mat> out;

    {
        py::gil_scoped_release release;
        if (auto* sparse_op = dynamic_cast<actionet::BackedSparseMatrixOperator*>(op_base.get())) {
            out = actionet::applyGuideThresholds(
                *sparse_op,
                bg,
                fg,
                static_cast<arma::uword>(std::max(chunk_guides, 1))
            );
        } else if (auto* dense_op = dynamic_cast<actionet::BackedDenseMatrixOperator*>(op_base.get())) {
            out = actionet::applyGuideThresholds(
                *dense_op,
                bg,
                fg,
                static_cast<arma::uword>(std::max(chunk_guides, 1))
            );
        } else {
            throw std::runtime_error("apply_guide_thresholds_backed_operator: unsupported operator type");
        }
    }

    py::dict result;
    result["background"] = arma_sparse_to_scipy(out(0));
    result["foreground"] = arma_sparse_to_scipy(out(1));
    return result;
}

// =====================================================================================================================

void init_tools(py::module_ &m) {
    // autocorrelation
    m.def("autocorrelation_moran", &autocorrelation_moran, "Compute Moran's I autocorrelation",
          py::arg("G"), py::arg("scores"), py::arg("normalization_method") = 1,
          py::arg("perm_no") = 30, py::arg("thread_no") = 0);

    m.def("autocorrelation_geary", &autocorrelation_geary, "Compute Geary's C autocorrelation",
          py::arg("G"), py::arg("scores"), py::arg("normalization_method") = 1,
          py::arg("perm_no") = 30, py::arg("thread_no") = 0);

    // enrichment
    m.def("compute_graph_label_enrichment", &compute_graph_label_enrichment,
          "Compute graph label enrichment",
          py::arg("G"), py::arg("scores"), py::arg("thread_no") = 0);

    m.def("assess_enrichment", &assess_enrichment, "Assess feature enrichment",
          py::arg("scores"), py::arg("associations"), py::arg("thread_no") = 0);

    // matrix_aggregate
    m.def("compute_grouped_sums_sparse", &compute_grouped_sums_sparse, "Compute grouped sums (sparse)",
          py::arg("S"), py::arg("sample_assignments"), py::arg("axis") = 0);

    m.def("compute_grouped_sums_sparse2", &compute_grouped_sums_sparse2, "Compute grouped sums (sparse output)",
          py::arg("S"), py::arg("sample_assignments"), py::arg("axis") = 0);

    m.def("compute_grouped_sums_dense", &compute_grouped_sums_dense, "Compute grouped sums (dense)",
          py::arg("S"), py::arg("sample_assignments"), py::arg("axis") = 0);

    m.def("compute_grouped_means_sparse", &compute_grouped_means_sparse, "Compute grouped means (sparse)",
          py::arg("S"), py::arg("sample_assignments"), py::arg("axis") = 0);

    m.def("compute_grouped_means_sparse2", &compute_grouped_means_sparse2, "Compute grouped means (sparse output)",
          py::arg("S"), py::arg("sample_assignments"), py::arg("axis") = 0);

    m.def("compute_grouped_means_dense", &compute_grouped_means_dense, "Compute grouped means (dense)",
          py::arg("S"), py::arg("sample_assignments"), py::arg("axis") = 0);

    m.def("compute_grouped_vars_sparse", &compute_grouped_vars_sparse, "Compute grouped variances (sparse)",
          py::arg("S"), py::arg("sample_assignments"), py::arg("axis") = 0);

    m.def("compute_grouped_vars_sparse2", &compute_grouped_vars_sparse2, "Compute grouped variances (sparse output)",
          py::arg("S"), py::arg("sample_assignments"), py::arg("axis") = 0);

    m.def("compute_grouped_vars_dense", &compute_grouped_vars_dense, "Compute grouped variances (dense)",
          py::arg("S"), py::arg("sample_assignments"), py::arg("axis") = 0);

    // matrix_transform
    m.def("normalize_matrix_sparse", &normalize_matrix_sparse, "Normalize matrix (sparse)",
          py::arg("X"), py::arg("p") = 1, py::arg("dim") = 0);

    m.def("normalize_matrix_dense", &normalize_matrix_dense, "Normalize matrix (dense)",
          py::arg("X"), py::arg("p") = 1, py::arg("dim") = 0);

    m.def("scale_matrix_dense", &scale_matrix_dense, "Scale matrix (dense)",
          py::arg("X"), py::arg("v"), py::arg("dim") = 0);

    m.def("scale_matrix_sparse", &scale_matrix_sparse, "Scale matrix (sparse)",
          py::arg("X"), py::arg("v"), py::arg("dim") = 0);

    m.def("normalize_graph", &normalize_graph, "Normalize graph",
          py::arg("G"), py::arg("norm_method") = 0);

    // mwm
    m.def("mwm_hungarian", &mwm_hungarian, "Maximum weight matching (Hungarian algorithm)",
          py::arg("G"));

    m.def("mwm_rank1", &mwm_rank1, "Maximum weight matching (rank-1)",
          py::arg("u"), py::arg("v"), py::arg("u_threshold") = 0, py::arg("v_threshold") = 0);


    // xicor
    m.def("xicor", &xicor, "Compute xicor correlation",
          py::arg("xvec"), py::arg("yvec"), py::arg("compute_pval") = true, py::arg("seed") = 0);

    m.def("xicor_matrix", &xicor_matrix, "Compute xicor correlation matrix",
          py::arg("X"), py::arg("Y"), py::arg("compute_pval") = true,
          py::arg("seed") = 0, py::arg("thread_no") = 0);

    // guide_calling
    m.def("fit_guides_gmm_sparse", &fit_guides_gmm_sparse,
          "Fit per-guide 2-component shared-variance GMMs for sparse inputs",
          py::arg("X"),
          py::arg("min_points") = 5,
          py::arg("min_counts") = 10.0,
          py::arg("n_init") = 8,
          py::arg("max_iter") = 200,
          py::arg("tol") = 1e-6,
          py::arg("variance_floor") = 1e-3,
          py::arg("apply_log10p1") = true,
          py::arg("seed") = 0,
          py::arg("n_threads") = 0);

    m.def("fit_guides_gmm_backed_operator", &fit_guides_gmm_backed_operator,
          "Fit per-guide 2-component shared-variance GMMs for backed operators",
          py::arg("op"),
          py::arg("min_points") = 5,
          py::arg("min_counts") = 10.0,
          py::arg("n_init") = 8,
          py::arg("max_iter") = 200,
          py::arg("tol") = 1e-6,
          py::arg("variance_floor") = 1e-3,
          py::arg("apply_log10p1") = true,
          py::arg("seed") = 0,
          py::arg("n_threads") = 0,
          py::arg("backed_chunk_guides") = 256);

    m.def("derive_guide_thresholds_quantile", &derive_guide_thresholds_quantile,
          "Derive quantile thresholds from fitted guide GMM parameters",
          py::arg("weights"), py::arg("means"), py::arg("sigma"), py::arg("status"),
          py::arg("bg_quantile") = 0.99, py::arg("fg_quantile") = 0.01);

    m.def("derive_guide_thresholds_equal_density", &derive_guide_thresholds_equal_density,
          "Derive equal-density intersection thresholds from fitted guide GMM parameters",
          py::arg("weights"), py::arg("means"), py::arg("sigma"), py::arg("status"));

    m.def("derive_guide_thresholds_valley", &derive_guide_thresholds_valley,
          "Derive numeric valley thresholds from fitted guide GMM parameters",
          py::arg("weights"), py::arg("means"), py::arg("sigma"), py::arg("status"),
          py::arg("grid_size") = 256);

    m.def("sweep_guide_thresholds_quantile", &sweep_guide_thresholds_quantile,
          "Sweep quantile thresholds from fitted guide GMM parameters",
          py::arg("weights"), py::arg("means"), py::arg("sigma"), py::arg("status"),
          py::arg("bg_quantiles"), py::arg("fg_quantiles"));

    m.def("apply_guide_thresholds_sparse", &apply_guide_thresholds_sparse,
          "Apply per-guide thresholds to sparse counts",
          py::arg("X"), py::arg("background_thresholds"), py::arg("foreground_thresholds"));

    m.def("apply_guide_thresholds_backed_operator", &apply_guide_thresholds_backed_operator,
          "Apply per-guide thresholds to backed operators",
          py::arg("op"), py::arg("background_thresholds"), py::arg("foreground_thresholds"),
          py::arg("chunk_guides") = 256);
}
