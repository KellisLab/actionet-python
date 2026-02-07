// Pybind11 interface for `tools` module
// Organized by module header in the order imported.

#include "wp_utils.h"
#include "libactionet.hpp"

namespace py = pybind11;

// autocorrelation =====================================================================================================

py::dict autocorrelation_moran(py::object G, py::array_t<double> scores, int normalization_method = 1,
                                int perm_no = 30, int thread_no = 0) {
    arma::sp_mat G_sp = scipy_to_arma_sparse(G);
    arma::mat scores_mat = numpy_to_arma_mat(scores);

    arma::field<arma::vec> out = actionet::autocorrelation_Moran(G_sp, scores_mat, normalization_method, perm_no, thread_no);

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

    arma::field<arma::vec> out = actionet::autocorrelation_Geary(G_sp, scores_mat, normalization_method, perm_no, thread_no);

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

    arma::mat logPvals = actionet::computeGraphLabelEnrichment(G_sp, scores_mat, thread_no);

    return arma_mat_to_numpy(logPvals);
}

py::dict assess_enrichment(py::array_t<double> scores, py::object associations, int thread_no = 0) {
    arma::mat scores_mat = numpy_to_arma_mat(scores);
    arma::sp_mat assoc_sp = scipy_to_arma_sparse(associations);

    arma::field<arma::mat> res = actionet::assess_enrichment(scores_mat, assoc_sp, thread_no);

    py::dict out;
    out["logPvals"] = arma_mat_to_numpy(res(0));
    out["thresholds"] = arma_mat_to_numpy(res(1));

    return out;
}

// matrix_aggregate ====================================================================================================

py::array_t<double> compute_grouped_sums_sparse(py::object S, py::array_t<double> sample_assignments, int axis = 0) {
    arma::sp_mat S_sp = scipy_to_arma_sparse(S);
    arma::vec assignments_vec = numpy_to_arma_vec(sample_assignments);

    arma::mat pb = actionet::computeGroupedSums(S_sp, assignments_vec, axis);
    return arma_mat_to_numpy(pb);
}

py::array_t<double> compute_grouped_sums_dense(py::array_t<double> S, py::array_t<double> sample_assignments, int axis = 0) {
    arma::mat S_mat = numpy_to_arma_mat(S);
    arma::vec assignments_vec = numpy_to_arma_vec(sample_assignments);

    arma::mat pb = actionet::computeGroupedSums(S_mat, assignments_vec, axis);
    return arma_mat_to_numpy(pb);
}

py::array_t<double> compute_grouped_means_sparse(py::object S, py::array_t<double> sample_assignments, int axis = 0) {
    arma::sp_mat S_sp = scipy_to_arma_sparse(S);
    arma::vec assignments_vec = numpy_to_arma_vec(sample_assignments);

    arma::mat pb = actionet::computeGroupedMeans(S_sp, assignments_vec, axis);
    return arma_mat_to_numpy(pb);
}

py::array_t<double> compute_grouped_means_dense(py::array_t<double> S, py::array_t<double> sample_assignments, int axis = 0) {
    arma::mat S_mat = numpy_to_arma_mat(S);
    arma::vec assignments_vec = numpy_to_arma_vec(sample_assignments);

    arma::mat pb = actionet::computeGroupedMeans(S_mat, assignments_vec, axis);
    return arma_mat_to_numpy(pb);
}

py::array_t<double> compute_grouped_vars_sparse(py::object S, py::array_t<double> sample_assignments, int axis = 0) {
    arma::sp_mat S_sp = scipy_to_arma_sparse(S);
    arma::vec assignments_vec = numpy_to_arma_vec(sample_assignments);

    arma::mat pb = actionet::computeGroupedVars(S_sp, assignments_vec, axis);
    return arma_mat_to_numpy(pb);
}

py::array_t<double> compute_grouped_vars_dense(py::array_t<double> S, py::array_t<double> sample_assignments, int axis = 0) {
    arma::mat S_mat = numpy_to_arma_mat(S);
    arma::vec assignments_vec = numpy_to_arma_vec(sample_assignments);

    arma::mat pb = actionet::computeGroupedVars(S_mat, assignments_vec, axis);
    return arma_mat_to_numpy(pb);
}

// matrix_transform ====================================================================================================

py::object normalize_matrix_sparse(py::object X, unsigned int p = 1, unsigned int dim = 0) {
    arma::sp_mat X_sp = scipy_to_arma_sparse(X);
    arma::sp_mat Xn = actionet::normalizeMatrix(X_sp, p, dim);
    return arma_sparse_to_scipy(Xn);
}

py::array_t<double> normalize_matrix_dense(py::array_t<double> X, unsigned int p = 1, unsigned int dim = 0) {
    arma::mat X_mat = numpy_to_arma_mat(X);
    arma::mat Xn = actionet::normalizeMatrix(X_mat, p, dim);
    return arma_mat_to_numpy(Xn);
}

py::array_t<double> scale_matrix_dense(py::array_t<double> X, py::array_t<double> v, unsigned int dim = 0) {
    arma::mat X_mat = numpy_to_arma_mat(X);
    arma::vec v_vec = numpy_to_arma_vec(v);
    arma::mat Xs = actionet::scaleMatrix(X_mat, v_vec, dim);
    return arma_mat_to_numpy(Xs);
}

py::object scale_matrix_sparse(py::object X, py::array_t<double> v, unsigned int dim = 0) {
    arma::sp_mat X_sp = scipy_to_arma_sparse(X);
    arma::vec v_vec = numpy_to_arma_vec(v);
    arma::sp_mat Xs = actionet::scaleMatrix(X_sp, v_vec, dim);
    return arma_sparse_to_scipy(Xs);
}

py::object normalize_graph(py::object G, int norm_method = 0) {
    arma::sp_mat G_sp = scipy_to_arma_sparse(G);
    arma::sp_mat Gn = actionet::normalizeGraph(G_sp, norm_method);
    return arma_sparse_to_scipy(Gn);
}

// mwm =================================================================================================================

py::array_t<double> mwm_hungarian(py::array_t<double> G) {
    arma::mat G_mat = numpy_to_arma_mat(G);
    arma::mat G_matched = actionet::MWM_hungarian(G_mat);
    return arma_mat_to_numpy(G_matched);
}

py::array_t<int> mwm_rank1(py::array_t<double> u, py::array_t<double> v,
                             double u_threshold = 0, double v_threshold = 0) {
    arma::vec u_vec = numpy_to_arma_vec(u);
    arma::vec v_vec = numpy_to_arma_vec(v);

    arma::umat pairs = actionet::MWM_rank1(u_vec, v_vec, u_threshold, v_threshold);

    // Convert to 1-indexed and return as numpy array
    std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(pairs.n_rows), 2};
    auto result = py::array_t<int>(shape);
    auto buf = result.request();
    int* ptr = static_cast<int*>(buf.ptr);
    for (size_t i = 0; i < pairs.n_rows; ++i) {
        ptr[i * 2] = static_cast<int>(pairs(i, 0)) + 1;
        ptr[i * 2 + 1] = static_cast<int>(pairs(i, 1)) + 1;
    }

    return result;
}


// xicor ===============================================================================================================

py::array_t<double> xicor(py::array_t<double> xvec, py::array_t<double> yvec,
                           bool compute_pval = true, int seed = 0) {
    arma::vec x_vec = numpy_to_arma_vec(xvec);
    arma::vec y_vec = numpy_to_arma_vec(yvec);

    arma::vec res = actionet::xicor(std::move(x_vec), std::move(y_vec), compute_pval, seed);

    return arma_vec_to_numpy(res);
}

py::dict xicor_matrix(py::array_t<double> X, py::array_t<double> Y,
                       bool compute_pval = true, int seed = 0, int thread_no = 0) {
    arma::mat X_mat = numpy_to_arma_mat(X);
    arma::mat Y_mat = numpy_to_arma_mat(Y);

    arma::field<arma::mat> out = actionet::XICOR(X_mat, Y_mat, compute_pval, seed, thread_no);

    py::dict res;
    res["XI"] = arma_mat_to_numpy(out(0));
    res["Z"] = arma_mat_to_numpy(out(1));

    return res;
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

    m.def("compute_grouped_sums_dense", &compute_grouped_sums_dense, "Compute grouped sums (dense)",
          py::arg("S"), py::arg("sample_assignments"), py::arg("axis") = 0);

    m.def("compute_grouped_means_sparse", &compute_grouped_means_sparse, "Compute grouped means (sparse)",
          py::arg("S"), py::arg("sample_assignments"), py::arg("axis") = 0);

    m.def("compute_grouped_means_dense", &compute_grouped_means_dense, "Compute grouped means (dense)",
          py::arg("S"), py::arg("sample_assignments"), py::arg("axis") = 0);

    m.def("compute_grouped_vars_sparse", &compute_grouped_vars_sparse, "Compute grouped variances (sparse)",
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
}
