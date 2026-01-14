// Pybind11 interface for `action` module
// Organized by module header in the order imported.

#include "wp_utils.h"
#include "libactionet.hpp"

namespace py = pybind11;

// aa ==================================================================================================================

py::dict run_aa(py::array_t<double> A, py::array_t<double> W0, int max_it = 100, double tol = 1e-6) {
    arma::mat A_mat = numpy_to_arma_mat(A);
    arma::mat W0_mat = numpy_to_arma_mat(W0);

    arma::field<arma::mat> res = actionet::runAA(A_mat, W0_mat, max_it, tol);

    py::dict out;
    out["C"] = arma_mat_to_numpy(res(0));
    out["H"] = arma_mat_to_numpy(res(1));
    out["W"] = arma_mat_to_numpy(A_mat * res(0));

    return out;
}

// action_decomp =======================================================================================================

py::dict decomp_action(py::array_t<double> S_r, int k_min = 2, int k_max = 30,
                       int max_it = 100, double tol = 1e-16, int thread_no = 0) {
    arma::mat S_r_mat = numpy_to_arma_mat(S_r);

    actionet::ResACTION trace = actionet::decompACTION(S_r_mat, k_min, k_max, max_it, tol, thread_no);

    py::dict res;
    py::list C_list;
    py::list H_list;

    for (int i = k_min; i <= k_max; i++) {
        arma::mat cur_C = trace.C[i];
        arma::mat cur_H = trace.H[i];
        C_list.append(arma_mat_to_numpy(cur_C));
        H_list.append(arma_mat_to_numpy(cur_H));
    }

    res["C"] = C_list;
    res["H"] = H_list;

    return res;
}

py::dict run_action(py::array_t<double> S_r, int k_min = 2, int k_max = 30,
                    int max_it = 100, double tol = 1e-16,
                    double spec_th = -3.0, int min_obs = 3, int thread_no = 0) {
    arma::mat S_r_mat = numpy_to_arma_mat(S_r);

    arma::field<arma::mat> res = actionet::runACTION(
        S_r_mat, k_min, k_max, max_it, tol, spec_th, min_obs, thread_no
    );

    py::dict out;
    out["H_stacked"] = arma_mat_to_numpy(res(0));
    out["C_stacked"] = arma_mat_to_numpy(res(1));
    out["H_merged"] = arma_mat_to_numpy(res(2));
    out["C_merged"] = arma_mat_to_numpy(res(3));

    // Convert assigned_archetypes
    arma::vec assigned = arma::vec(res(4));
    auto assigned_arr = py::array_t<int>(assigned.n_elem);
    auto assigned_buf = assigned_arr.request();
    int* assigned_ptr = static_cast<int*>(assigned_buf.ptr);
    for (size_t i = 0; i < assigned.n_elem; ++i) {
        assigned_ptr[i] = static_cast<int>(assigned(i));
    }
    out["assigned_archetypes"] = assigned_arr;

    return out;
}

// action_post =========================================================================================================

py::dict collect_archetypes(py::list C_trace, py::list H_trace, double spec_th = -3.0, int min_obs = 3) {
    int n_list = py::len(H_trace);
    arma::field<arma::mat> C_trace_vec(n_list + 1);
    arma::field<arma::mat> H_trace_vec(n_list + 1);

    for (int i = 0; i < n_list; i++) {
        if (!C_trace[i].is_none() && !H_trace[i].is_none()) {
            C_trace_vec[i + 1] = numpy_to_arma_mat(C_trace[i].cast<py::array_t<double>>());
            H_trace_vec[i + 1] = numpy_to_arma_mat(H_trace[i].cast<py::array_t<double>>());
        }
    }

    actionet::ResCollectArch results = actionet::collectArchetypes(C_trace_vec, H_trace_vec, spec_th, min_obs);

    py::dict out;
    // Convert selected_archs to Python list (1-indexed)
    py::list selected_archs_list;
    for (size_t i = 0; i < results.selected_archs.n_elem; i++) {
        selected_archs_list.append(results.selected_archs(i) + 1);
    }
    out["selected_archs"] = selected_archs_list;
    out["C_stacked"] = arma_mat_to_numpy(results.C_stacked);
    out["H_stacked"] = arma_mat_to_numpy(results.H_stacked);

    return out;
}

py::dict merge_archetypes(py::array_t<double> S_r, py::array_t<double> C_stacked,
                          py::array_t<double> H_stacked, int thread_no = 0) {
    arma::mat S_r_mat = numpy_to_arma_mat(S_r);
    arma::mat C_stacked_mat = numpy_to_arma_mat(C_stacked);
    arma::mat H_stacked_mat = numpy_to_arma_mat(H_stacked);

    actionet::ResMergeArch results = actionet::mergeArchetypes(S_r_mat, C_stacked_mat, H_stacked_mat, thread_no);

    py::dict out;
    // Convert to 1-indexed
    auto selected_arr = py::array_t<int>(results.selected_archetypes.n_elem);
    auto selected_buf = selected_arr.request();
    int* selected_ptr = static_cast<int*>(selected_buf.ptr);
    for (size_t i = 0; i < results.selected_archetypes.n_elem; ++i) {
        selected_ptr[i] = static_cast<int>(results.selected_archetypes(i)) + 1;
    }
    out["selected_archetypes"] = selected_arr;

    out["C_merged"] = arma_mat_to_numpy(results.C_merged);
    out["H_merged"] = arma_mat_to_numpy(results.H_merged);

    auto assigned_arr = py::array_t<int>(results.assigned_archetypes.n_elem);
    auto assigned_buf = assigned_arr.request();
    int* assigned_ptr = static_cast<int*>(assigned_buf.ptr);
    for (size_t i = 0; i < results.assigned_archetypes.n_elem; ++i) {
        assigned_ptr[i] = static_cast<int>(results.assigned_archetypes(i)) + 1;
    }
    out["assigned_archetypes"] = assigned_arr;

    return out;
}

// reduce_kernel =======================================================================================================

py::dict reduce_kernel_sparse(py::object S, int k = 50, int svd_alg = 0,
                               int max_it = 0, int seed = 0, bool verbose = true) {
    arma::sp_mat S_sp = scipy_to_arma_sparse(S);
    arma::field<arma::mat> res = actionet::reduceKernel(S_sp, k, svd_alg, max_it, seed, verbose);

    py::dict out;
    out["S_r"] = arma_mat_to_numpy(res(0));
    out["sigma"] = arma_mat_to_numpy(res(1));
    out["V"] = arma_mat_to_numpy(res(2));
    out["A"] = arma_mat_to_numpy(res(3));
    out["B"] = arma_mat_to_numpy(res(4));
    return out;
}

py::dict reduce_kernel_dense(py::array_t<double> S, int k = 50, int svd_alg = 0,
                              int max_it = 0, int seed = 0, bool verbose = true) {
    arma::mat S_mat = numpy_to_arma_mat(S);
    arma::field<arma::mat> res = actionet::reduceKernel(S_mat, k, svd_alg, max_it, seed, verbose);

    py::dict out;
    out["S_r"] = arma_mat_to_numpy(res(0));
    out["sigma"] = arma_mat_to_numpy(res(1));
    out["V"] = arma_mat_to_numpy(res(2));
    out["A"] = arma_mat_to_numpy(res(3));
    out["B"] = arma_mat_to_numpy(res(4));
    return out;
}

// simplex_regression ==================================================================================================

py::array_t<double> run_simplex_regression(py::array_t<double> A, py::array_t<double> B, bool computeXtX = false) {
    arma::mat A_mat = numpy_to_arma_mat(A);
    arma::mat B_mat = numpy_to_arma_mat(B);

    arma::mat X = actionet::runSimplexRegression(A_mat, B_mat, computeXtX);

    return arma_mat_to_numpy(X);
}

// spa =================================================================================================================

py::dict run_spa(py::array_t<double> A, int k) {
    arma::mat A_mat = numpy_to_arma_mat(A);

    actionet::ResSPA res = actionet::runSPA(A_mat, k);

    // Convert to 1-indexed for Python
    auto cols_arr = py::array_t<int>(k);
    auto cols_buf = cols_arr.request();
    int* cols_ptr = static_cast<int*>(cols_buf.ptr);
    for (int i = 0; i < k; i++) {
        cols_ptr[i] = static_cast<int>(res.selected_cols[i]) + 1;
    }

    py::dict out;
    out["selected_cols"] = cols_arr;
    out["norms"] = arma_vec_to_numpy(res.column_norms);

    return out;
}

// =====================================================================================================================

void init_action(py::module_ &m) {
    // aa
    m.def("run_aa", &run_aa, "Run archetypal analysis",
          py::arg("A"), py::arg("W0"), py::arg("max_it") = 100, py::arg("tol") = 1e-6);

    // action_decomp
    m.def("decomp_action", &decomp_action, "Decompose ACTION (returns trace)",
          py::arg("S_r"), py::arg("k_min") = 2, py::arg("k_max") = 30,
          py::arg("max_it") = 100, py::arg("tol") = 1e-16, py::arg("thread_no") = 0);

    m.def("run_action", &run_action, "Run ACTION decomposition",
          py::arg("S_r"), py::arg("k_min") = 2, py::arg("k_max") = 30,
          py::arg("max_it") = 100, py::arg("tol") = 1e-16,
          py::arg("spec_th") = -3.0, py::arg("min_obs") = 3, py::arg("thread_no") = 0);

    // action_post
    m.def("collect_archetypes", &collect_archetypes, "Collect and filter archetypes",
          py::arg("C_trace"), py::arg("H_trace"), py::arg("spec_th") = -3.0, py::arg("min_obs") = 3);

    m.def("merge_archetypes", &merge_archetypes, "Merge redundant archetypes",
          py::arg("S_r"), py::arg("C_stacked"), py::arg("H_stacked"), py::arg("thread_no") = 0);

    // reduce_kernel
    m.def("reduce_kernel_sparse", &reduce_kernel_sparse, "Reduce kernel (sparse)",
          py::arg("S"), py::arg("k") = 50, py::arg("svd_alg") = 0,
          py::arg("max_it") = 0, py::arg("seed") = 0, py::arg("verbose") = true);

    m.def("reduce_kernel_dense", &reduce_kernel_dense, "Reduce kernel (dense)",
          py::arg("S"), py::arg("k") = 50, py::arg("svd_alg") = 0,
          py::arg("max_it") = 0, py::arg("seed") = 0, py::arg("verbose") = true);

    // simplex_regression
    m.def("run_simplex_regression", &run_simplex_regression, "Simplex-constrained regression",
          py::arg("A"), py::arg("B"), py::arg("computeXtX") = false);

    // spa
    m.def("run_spa", &run_spa, "Run successive projections algorithm",
          py::arg("A"), py::arg("k"));
}
