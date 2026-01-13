#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Armadillo includes (libactionet uses Armadillo)
#define ARMA_64BIT_WORD
#define ARMA_DONT_USE_WRAPPER
#include <armadillo>

// libactionet main header
#include "libactionet.hpp"

namespace py = pybind11;

// ========== Conversion Utilities ==========

// Convert NumPy array to Armadillo dense matrix
arma::mat numpy_to_arma_mat(py::array_t<double> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Expected 2D array");
    }
    
    auto ptr = static_cast<double*>(buf.ptr);
    // Copy data (Armadillo uses column-major, NumPy default is row-major)
    arma::mat mat(buf.shape[0], buf.shape[1]);
    for (size_t i = 0; i < buf.shape[0]; ++i) {
        for (size_t j = 0; j < buf.shape[1]; ++j) {
            mat(i, j) = ptr[i * buf.shape[1] + j];
        }
    }
    return mat;
}

// Convert SciPy sparse matrix to Armadillo sparse matrix
arma::sp_mat scipy_to_arma_sparse(py::object scipy_sparse) {
    // Convert to CSC format
    py::object csc = scipy_sparse.attr("tocsc")();
    
    auto data = csc.attr("data").cast<py::array_t<double>>();
    auto indices = csc.attr("indices").cast<py::array_t<int>>();
    auto indptr = csc.attr("indptr").cast<py::array_t<int>>();
    auto shape = csc.attr("shape").cast<std::pair<int, int>>();
    
    auto data_ptr = data.data();
    auto indices_ptr = indices.data();
    auto indptr_ptr = indptr.data();
    
    arma::umat locations(2, data.size());
    arma::vec values(data.size());
    
    size_t idx = 0;
    for (int col = 0; col < shape.second; ++col) {
        for (int j = indptr_ptr[col]; j < indptr_ptr[col + 1]; ++j) {
            locations(0, idx) = indices_ptr[j];  // row
            locations(1, idx) = col;              // col
            values(idx) = data_ptr[j];
            ++idx;
        }
    }
    
    return arma::sp_mat(locations, values, shape.first, shape.second);
}

// Convert Armadillo dense matrix to NumPy array
py::array_t<double> arma_mat_to_numpy(const arma::mat& mat) {
    py::array_t<double> arr({mat.n_rows, mat.n_cols});
    auto buf = arr.request();
    double* ptr = static_cast<double*>(buf.ptr);
    
    for (size_t i = 0; i < mat.n_rows; ++i) {
        for (size_t j = 0; j < mat.n_cols; ++j) {
            ptr[i * mat.n_cols + j] = mat(i, j);
        }
    }
    return arr;
}

// Convert Armadillo sparse matrix to SciPy sparse matrix
py::object arma_sparse_to_scipy(const arma::sp_mat& sp_mat) {
    py::module scipy_sparse = py::module::import("scipy.sparse");
    
    std::vector<double> data;
    std::vector<int> rows;
    std::vector<int> cols;
    
    for (arma::sp_mat::const_iterator it = sp_mat.begin(); it != sp_mat.end(); ++it) {
        data.push_back(*it);
        rows.push_back(it.row());
        cols.push_back(it.col());
    }
    
    return scipy_sparse.attr("coo_matrix")(
        py::make_tuple(
            py::array_t<double>(data.size(), data.data()),
            py::make_tuple(
                py::array_t<int>(rows.size(), rows.data()),
                py::array_t<int>(cols.size(), cols.data())
            )
        ),
        py::make_tuple(sp_mat.n_rows, sp_mat.n_cols)
    ).attr("tocsr")();
}

// ========== ACTION MODULE BINDINGS ==========

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

// ========== NETWORK MODULE BINDINGS ==========

py::object build_network(py::array_t<double> H, std::string algorithm = "k*nn",
                         std::string distance_metric = "jsd", double density = 1.0,
                         int thread_no = 0, double M = 16, double ef_construction = 200,
                         double ef = 50, bool mutual_edges_only = true, int k = 10) {
    arma::mat H_mat = numpy_to_arma_mat(H);
    
    arma::sp_mat G = actionet::buildNetwork(
        H_mat, algorithm, distance_metric, density, thread_no, 
        M, ef_construction, ef, mutual_edges_only, k
    );
    
    return arma_sparse_to_scipy(G);
}

py::array_t<double> compute_network_diffusion(py::object G, py::array_t<double> X0,
                                               double alpha = 0.85, int max_it = 5,
                                               int thread_no = 0, bool approx = false,
                                               int norm_method = 0, double tol = 1e-8) {
    arma::sp_mat G_sp = scipy_to_arma_sparse(G);
    arma::mat X0_mat = numpy_to_arma_mat(X0);
    
    arma::mat X = actionet::computeNetworkDiffusion(
        G_sp, X0_mat, alpha, max_it, thread_no, approx, norm_method, tol
    );
    
    return arma_mat_to_numpy(X);
}

// ========== ANNOTATION MODULE BINDINGS ==========

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

// ========== SVD MODULE BINDINGS ==========

py::dict run_svd_sparse(py::object A, int k = 30, int max_it = 0, int seed = 0,
                        int algorithm = 0, bool verbose = true) {
    arma::sp_mat A_sp = scipy_to_arma_sparse(A);
    arma::field<arma::mat> res = actionet::runSVD(A_sp, k, max_it, seed, algorithm, verbose);
    
    py::dict out;
    out["u"] = arma_mat_to_numpy(res(0));
    out["d"] = arma_mat_to_numpy(res(1));
    out["v"] = arma_mat_to_numpy(res(2));
    return out;
}

// ========== BATCH CORRECTION MODULE BINDINGS ==========

py::dict orthogonalize_batch_effect_sparse(py::object S, py::array_t<double> old_S_r,
                                             py::array_t<double> old_V, py::array_t<double> old_A,
                                             py::array_t<double> old_B, py::array_t<double> old_sigma,
                                             py::array_t<double> design) {
    arma::sp_mat S_sp = scipy_to_arma_sparse(S);
    arma::mat S_r_mat = numpy_to_arma_mat(old_S_r);
    arma::mat V_mat = numpy_to_arma_mat(old_V);
    arma::mat A_mat = numpy_to_arma_mat(old_A);
    arma::mat B_mat = numpy_to_arma_mat(old_B);
    arma::vec sigma_vec = arma::vec(old_sigma.data(), old_sigma.size(), false, true);
    arma::mat design_mat = numpy_to_arma_mat(design);

    // Prepare SVD results field
    arma::field<arma::mat> SVD_results(5);
    SVD_results(0) = V_mat;
    SVD_results(1) = sigma_vec;
    SVD_results(2) = S_r_mat;
    for (size_t i = 0; i < sigma_vec.n_elem; i++) {
        SVD_results(2).col(i) /= sigma_vec(i);
    }
    SVD_results(3) = A_mat;
    SVD_results(4) = B_mat;

    arma::field<arma::mat> orth_reduction = actionet::orthogonalizeBatchEffect(S_sp, SVD_results, design_mat);

    arma::vec sigma = orth_reduction(1).col(0);
    arma::mat V = orth_reduction(2);
    for (size_t i = 0; i < V.n_cols; i++) {
        V.col(i) *= sigma(i);
    }

    py::dict out;
    out["V"] = arma_mat_to_numpy(orth_reduction(0));
    out["sigma"] = arma_mat_to_numpy(sigma);
    out["S_r"] = arma_mat_to_numpy(V.t());
    out["A"] = arma_mat_to_numpy(orth_reduction(3));
    out["B"] = arma_mat_to_numpy(orth_reduction(4));
    return out;
}

py::dict orthogonalize_batch_effect_dense(py::array_t<double> S, py::array_t<double> old_S_r,
                                            py::array_t<double> old_V, py::array_t<double> old_A,
                                            py::array_t<double> old_B, py::array_t<double> old_sigma,
                                            py::array_t<double> design) {
    arma::mat S_mat = numpy_to_arma_mat(S);
    arma::mat S_r_mat = numpy_to_arma_mat(old_S_r);
    arma::mat V_mat = numpy_to_arma_mat(old_V);
    arma::mat A_mat = numpy_to_arma_mat(old_A);
    arma::mat B_mat = numpy_to_arma_mat(old_B);
    arma::vec sigma_vec = arma::vec(old_sigma.data(), old_sigma.size(), false, true);
    arma::mat design_mat = numpy_to_arma_mat(design);

    // Prepare SVD results field
    arma::field<arma::mat> SVD_results(5);
    SVD_results(0) = V_mat;
    SVD_results(1) = sigma_vec;
    SVD_results(2) = S_r_mat;
    for (size_t i = 0; i < sigma_vec.n_elem; i++) {
        SVD_results(2).col(i) /= sigma_vec(i);
    }
    SVD_results(3) = A_mat;
    SVD_results(4) = B_mat;

    arma::field<arma::mat> orth_reduction = actionet::orthogonalizeBatchEffect(S_mat, SVD_results, design_mat);

    arma::vec sigma = orth_reduction(1).col(0);
    arma::mat V = orth_reduction(2);
    for (size_t i = 0; i < V.n_cols; i++) {
        V.col(i) *= sigma(i);
    }

    py::dict out;
    out["V"] = arma_mat_to_numpy(orth_reduction(0));
    out["sigma"] = arma_mat_to_numpy(sigma);
    out["S_r"] = arma_mat_to_numpy(V.t());
    out["A"] = arma_mat_to_numpy(orth_reduction(3));
    out["B"] = arma_mat_to_numpy(orth_reduction(4));
    return out;
}

py::dict orthogonalize_basal_sparse(py::object S, py::array_t<double> old_S_r,
                                      py::array_t<double> old_V, py::array_t<double> old_A,
                                      py::array_t<double> old_B, py::array_t<double> old_sigma,
                                      py::array_t<double> basal) {
    arma::sp_mat S_sp = scipy_to_arma_sparse(S);
    arma::mat S_r_mat = numpy_to_arma_mat(old_S_r);
    arma::mat V_mat = numpy_to_arma_mat(old_V);
    arma::mat A_mat = numpy_to_arma_mat(old_A);
    arma::mat B_mat = numpy_to_arma_mat(old_B);
    arma::vec sigma_vec = arma::vec(old_sigma.data(), old_sigma.size(), false, true);
    arma::mat basal_mat = numpy_to_arma_mat(basal);

    // Prepare SVD results field
    arma::field<arma::mat> SVD_results(5);
    SVD_results(0) = V_mat;
    SVD_results(1) = sigma_vec;
    SVD_results(2) = S_r_mat;
    for (size_t i = 0; i < sigma_vec.n_elem; i++) {
        SVD_results(2).col(i) /= sigma_vec(i);
    }
    SVD_results(3) = A_mat;
    SVD_results(4) = B_mat;

    arma::field<arma::mat> orth_reduction = actionet::orthogonalizeBasal(S_sp, SVD_results, basal_mat);

    arma::vec sigma = orth_reduction(1).col(0);
    arma::mat V = orth_reduction(2);
    for (size_t i = 0; i < V.n_cols; i++) {
        V.col(i) *= sigma(i);
    }

    py::dict out;
    out["V"] = arma_mat_to_numpy(orth_reduction(0));
    out["sigma"] = arma_mat_to_numpy(sigma);
    out["S_r"] = arma_mat_to_numpy(V.t());
    out["A"] = arma_mat_to_numpy(orth_reduction(3));
    out["B"] = arma_mat_to_numpy(orth_reduction(4));
    return out;
}

// ========== VISUALIZATION MODULE BINDINGS ==========

py::array_t<double> layout_network(py::object G, py::array_t<double> initial_coords,
                                    std::string method = "umap", unsigned int n_components = 2,
                                    float spread = 1.0, float min_dist = 1.0,
                                    unsigned int n_epochs = 0, int seed = 0,
                                    int thread_no = 0, bool verbose = true) {
    arma::sp_mat G_sp = scipy_to_arma_sparse(G);
    arma::mat init_mat = numpy_to_arma_mat(initial_coords);
    
    arma::mat coords = actionet::layoutNetwork(
        G_sp, init_mat, method, n_components, spread, min_dist,
        n_epochs, 1.0f, 1.0f, 5.0f, false, true, true, 1,
        seed, thread_no, verbose, 0.0f, 0.0f, "adam", -1.0f, 0.5f, 0.9f, 1e-7f
    );
    
    return arma_mat_to_numpy(coords);
}

// ========== MODULE DEFINITION ==========

PYBIND11_MODULE(_core, m) {
    m.doc() = "ACTIONet C++ core bindings";
    
    // Action module
    m.def("run_action", &run_action, "Run ACTION decomposition",
          py::arg("S_r"), py::arg("k_min") = 2, py::arg("k_max") = 30,
          py::arg("max_it") = 100, py::arg("tol") = 1e-16,
          py::arg("spec_th") = -3.0, py::arg("min_obs") = 3, py::arg("thread_no") = 0);
    
    m.def("reduce_kernel_sparse", &reduce_kernel_sparse, "Reduce kernel (sparse)",
          py::arg("S"), py::arg("k") = 50, py::arg("svd_alg") = 0,
          py::arg("max_it") = 0, py::arg("seed") = 0, py::arg("verbose") = true);
    
    m.def("reduce_kernel_dense", &reduce_kernel_dense, "Reduce kernel (dense)",
          py::arg("S"), py::arg("k") = 50, py::arg("svd_alg") = 0,
          py::arg("max_it") = 0, py::arg("seed") = 0, py::arg("verbose") = true);
    
    // Network module
    m.def("build_network", &build_network, "Build cell-cell network",
          py::arg("H"), py::arg("algorithm") = "k*nn", py::arg("distance_metric") = "jsd",
          py::arg("density") = 1.0, py::arg("thread_no") = 0, py::arg("M") = 16,
          py::arg("ef_construction") = 200, py::arg("ef") = 50,
          py::arg("mutual_edges_only") = true, py::arg("k") = 10);
    
    m.def("compute_network_diffusion", &compute_network_diffusion,
          "Compute network diffusion",
          py::arg("G"), py::arg("X0"), py::arg("alpha") = 0.85, py::arg("max_it") = 5,
          py::arg("thread_no") = 0, py::arg("approx") = false,
          py::arg("norm_method") = 0, py::arg("tol") = 1e-8);
    
    // Annotation module
    m.def("compute_feature_specificity_sparse", &compute_feature_specificity_sparse,
          "Compute feature specificity",
          py::arg("S"), py::arg("labels"), py::arg("thread_no") = 0);
    
    // SVD module
    m.def("run_svd_sparse", &run_svd_sparse, "Run SVD (sparse)",
          py::arg("A"), py::arg("k") = 30, py::arg("max_it") = 0, py::arg("seed") = 0,
          py::arg("algorithm") = 0, py::arg("verbose") = true);
    
    // Batch correction module
    m.def("orthogonalize_batch_effect_sparse", &orthogonalize_batch_effect_sparse,
          "Orthogonalize batch effects (sparse)",
          py::arg("S"), py::arg("old_S_r"), py::arg("old_V"), py::arg("old_A"),
          py::arg("old_B"), py::arg("old_sigma"), py::arg("design"));

    m.def("orthogonalize_batch_effect_dense", &orthogonalize_batch_effect_dense,
          "Orthogonalize batch effects (dense)",
          py::arg("S"), py::arg("old_S_r"), py::arg("old_V"), py::arg("old_A"),
          py::arg("old_B"), py::arg("old_sigma"), py::arg("design"));

    m.def("orthogonalize_basal_sparse", &orthogonalize_basal_sparse,
          "Orthogonalize basal expression (sparse)",
          py::arg("S"), py::arg("old_S_r"), py::arg("old_V"), py::arg("old_A"),
          py::arg("old_B"), py::arg("old_sigma"), py::arg("basal"));

    // Visualization module
    m.def("layout_network", &layout_network, "Layout network (UMAP/t-SNE)",
          py::arg("G"), py::arg("initial_coords"), py::arg("method") = "umap",
          py::arg("n_components") = 2, py::arg("spread") = 1.0, py::arg("min_dist") = 1.0,
          py::arg("n_epochs") = 0, py::arg("seed") = 0,
          py::arg("thread_no") = 0, py::arg("verbose") = true);
}
