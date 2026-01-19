// Pybind11 interface for `network` module
// Organized by module header in the order imported.

#include "wp_utils.h"
#include "libactionet.hpp"

namespace py = pybind11;

// build_network =======================================================================================================

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

// label_propagation ===================================================================================================

py::array_t<double> run_lpa(py::object G, py::array_t<double> labels, double lambda_param = 1.0, int iters = 3,
                             double sig_threshold = 3.0, py::object fixed_labels = py::none(), int thread_no = 0) {
    arma::sp_mat G_sp = scipy_to_arma_sparse(G);
    arma::vec labels_vec = numpy_to_arma_vec(labels);

    arma::uvec fixed_labels_vec;
    if (!fixed_labels.is_none()) {
        auto fixed_arr = fixed_labels.cast<py::array_t<int>>();
        auto fixed_buf = fixed_arr.request();
        auto fixed_ptr = static_cast<int*>(fixed_buf.ptr);
        fixed_labels_vec.set_size(fixed_buf.size);
        for (size_t i = 0; i < fixed_buf.size; i++) {
            fixed_labels_vec(i) = fixed_ptr[i] - 1;  // Convert to 0-indexed
        }
    }

    arma::vec new_labels = actionet::runLPA(G_sp, labels_vec, lambda_param, iters, sig_threshold, fixed_labels_vec, thread_no);

    return arma_vec_to_numpy(new_labels);
}

// network_diffusion ===================================================================================================

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

// network_measures ====================================================================================================

py::array_t<int> compute_coreness(py::object G) {
    arma::sp_mat G_sp = scipy_to_arma_sparse(G);

    arma::uvec core_num = actionet::computeCoreness(G_sp);

    auto result = py::array_t<int>(core_num.n_elem);
    auto buf = result.request();
    int* ptr = static_cast<int*>(buf.ptr);
    for (size_t i = 0; i < core_num.n_elem; ++i) {
        ptr[i] = static_cast<int>(core_num(i));
    }

    return result;
}

py::array_t<double> compute_archetype_centrality(py::object G, py::array_t<int> sample_assignments) {
    arma::sp_mat G_sp = scipy_to_arma_sparse(G);

    auto assignments_buf = sample_assignments.request();
    auto assignments_ptr = static_cast<int*>(assignments_buf.ptr);
    arma::uvec assignments_vec(assignments_buf.size);
    for (size_t i = 0; i < assignments_buf.size; ++i) {
        assignments_vec(i) = assignments_ptr[i];
    }

    arma::vec conn = actionet::computeArchetypeCentrality(G_sp, assignments_vec);

    return arma_vec_to_numpy(conn);
}

// =====================================================================================================================

void init_network(py::module_ &m) {
    // build_network
    m.def("build_network", &build_network, "Build cell-cell network",
          py::arg("H"), py::arg("algorithm") = "k*nn", py::arg("distance_metric") = "jsd",
          py::arg("density") = 1.0, py::arg("thread_no") = 0, py::arg("M") = 16,
          py::arg("ef_construction") = 200, py::arg("ef") = 50,
          py::arg("mutual_edges_only") = true, py::arg("k") = 10);

    // label_propagation
    m.def("run_lpa", &run_lpa, "Run label propagation algorithm",
          py::arg("G"), py::arg("labels"), py::arg("lambda_param") = 1.0, py::arg("iters") = 3,
          py::arg("sig_threshold") = 3.0, py::arg("fixed_labels") = py::none(), py::arg("thread_no") = 0);

    // network_diffusion
    m.def("compute_network_diffusion", &compute_network_diffusion,
          "Compute network diffusion",
          py::arg("G"), py::arg("X0"), py::arg("alpha") = 0.85, py::arg("max_it") = 5,
          py::arg("thread_no") = 0, py::arg("approx") = false,
          py::arg("norm_method") = 0, py::arg("tol") = 1e-8);

    // network_measures
    m.def("compute_coreness", &compute_coreness, "Compute graph coreness",
          py::arg("G"));

    m.def("compute_archetype_centrality", &compute_archetype_centrality,
          "Compute archetype centrality",
          py::arg("G"), py::arg("sample_assignments"));
}
