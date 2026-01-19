// Pybind11 interface for `decomposition` module
// Organized by module header in the order imported.

#include "wp_utils.h"
#include "libactionet.hpp"

namespace py = pybind11;

// orthogonalization ===================================================================================================

py::dict orthogonalize_batch_effect_sparse(py::object S, py::array_t<double> old_S_r,
                                             py::array_t<double> old_V, py::array_t<double> old_A,
                                             py::array_t<double> old_B, py::array_t<double> old_sigma,
                                             py::array_t<double> design) {
    arma::sp_mat S_sp = scipy_to_arma_sparse(S);
    arma::mat S_r_mat = numpy_to_arma_mat(old_S_r);
    arma::mat V_mat = numpy_to_arma_mat(old_V);
    arma::mat A_mat = numpy_to_arma_mat(old_A);
    arma::mat B_mat = numpy_to_arma_mat(old_B);
    arma::vec sigma_vec = numpy_to_arma_vec(old_sigma);
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
    arma::vec sigma_vec = numpy_to_arma_vec(old_sigma);
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
    arma::vec sigma_vec = numpy_to_arma_vec(old_sigma);
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

// =====================================================================================================================

void init_decomposition(py::module_ &m) {
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
}
