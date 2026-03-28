// Pybind11 interface for `decomposition` module
// Organized by module header in the order imported.

#include "wp_utils.h"
#include "libactionet.hpp"
#include "io/backed_h5ad/create_backed_operator.hpp"

namespace py = pybind11;

// orthogonalization ===================================================================================================

py::dict orthogonalize_batch_effect_sparse(py::object S, py::array_t<double> old_S_r,
                                             py::array_t<double> old_U, py::array_t<double> old_A,
                                             py::array_t<double> old_B, py::array_t<double> old_sigma,
                                             py::array_t<double> design) {
    arma::sp_mat S_sp = scipy_to_arma_sparse(S);
    arma::mat S_r_mat = numpy_to_arma_mat(old_S_r);
    arma::mat U_mat = numpy_to_arma_mat(old_U);
    arma::mat A_mat = numpy_to_arma_mat(old_A);
    arma::mat B_mat = numpy_to_arma_mat(old_B);
    arma::vec sigma_vec = numpy_to_arma_vec(old_sigma);
    arma::mat design_mat = numpy_to_arma_mat(design);

    // Plan 02 public layout: {S_r (cells x k), sigma, U (genes x k), A, B}
    arma::field<arma::mat> SVD_results(5);
    SVD_results(0) = S_r_mat;
    SVD_results(1) = sigma_vec;
    SVD_results(2) = U_mat;
    SVD_results(3) = A_mat;
    SVD_results(4) = B_mat;

    arma::field<arma::mat> orth_reduction;
    {
        py::gil_scoped_release release;
        orth_reduction = actionet::orthogonalizeBatchEffect(S_sp, SVD_results, design_mat);
    }

    py::dict out;
    out["S_r"]    = arma_mat_to_numpy_c(orth_reduction(0));
    out["sigma"]  = arma_vec_to_numpy(arma::vec(orth_reduction(1)));
    out["U"]      = arma_mat_to_numpy_c(orth_reduction(2));
    out["A"]      = arma_mat_to_numpy_c(orth_reduction(3));
    out["B"]      = arma_mat_to_numpy_c(orth_reduction(4));
    return out;
}

py::dict orthogonalize_batch_effect_dense(py::array_t<double> S, py::array_t<double> old_S_r,
                                            py::array_t<double> old_U, py::array_t<double> old_A,
                                            py::array_t<double> old_B, py::array_t<double> old_sigma,
                                            py::array_t<double> design) {
    arma::mat S_mat = numpy_to_arma_mat(S);
    arma::mat S_r_mat = numpy_to_arma_mat(old_S_r);
    arma::mat U_mat = numpy_to_arma_mat(old_U);
    arma::mat A_mat = numpy_to_arma_mat(old_A);
    arma::mat B_mat = numpy_to_arma_mat(old_B);
    arma::vec sigma_vec = numpy_to_arma_vec(old_sigma);
    arma::mat design_mat = numpy_to_arma_mat(design);

    // Plan 02 public layout: {S_r (cells x k), sigma, U (genes x k), A, B}
    arma::field<arma::mat> SVD_results(5);
    SVD_results(0) = S_r_mat;
    SVD_results(1) = sigma_vec;
    SVD_results(2) = U_mat;
    SVD_results(3) = A_mat;
    SVD_results(4) = B_mat;

    arma::field<arma::mat> orth_reduction;
    {
        py::gil_scoped_release release;
        orth_reduction = actionet::orthogonalizeBatchEffect(S_mat, SVD_results, design_mat);
    }

    py::dict out;
    out["S_r"]    = arma_mat_to_numpy_c(orth_reduction(0));
    out["sigma"]  = arma_vec_to_numpy(arma::vec(orth_reduction(1)));
    out["U"]      = arma_mat_to_numpy_c(orth_reduction(2));
    out["A"]      = arma_mat_to_numpy_c(orth_reduction(3));
    out["B"]      = arma_mat_to_numpy_c(orth_reduction(4));
    return out;
}

py::dict orthogonalize_basal_sparse(py::object S, py::array_t<double> old_S_r,
                                      py::array_t<double> old_U, py::array_t<double> old_A,
                                      py::array_t<double> old_B, py::array_t<double> old_sigma,
                                      py::array_t<double> basal) {
    arma::sp_mat S_sp = scipy_to_arma_sparse(S);
    arma::mat S_r_mat = numpy_to_arma_mat(old_S_r);
    arma::mat U_mat = numpy_to_arma_mat(old_U);
    arma::mat A_mat = numpy_to_arma_mat(old_A);
    arma::mat B_mat = numpy_to_arma_mat(old_B);
    arma::vec sigma_vec = numpy_to_arma_vec(old_sigma);
    arma::mat basal_mat = numpy_to_arma_mat(basal);

    // Plan 02 public layout: {S_r (cells x k), sigma, U (genes x k), A, B}
    arma::field<arma::mat> SVD_results(5);
    SVD_results(0) = S_r_mat;
    SVD_results(1) = sigma_vec;
    SVD_results(2) = U_mat;
    SVD_results(3) = A_mat;
    SVD_results(4) = B_mat;

    arma::field<arma::mat> orth_reduction;
    {
        py::gil_scoped_release release;
        orth_reduction = actionet::orthogonalizeBasal(S_sp, SVD_results, basal_mat);
    }

    py::dict out;
    out["S_r"]    = arma_mat_to_numpy_c(orth_reduction(0));
    out["sigma"]  = arma_vec_to_numpy(arma::vec(orth_reduction(1)));
    out["U"]      = arma_mat_to_numpy_c(orth_reduction(2));
    out["A"]      = arma_mat_to_numpy_c(orth_reduction(3));
    out["B"]      = arma_mat_to_numpy_c(orth_reduction(4));
    return out;
}

py::dict orthogonalize_basal_dense(py::array_t<double> S, py::array_t<double> old_S_r,
                                     py::array_t<double> old_U, py::array_t<double> old_A,
                                     py::array_t<double> old_B, py::array_t<double> old_sigma,
                                     py::array_t<double> basal) {
    arma::mat S_mat = numpy_to_arma_mat(S);
    arma::mat S_r_mat = numpy_to_arma_mat(old_S_r);
    arma::mat U_mat = numpy_to_arma_mat(old_U);
    arma::mat A_mat = numpy_to_arma_mat(old_A);
    arma::mat B_mat = numpy_to_arma_mat(old_B);
    arma::vec sigma_vec = numpy_to_arma_vec(old_sigma);
    arma::mat basal_mat = numpy_to_arma_mat(basal);

    // Plan 02 public layout: {S_r (cells x k), sigma, U (genes x k), A, B}
    arma::field<arma::mat> SVD_results(5);
    SVD_results(0) = S_r_mat;
    SVD_results(1) = sigma_vec;
    SVD_results(2) = U_mat;
    SVD_results(3) = A_mat;
    SVD_results(4) = B_mat;

    arma::field<arma::mat> orth_reduction;
    {
        py::gil_scoped_release release;
        orth_reduction = actionet::orthogonalizeBasal(S_mat, SVD_results, basal_mat);
    }

    py::dict out;
    out["S_r"]    = arma_mat_to_numpy_c(orth_reduction(0));
    out["sigma"]  = arma_vec_to_numpy(arma::vec(orth_reduction(1)));
    out["U"]      = arma_mat_to_numpy_c(orth_reduction(2));
    out["A"]      = arma_mat_to_numpy_c(orth_reduction(3));
    out["B"]      = arma_mat_to_numpy_c(orth_reduction(4));
    return out;
}

// svd_main ============================================================================================================

py::dict perturbed_svd(py::array_t<double> u, py::array_t<double> d, py::array_t<double> v,
                       py::array_t<double> A, py::array_t<double> B) {
    arma::mat u_mat = numpy_to_arma_mat(u);
    arma::vec d_vec = numpy_to_arma_vec(d);
    arma::mat v_mat = numpy_to_arma_mat(v);
    arma::mat A_mat = numpy_to_arma_mat(A);
    arma::mat B_mat = numpy_to_arma_mat(B);

    arma::field<arma::mat> SVD_results(3);
    SVD_results(0) = u_mat;
    SVD_results(1) = d_vec;
    SVD_results(2) = v_mat;

    arma::field<arma::mat> perturbed;
    {
        py::gil_scoped_release release;
        perturbed = actionet::perturbedSVD(SVD_results, A_mat, B_mat);
    }

    py::dict out;
    out["u"] = arma_mat_to_numpy(perturbed(0));
    out["d"] = arma_vec_to_numpy(perturbed(1).col(0));
    out["v"] = arma_mat_to_numpy(perturbed(2));
    return out;
}

py::dict perturbed_svd_with_prior(py::array_t<double> u, py::array_t<double> d, py::array_t<double> v,
                                  py::array_t<double> old_A, py::array_t<double> old_B,
                                  py::array_t<double> A_new, py::array_t<double> B_new) {
    arma::mat u_mat = numpy_to_arma_mat(u);
    arma::vec d_vec = numpy_to_arma_vec(d);
    arma::mat v_mat = numpy_to_arma_mat(v);
    arma::mat old_A_mat = numpy_to_arma_mat(old_A);
    arma::mat old_B_mat = numpy_to_arma_mat(old_B);
    arma::mat A_new_mat = numpy_to_arma_mat(A_new);
    arma::mat B_new_mat = numpy_to_arma_mat(B_new);

    actionet::SVDResult svd;
    svd.U = u_mat;
    svd.sigma = d_vec;
    svd.V = v_mat;

    actionet::PerturbedSVDResult prior;
    const actionet::PerturbedSVDResult* prior_ptr = nullptr;
    if (old_A_mat.n_elem > 0 && old_B_mat.n_elem > 0) {
        prior.A = old_A_mat;
        prior.B = old_B_mat;
        prior_ptr = &prior;
    }

    actionet::PerturbedSVDResult perturbed;
    {
        py::gil_scoped_release release;
        perturbed = actionet::perturbedSVD(svd, A_new_mat, B_new_mat, prior_ptr);
    }

    py::dict out;
    out["u"] = arma_mat_to_numpy(perturbed.U);
    out["d"] = arma_vec_to_numpy(perturbed.sigma);
    out["v"] = arma_mat_to_numpy(perturbed.V);
    out["A"] = arma_mat_to_numpy(perturbed.A);
    out["B"] = arma_mat_to_numpy(perturbed.B);
    return out;
}

// svd_main ============================================================================================================

py::dict run_svd_sparse(py::object A, int k = 30, int max_it = 0, int seed = 0,
                        int algorithm = 0, bool verbose = true) {
    arma::sp_mat A_sp = scipy_to_arma_sparse(A);
    arma::field<arma::mat> res;
    {
        py::gil_scoped_release release;
        res = actionet::runSVD(A_sp, k, max_it, seed, algorithm, verbose);
    }

    py::dict out;
    out["u"] = arma_mat_to_numpy(res(0));
    out["d"] = arma_mat_to_numpy(res(1));
    out["v"] = arma_mat_to_numpy(res(2));
    return out;
}

py::dict run_svd_dense(py::object A, int k = 30, int max_it = 0, int seed = 0,
                       int algorithm = 0, bool verbose = true) {
    arma::mat A_mat = numpy_to_arma_mat(A);
    arma::field<arma::mat> res;
    {
        py::gil_scoped_release release;
        res = actionet::runSVD(A_mat, k, max_it, seed, algorithm, verbose);
    }

    py::dict out;
    out["u"] = arma_mat_to_numpy(res(0));
    out["d"] = arma_mat_to_numpy(res(1));
    out["v"] = arma_mat_to_numpy(res(2));
    return out;
}

py::dict run_svd_operator(py::object op, int k = 30, int max_it = 0, int seed = 0,
                          int algorithm = actionet::ALG_HALKO, bool verbose = true) {
    PythonMatrixOperator mat_op(std::move(op));
    actionet::SVDResult res = actionet::runSVD_Operator(mat_op, k, max_it, seed, algorithm, verbose);

    py::dict out;
    out["u"] = arma_mat_to_numpy(res.U);
    out["d"] = arma_vec_to_numpy(res.sigma);
    out["v"] = arma_mat_to_numpy(res.V);
    return out;
}

// =====================================================================================================================

py::dict orthogonalize_batch_effect_operator(
    std::shared_ptr<actionet::MatrixOperator> op,
    py::array_t<double> old_S_r,
    py::array_t<double> old_U, py::array_t<double> old_A,
    py::array_t<double> old_B, py::array_t<double> old_sigma,
    py::array_t<double> design) {

    if (!op) {
        throw std::runtime_error("orthogonalize_batch_effect_operator: operator is null");
    }

    arma::mat S_r_mat = numpy_to_arma_mat(old_S_r);
    arma::mat U_mat = numpy_to_arma_mat(old_U);
    arma::mat A_mat = numpy_to_arma_mat(old_A);
    arma::mat B_mat = numpy_to_arma_mat(old_B);
    arma::vec sigma_vec = numpy_to_arma_vec(old_sigma);
    arma::mat design_mat = numpy_to_arma_mat(design);

    // Use the typed KernelReductionResult API (Plan 02A).
    actionet::KernelReductionResult reduction;
    reduction.S_r   = S_r_mat;
    reduction.sigma = sigma_vec;
    reduction.U     = U_mat;
    reduction.A     = A_mat;
    reduction.B     = B_mat;

    actionet::KernelReductionResult result = actionet::orthogonalizeBatchEffect_Operator(
        *op, reduction, design_mat);

    py::dict out;
    out["S_r"]   = arma_mat_to_numpy_c(result.S_r);
    out["sigma"] = arma_vec_to_numpy(result.sigma);
    out["U"]     = arma_mat_to_numpy_c(result.U);
    out["A"]     = arma_mat_to_numpy_c(result.A);
    out["B"]     = arma_mat_to_numpy_c(result.B);
    return out;
}

py::dict orthogonalize_basal_operator(
    std::shared_ptr<actionet::MatrixOperator> op,
    py::array_t<double> old_S_r,
    py::array_t<double> old_U, py::array_t<double> old_A,
    py::array_t<double> old_B, py::array_t<double> old_sigma,
    py::array_t<double> basal) {

    if (!op) {
        throw std::runtime_error("orthogonalize_basal_operator: operator is null");
    }

    arma::mat S_r_mat = numpy_to_arma_mat(old_S_r);
    arma::mat U_mat = numpy_to_arma_mat(old_U);
    arma::mat A_mat = numpy_to_arma_mat(old_A);
    arma::mat B_mat = numpy_to_arma_mat(old_B);
    arma::vec sigma_vec = numpy_to_arma_vec(old_sigma);
    arma::mat basal_mat = numpy_to_arma_mat(basal);

    // Use the typed KernelReductionResult API (Plan 02A).
    actionet::KernelReductionResult reduction;
    reduction.S_r   = S_r_mat;
    reduction.sigma = sigma_vec;
    reduction.U     = U_mat;
    reduction.A     = A_mat;
    reduction.B     = B_mat;

    actionet::KernelReductionResult result = actionet::orthogonalizeBasal_Operator(
        *op, reduction, basal_mat);

    py::dict out;
    out["S_r"]   = arma_mat_to_numpy_c(result.S_r);
    out["sigma"] = arma_vec_to_numpy(result.sigma);
    out["U"]     = arma_mat_to_numpy_c(result.U);
    out["A"]     = arma_mat_to_numpy_c(result.A);
    out["B"]     = arma_mat_to_numpy_c(result.B);
    return out;
}

// =====================================================================================================================

void init_decomposition(py::module_ &m) {
    // orthogonalization
    m.def("orthogonalize_batch_effect_sparse", &orthogonalize_batch_effect_sparse,
          "Orthogonalize batch effects (sparse)",
          py::arg("S"), py::arg("old_S_r"), py::arg("old_U"), py::arg("old_A"),
          py::arg("old_B"), py::arg("old_sigma"), py::arg("design"));

    m.def("orthogonalize_batch_effect_dense", &orthogonalize_batch_effect_dense,
          "Orthogonalize batch effects (dense)",
          py::arg("S"), py::arg("old_S_r"), py::arg("old_U"), py::arg("old_A"),
          py::arg("old_B"), py::arg("old_sigma"), py::arg("design"));

    m.def("orthogonalize_basal_sparse", &orthogonalize_basal_sparse,
          "Orthogonalize basal expression (sparse)",
          py::arg("S"), py::arg("old_S_r"), py::arg("old_U"), py::arg("old_A"),
          py::arg("old_B"), py::arg("old_sigma"), py::arg("basal"));

    m.def("orthogonalize_basal_dense", &orthogonalize_basal_dense,
          "Orthogonalize basal expression (dense)",
          py::arg("S"), py::arg("old_S_r"), py::arg("old_U"), py::arg("old_A"),
          py::arg("old_B"), py::arg("old_sigma"), py::arg("basal"));

    // svd_main
    m.def("perturbed_svd", &perturbed_svd, "Compute perturbed SVD",
          py::arg("u"), py::arg("d"), py::arg("v"), py::arg("A"), py::arg("B"));

    m.def("perturbed_svd_with_prior", &perturbed_svd_with_prior,
          "Compute perturbed SVD with prior perturbation terms",
          py::arg("u"), py::arg("d"), py::arg("v"),
          py::arg("old_A"), py::arg("old_B"),
          py::arg("A_new"), py::arg("B_new"));

    m.def("run_svd_sparse", &run_svd_sparse, "Run SVD (sparse)",
          py::arg("A"), py::arg("k") = 30, py::arg("max_it") = 0, py::arg("seed") = 0,
          py::arg("algorithm") = 0, py::arg("verbose") = true);

    m.def("run_svd_dense", &run_svd_dense, "Run SVD (dense)",
          py::arg("A"), py::arg("k") = 30, py::arg("max_it") = 0, py::arg("seed") = 0,
          py::arg("algorithm") = 0, py::arg("verbose") = true);

    m.def("run_svd_operator", &run_svd_operator, "Run SVD (generic operator)",
          py::arg("op"), py::arg("k") = 30, py::arg("max_it") = 0, py::arg("seed") = 0,
          py::arg("algorithm") = actionet::ALG_HALKO, py::arg("verbose") = true);

    m.def("orthogonalize_batch_effect_operator", &orthogonalize_batch_effect_operator,
          "Orthogonalize batch effects (operator-backed)",
          py::arg("op"), py::arg("old_S_r"), py::arg("old_U"), py::arg("old_A"),
          py::arg("old_B"), py::arg("old_sigma"), py::arg("design"));

    m.def("orthogonalize_basal_operator", &orthogonalize_basal_operator,
          "Orthogonalize basal expression (operator-backed)",
          py::arg("op"), py::arg("old_S_r"), py::arg("old_U"), py::arg("old_A"),
          py::arg("old_B"), py::arg("old_sigma"), py::arg("basal"));
}
