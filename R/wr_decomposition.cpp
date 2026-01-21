// Rcpp interface for `decomposition` module
// Organized by module header in th order imported.
#include "actionet_r_config.h"

// orthogonalization ========================================================================================================
// TODO: This whole submodule is fucked. Fix it.
// [[Rcpp::export]]
Rcpp::List C_orthogonalizeBatchEffect(arma::sp_mat& S, arma::mat& old_S_r, arma::mat& old_V, arma::mat& old_A,
                                      arma::mat& old_B, arma::vec& old_sigma, arma::mat& design) {
    arma::field<arma::mat> SVD_results(5);

    SVD_results(0) = old_V;
    SVD_results(1) = old_sigma;
    SVD_results(2) = old_S_r;
    for (int i = 0; i < old_sigma.n_elem; i++) {
        SVD_results(2).col(i) /= old_sigma(i);
    }
    SVD_results(3) = old_A;
    SVD_results(4) = old_B;

    arma::field<arma::mat> orthogonalized_reduction =
        actionet::orthogonalizeBatchEffect(S, SVD_results, design);

    Rcpp::List res;
    res["V"] = orthogonalized_reduction(0);

    arma::vec sigma = orthogonalized_reduction(1).col(0);
    res["sigma"] = sigma;

    arma::mat V = orthogonalized_reduction(2);
    for (int i = 0; i < V.n_cols; i++) {
        V.col(i) *= sigma(i);
    }
    res["S_r"] = arma::trans(V);

    res["A"] = orthogonalized_reduction(3);
    res["B"] = orthogonalized_reduction(4);

    return res;
}

//[[Rcpp::export]]
Rcpp::List C_orthogonalizeBatchEffect_full(arma::mat& S, arma::mat& old_S_r, arma::mat& old_V, arma::mat& old_A,
                                           arma::mat& old_B, arma::vec& old_sigma, arma::mat& design) {
    arma::field<arma::mat> SVD_results(5);

    SVD_results(0) = old_V;
    SVD_results(1) = old_sigma;
    SVD_results(2) = old_S_r;
    for (int i = 0; i < old_sigma.n_elem; i++) {
        SVD_results(2).col(i) /= old_sigma(i);
    }
    SVD_results(3) = old_A;
    SVD_results(4) = old_B;

    arma::field<arma::mat> orthogonalized_reduction =
        actionet::orthogonalizeBatchEffect(S, SVD_results, design);

    Rcpp::List res;
    res["V"] = orthogonalized_reduction(0);

    arma::vec sigma = orthogonalized_reduction(1).col(0);
    res["sigma"] = sigma;

    arma::mat V = orthogonalized_reduction(2);
    for (int i = 0; i < V.n_cols; i++) {
        V.col(i) *= sigma(i);
    }
    res["S_r"] = arma::trans(V);
    res["A"] = orthogonalized_reduction(3);
    res["B"] = orthogonalized_reduction(4);

    return res;
}

// [[Rcpp::export]]
Rcpp::List C_orthogonalizeBasal(arma::sp_mat& S, arma::mat& old_S_r, arma::mat& old_V, arma::mat& old_A,
                                arma::mat& old_B, arma::vec& old_sigma, arma::mat& basal) {
    arma::field<arma::mat> SVD_results(5);

    SVD_results(0) = old_V;
    SVD_results(1) = old_sigma;
    SVD_results(2) = old_S_r;
    for (int i = 0; i < old_sigma.n_elem; i++) {
        SVD_results(2).col(i) /= old_sigma(i);
    }
    SVD_results(3) = old_A;
    SVD_results(4) = old_B;

    arma::field<arma::mat> orthogonalized_reduction = actionet::orthogonalizeBasal(S, SVD_results, basal);

    Rcpp::List res;
    res["V"] = orthogonalized_reduction(0);

    arma::vec sigma = orthogonalized_reduction(1).col(0);
    res["sigma"] = sigma;

    arma::mat V = orthogonalized_reduction(2);
    for (int i = 0; i < V.n_cols; i++) {
        V.col(i) *= sigma(i);
    }
    res["S_r"] = arma::trans(V);

    res["A"] = orthogonalized_reduction(3);
    res["B"] = orthogonalized_reduction(4);

    return res;
}

//[[Rcpp::export]]
Rcpp::List C_orthogonalizeBasal_full(arma::mat& S, arma::mat& old_S_r, arma::mat& old_V, arma::mat& old_A,
                                     arma::mat& old_B, arma::vec& old_sigma, arma::mat& basal) {
    arma::field<arma::mat> SVD_results(5);

    SVD_results(0) = old_V;
    SVD_results(1) = old_sigma;
    SVD_results(2) = old_S_r;
    for (int i = 0; i < old_sigma.n_elem; i++) {
        SVD_results(2).col(i) /= old_sigma(i);
    }
    SVD_results(3) = old_A;
    SVD_results(4) = old_B;

    arma::field<arma::mat> orthogonalized_reduction = actionet::orthogonalizeBasal(S, SVD_results, basal);

    Rcpp::List res;
    res["V"] = orthogonalized_reduction(0);

    arma::vec sigma = orthogonalized_reduction(1).col(0);
    res["sigma"] = sigma;

    arma::mat V = orthogonalized_reduction(2);
    for (int i = 0; i < V.n_cols; i++) {
        V.col(i) *= sigma(i);
    }
    res["S_r"] = arma::trans(V);
    res["A"] = orthogonalized_reduction(3);
    res["B"] = orthogonalized_reduction(4);

    return res;
}

// svd_main =========================================================================================================

//' Computes SVD decomposition
//'
//' This is direct implementation of the randomized SVD algorithm:
//' From: IRLBA R Package
//'
//' @param A Input matrix ("sparseMatrix")
//' @param k Dimension of SVD decomposition
//' @param max_it Number of iterations (default=5)
//' @param seed Random seed (default=0)
//' @param algorithm SVD algorithm to use. Currently supported methods are blah blah blah
//'
//' @return A named list with U, sigma, and V components
//'
//' @examples
//' A = randn(100, 20)
//' svd.out = runSVD(A, dim = 3)
//' U = svd.out$u
// [[Rcpp::export]]
Rcpp::List C_runSVDSparse(arma::sp_mat& A, int k = 30, int max_it = 0, int seed = 0, int algorithm = 0,
                          bool verbose = true) {
    arma::field<arma::mat> SVD_out = actionet::runSVD(A, k, max_it, seed, algorithm, verbose);

    Rcpp::List res;
    res["u"] = SVD_out(0);
    res["d"] = SVD_out(1);
    res["v"] = SVD_out(2);

    return res;
}

// [[Rcpp::export]]
Rcpp::List C_runSVDDense(arma::mat& A, int k = 30, int max_it = 0, int seed = 0, int algorithm = 0,
                         bool verbose = true) {
    arma::field<arma::mat> SVD_out = actionet::runSVD(A, k, max_it, seed, algorithm, verbose);

    Rcpp::List res;
    res["u"] = SVD_out(0);
    res["d"] = SVD_out(1);
    res["v"] = SVD_out(2);

    return res;
}

// [[Rcpp::export]]
Rcpp::List C_perturbedSVD(arma::mat u, arma::vec d, arma::mat v, arma::mat A, arma::mat B) {
    // TODO: Jank. Put this in the function
    if (1 < d.n_cols)
        d = d.diag();

    arma::field<arma::mat> SVD_results(3);
    SVD_results(0) = u;
    SVD_results(1) = d;
    SVD_results(2) = v;

    arma::field<arma::mat> perturbed_SVD = actionet::perturbedSVD(SVD_results, A, B);

    Rcpp::List res;
    res["u"] = perturbed_SVD(0);
    res["d"] = perturbed_SVD(1).col(0);
    res["v"] = perturbed_SVD(2);

    return res;
}
