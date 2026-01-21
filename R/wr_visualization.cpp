// Rcpp interface for `visualization` module
// Organized by module header in th order imported.
#include "actionet_r_config.h"

// generate_layout =====================================================================================================

// [[Rcpp::export]]
arma::mat C_layoutNetwork(arma::sp_mat& G, arma::mat& initial_coordinates, std::string method = "umap",
                          unsigned int n_components = 2, float spread = 1, float min_dist = 1,
                          unsigned int n_epochs = 0,
                          float learning_rate = 1, float repulsion_strength = 1, float negative_sample_rate = 5,
                          bool approx_pow = false, bool pcg_rand = true, bool batch = true, unsigned int grain_size = 1,
                          int seed = 0, int thread_no = 0, bool verbose = true, float a = 0, float b = 0,
                          std::string opt_method = "adam", float alpha = -1, float beta1 = 0.5,
                          float beta2 = 0.9, float eps = 1e-7) {
    arma::mat coordinates = actionet::layoutNetwork(G, initial_coordinates, std::move(method), n_components, spread,
                                                    min_dist,
                                                    n_epochs, learning_rate, repulsion_strength, negative_sample_rate,
                                                    approx_pow, pcg_rand, batch, grain_size, seed, thread_no, verbose,
                                                    a, b, std::move(opt_method), alpha, beta1, beta2, eps);

    return coordinates;
}

// [[Rcpp::export]]
arma::mat C_computeNodeColors(const arma::mat& coordinates, int thread_no = 1) {
    arma::mat colors = actionet::computeNodeColors(coordinates, thread_no);
    return (colors);
}
