// Rcpp interface for `network` module
// Organized by module header in th order imported.
#include "actionet_r_config.h"

// build_network =======================================================================================================

//' Builds an interaction network from the multi-level archetypal decompositions
//'
//' @param H_stacked Output of the collectArchetypes() function.
//' @param density Overall density of constructed graph. The higher the density,
//' the more edges are retained (default = 1.0).
//' @param thread_no Number of parallel threads (default = 0).
//' @param mutual_edges_only Symmetrization strategy for nearest-neighbor edges.
//' If it is true, only mutual nearest-neighbors are returned (default=TRUE).
//'
//' @return G Adjacency matrix of the ACTIONet graph.
//'
//' @examples
//' prune.out = collectArchetypes(ACTION.out$C, ACTION.out$H)
//'	G = buildNetwork(prune.out$H_stacked)
// [[Rcpp::export]]
arma::sp_mat C_buildNetwork(const arma::mat& H, std::string algorithm = "k*nn", std::string distance_metric = "jsd",
                            double density = 1.0, int thread_no = 0, double M = 16, double ef_construction = 200,
                            double ef = 50, bool mutual_edges_only = true, int k = 10) {
    arma::sp_mat G = actionet::buildNetwork(H, std::move(algorithm), std::move(distance_metric), density, thread_no, M,
                                            ef_construction, ef, mutual_edges_only, k);

    return G;
}

// label_propagation ===================================================================================================

// [[Rcpp::export]]
arma::vec C_runLPA(arma::sp_mat& G, arma::vec& labels, double lambda = 1, int iters = 3,
                   double sig_threshold = 3, Rcpp::Nullable<Rcpp::IntegerVector> fixed_labels_ = R_NilValue,
                   int thread_no = 0) {
    // TODO: This is ugly. Find a better way to fix labels.
    arma::uvec fixed_labels_vec;
    if (fixed_labels_.isNotNull()) {
        Rcpp::NumericVector fixed_labels(fixed_labels_);
        fixed_labels_vec.set_size(fixed_labels.size());
        for (int i = 0; i < fixed_labels.size(); i++) {
            fixed_labels_vec(i) = fixed_labels(i) - 1;
        }
    }

    arma::vec new_labels =
        actionet::runLPA(G, labels, lambda, iters, sig_threshold, fixed_labels_vec, thread_no);

    return (new_labels);
}

// network_diffusion ===================================================================================================

//' Computes network diffusion over a given network, starting with an arbitrarty
//' set of initial scores
//'
//' @param G Input graph
//' @param X0 Matrix of initial values per diffusion (ncol(G) == nrow(G) == ncol(X0))
//' @param thread_no Number of parallel threads (default=0)
//' @param alpha Random-walk depth ( between [0, 1] ) ' @param max_it PageRank iterations
//'
//' @return Matrix of diffusion scores
//'
//' @examples
//' G = colNets(ace)$ACTIONet
//' gene.expression = Matrix::t(logcounts(ace))[c("CD19", "CD14", "CD16"), ]
//' smoothed.expression = computeNetworkDiffusionApprox(G, gene.expression)
// [[Rcpp::export]]
arma::mat C_computeNetworkDiffusion(arma::sp_mat& G, arma::mat& X0, double alpha = 0.85, int max_it = 5,
                                    int thread_no = 0, bool approx = false, int norm_method = 0, double tol = 1e-8) {
    arma::mat X = actionet::computeNetworkDiffusion(G, X0, alpha, max_it, thread_no, approx, norm_method, tol);
    return (X);
}

// network_measures ====================================================================================================

//' Compute coreness of graph vertices
//'
//' @param G Input graph
//'
//' @return cn core-number of each graph node
//'
//' @examples
//' G = colNets(ace)$ACTIONet
//' cn = computeCoreness(G)
// [[Rcpp::export]]
arma::uvec C_computeCoreness(arma::sp_mat& G) {
    arma::uvec core_num = actionet::computeCoreness(G);
    return (core_num);
}

//' Compute coreness of subgraph vertices induced by each archetype
//'
//' @param G Input graph
//' @param sample_assignments Archetype discretization (output of mergeArchetypes())
//'
//' @return cn core-number of each graph node
//'
//' @examples
//' G = colNets(ace)$ACTIONet
//' assignments = ace$archetype.assignment
//' connectivity = computeCoreness(G, assignments)
// [[Rcpp::export]]
arma::vec C_computeArchetypeCentrality(arma::sp_mat& G, const arma::uvec& sample_assignments) {
    arma::vec conn = actionet::computeArchetypeCentrality(G, sample_assignments);

    return (conn);
}
