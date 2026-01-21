// Rcpp interface for `annotation` module
// Organized by module header in th order imported.
#include "actionet_r_config.h"

// marker_stats ========================================================================================================

// [[Rcpp::export]]
arma::mat C_computeFeatureStats(arma::sp_mat& G, arma::sp_mat& S, arma::sp_mat& X, int norm_method = 2,
                                double alpha = 0.85, int max_it = 5, bool approx = false, int thread_no = 0,
                                bool ignore_baseline = false) {
    arma::mat stats = actionet::computeFeatureStats(G, S, X, norm_method, alpha, max_it,
                                                    approx, thread_no, ignore_baseline);

    return (stats);
}

// [[Rcpp::export]]
arma::mat C_computeFeatureStatsVision(arma::sp_mat& G, arma::sp_mat& S, arma::sp_mat& X, int norm_method = 2,
                                      double alpha = 0.85, int max_it = 5, bool approx = false, int thread_no = 0) {
    arma::mat stats = actionet::computeFeatureStatsVision(G, S, X, norm_method,
                                                          alpha, max_it, approx, thread_no);

    return (stats);
}

// specificity =========================================================================================================

//' Compute feature specificity (from archetype footprints)
//'
//' @param S Input matrix (sparseMatrix)
//' @param H A soft membership matrix - Typically H_merged from the mergeArchetypes() function.
//'
//' @return A list with the over/under-logPvals
//'
//' @examples
//' prune.out = collectArchetypes(ACTION.out$C, ACTION.out$H)
//'	G = buildNetwork(prune.out$H_stacked)
//' unification.out = mergeArchetypes(G, S_r, prune.out$C_stacked, prune.out$H_stacked)
//' cell.clusters = unification.out$sample_assignments
//' S.norm = renormalize_input_matrix(S, cell.clusters)
//' logPvals.list = compute_archetype_feature_specificity(S.norm, unification.out$H_merged)
//' specificity.scores = logPvals.list$upper_significance
// [[Rcpp::export]]
Rcpp::List C_archetypeFeatureSpecificitySparse(arma::sp_mat& S, arma::mat& H, int thread_no = 0) {
    arma::field<arma::mat> res = actionet::computeFeatureSpecificity(S, H, thread_no);

    Rcpp::List out_list;
    out_list["archetypes"] = res(0);
    out_list["upper_significance"] = res(1);
    out_list["lower_significance"] = res(2);

    return (out_list);
}

// [[Rcpp::export]]
Rcpp::List C_archetypeFeatureSpecificityDense(arma::mat& S, arma::mat& H, int thread_no = 0) {
    arma::field<arma::mat> res = actionet::computeFeatureSpecificity(S, H, thread_no);

    Rcpp::List out_list;
    out_list["archetypes"] = res(0);
    out_list["upper_significance"] = res(1);
    out_list["lower_significance"] = res(2);

    return (out_list);
}

//' Compute feature specificity (from cluster assignments)
//'
//' @param S Input matrix ("sparseMatrix")
//' @param sample_assignments Vector of cluster assignments
//'
//' @return A list with the over/under-logPvals
//'
//' @examples
//' prune.out = collectArchetypes(ACTION.out$C, ACTION.out$H)
//'	G = buildNetwork(prune.out$H_stacked)
//' unification.out = mergeArchetypes(G, S_r, prune.out$C_stacked, prune.out$H_stacked)
//' cell.clusters = unification.out$sample_assignments
//' S.norm = renormalize_input_matrix(S, cell.clusters)
//' logPvals.list = compute_cluster_feature_specificity(S.norm, cell.clusters)
//' specificity.scores = logPvals.list$upper_significance
// [[Rcpp::export]]
Rcpp::List C_computeFeatureSpecificitySparse(arma::sp_mat& S, arma::uvec& labels, int thread_no = 0) {
    arma::field<arma::mat> res = actionet::computeFeatureSpecificity(S, labels, thread_no);

    Rcpp::List out_list;
    out_list["average_profile"] = res(0);
    out_list["upper_significance"] = res(1);
    out_list["lower_significance"] = res(2);

    return (out_list);
}

// [[Rcpp::export]]
Rcpp::List C_computeFeatureSpecificityDense(arma::mat& S, arma::uvec& labels, int thread_no = 0) {
    arma::field<arma::mat> res = actionet::computeFeatureSpecificity(S, labels, thread_no);

    Rcpp::List out_list;
    out_list["average_profile"] = res(0);
    out_list["upper_significance"] = res(1);
    out_list["lower_significance"] = res(2);

    return (out_list);
}
