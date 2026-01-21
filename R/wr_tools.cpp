// Rcpp interface for `tools` module
// Organized by module header in th order imported.
#include "actionet_r_config.h"

// autocorrelation =====================================================================================================

// TODO: Unused. Remove?
// [[Rcpp::export]]
Rcpp::List
    C_autocorrelation_Moran_parametric(const arma::sp_mat& G, const arma::mat& scores, int normalization_method = 4,
                                       int thread_no = 0) {
    arma::field<arma::vec> out = actionet::autocorrelation_Moran_parametric(G, scores, normalization_method, thread_no);

    Rcpp::List res;
    res["stat"] = out[0];
    res["zscore"] = out[1];
    res["mu"] = out[2];
    res["sigma"] = out[3];

    return (res);
}

// [[Rcpp::export]]
Rcpp::List C_autocorrelation_Moran(const arma::sp_mat& G, const arma::mat& scores, int normalization_method = 1,
                                   int perm_no = 30,
                                   int thread_no = 0) {
    arma::field<arma::vec> out = actionet::autocorrelation_Moran(G, scores, normalization_method, perm_no, thread_no);

    Rcpp::List res;
    res["Moran_I"] = out[0];
    res["zscore"] = out[1];
    res["mu"] = out[2];
    res["sigma"] = out[3];

    return (res);
}

// [[Rcpp::export]]
Rcpp::List C_autocorrelation_Geary(const arma::sp_mat& G, const arma::mat& scores, int normalization_method = 1,
                                   int perm_no = 30,
                                   int thread_no = 0) {
    arma::field<arma::vec> out = actionet::autocorrelation_Geary(G, scores, normalization_method, perm_no, thread_no);

    Rcpp::List res;
    res["Geary_C"] = out[0];
    res["zscore"] = out[1];
    res["mu"] = out[2];
    res["sigma"] = out[3];

    return (res);
}

// enrichment ==========================================================================================================

// [[Rcpp::export]]
arma::mat C_computeGraphLabelEnrichment(const arma::sp_mat& G, arma::mat& scores, int thread_no = 0) {
    arma::mat logPvals = actionet::computeGraphLabelEnrichment(G, scores, thread_no);

    return (logPvals);
}

//' Computes feature enrichment wrt a given annotation
//'
//' @param scores Specificity scores of features
//' @param associations Binary matrix of annotations
//' @param L Length of the top-ranked scores to scan
//'
//' @return Matrix of log-pvalues
//'
//' @examples
//' data("gProfilerDB_human")
//' G = colNets(ace)$ACTIONet
//' associations = gProfilerDB_human$SYMBOL$REAC
//' common.genes = intersect(rownames(ace), rownames(associations))
//' specificity_scores = rowFactors(ace)[["H_merged_upper_significance"]]
//' logPvals = computeFeatureSpecificity(
//' specificity_scores[common.genes, ], annotations[common.genes, ]
//' )
//' rownames(logPvals) = colnames(specificity_scores)
//' colnames(logPvals) = colnames(annotations)
// [[Rcpp::export]]
Rcpp::List C_assess_enrichment(arma::mat& scores, arma::sp_mat& associations, int thread_no = 0) {
    arma::field<arma::mat> res = actionet::assess_enrichment(scores, associations, thread_no);

    Rcpp::List out_list;
    out_list["logPvals"] = res(0);
    out_list["thresholds"] = res(1);

    return (out_list);
}

// matrix_aggregate =========================================================================================================

//' Aggregate matrix within groups (rowwise or columnwise)
//'
//' @param S matrix of type "dMatrix"
//' @param sample_assignments Vector of groupings. Group labels must be continuous integers or coercible to such.
//' @param axis 0 for rowwise (default), 1 for columnwise
//' @return Aggregated matrix
//'
// [[Rcpp::export]]
arma::mat C_computeGroupedSumsSparse(arma::sp_mat& S, arma::vec& sample_assignments, int axis = 0) {
    arma::mat pb = actionet::computeGroupedSums(S, sample_assignments, axis);
    return pb;
}

// [[Rcpp::export]]
arma::sp_mat C_computeGroupedSumsSparse2(arma::sp_mat& S, arma::vec& sample_assignments, int axis = 0) {
    arma::sp_mat pb = actionet::computeGroupedSums2(S, sample_assignments, axis);
    return pb;
}

// [[Rcpp::export]]
arma::mat C_computeGroupedSumsDense(arma::mat& S, arma::vec& sample_assignments, int axis = 0) {
    arma::mat pb = actionet::computeGroupedSums(S, sample_assignments, axis);
    return pb;
}

//' Average matrix within groups (rowwise or columnwise)
//'
//' @param S matrix
//' @param sample_assignments Vector of groupings. Group labels must be continuous integers or coercible to such.
//' @param axis 0 for rowwise (default), 1 for columnwise
//' @return Averaged matrix
//'
// [[Rcpp::export]]
arma::mat C_computeGroupedMeansSparse(arma::sp_mat& S, arma::vec& sample_assignments, int axis = 0) {
    arma::mat pb = actionet::computeGroupedMeans(S, sample_assignments, axis);
    return pb;
}

// [[Rcpp::export]]
arma::sp_mat C_computeGroupedMeansSparse2(arma::sp_mat& S, arma::vec& sample_assignments, int axis = 0) {
    arma::sp_mat pb = actionet::computeGroupedMeans2(S, sample_assignments, axis);
    return pb;
}

// [[Rcpp::export]]
arma::mat C_computeGroupedMeansDense(arma::mat& S, arma::vec& sample_assignments, int axis = 0) {
    arma::mat pb = actionet::computeGroupedMeans(S, sample_assignments, axis);
    return pb;
}

//' Variance matrix within groups (rowwise or columnwise)
//'
//' @param S matrix
//' @param sample_assignments Vector of groupings. Group labels must be continuous integers or coercible to such.
//' @param axis 0 for rowwise (default), 1 for columnwise
//' @return Variance matrix
//'
// [[Rcpp::export]]
arma::mat C_computeGroupedVarsSparse(arma::sp_mat& S, arma::vec& sample_assignments, int axis = 0) {
    arma::mat pb = actionet::computeGroupedVars(S, sample_assignments, axis);
    return pb;
}

// [[Rcpp::export]]
arma::sp_mat C_computeGroupedVarsSparse2(arma::sp_mat& S, arma::vec& sample_assignments, int axis = 0) {
    arma::sp_mat pb = actionet::computeGroupedVars2(S, sample_assignments, axis);
    return pb;
}

// [[Rcpp::export]]
arma::mat C_computeGroupedVarsDense(arma::mat& S, arma::vec& sample_assignments, int axis = 0) {
    arma::mat pb = actionet::computeGroupedVars(S, sample_assignments, axis);
    return pb;
}

// matrix_transform =======================================================================================================

// [[Rcpp::export]]
arma::sp_mat C_normalizeMatrixSparse(arma::sp_mat& X, unsigned int p = 1, unsigned int dim = 0) {
    arma::sp_mat Xn = actionet::normalizeMatrix(X, p, dim);
    return (Xn);
}

// [[Rcpp::export]]
arma::mat C_normalizeMatrixDense(arma::mat& X, unsigned int p = 1, unsigned int dim = 0) {
    arma::mat Xn = actionet::normalizeMatrix(X, p, dim);

    return (Xn);
}

// [[Rcpp::export]]
arma::mat C_scaleMatrixDense(arma::mat& X, arma::vec& v, unsigned int dim = 0) {
    arma::mat Xs = actionet::scaleMatrix(X, v, dim);

    return (Xs);
}

// [[Rcpp::export]]
arma::sp_mat C_scaleMatrixSparse(arma::sp_mat& X, arma::vec& v, unsigned int dim = 0) {
    arma::sp_mat Xs = actionet::scaleMatrix(X, v, dim);
    return (Xs);
}

// [[Rcpp::export]]
arma::sp_mat C_normalizeGraph(arma::sp_mat& G, int norm_method = 0) {
    arma::sp_mat Gn = actionet::normalizeGraph(G, norm_method);
    return (Gn);
}

// mwm =================================================================================================================

//' Computes the maximum-weight bipartite graph matching
//'
//' @param G Adjacency matrix of the input graph
//'
//' @return G_matched An adjacency matrix with a maximum of one nonzero entry on
//' rows/columns
//'
//' @examples
//' G_matched = MWM_hungarian(G)
// [[Rcpp::export]]
arma::mat C_MWM_hungarian(arma::mat& G) {
    arma::mat G_matched = actionet::MWM_hungarian(G);

    return G_matched;
}

// [[Rcpp::export]]
arma::umat C_MWM_rank1(const arma::vec& u, const arma::vec& v, double u_threshold = 0, double v_threshold = 0) {
    arma::umat pairs = actionet::MWM_rank1(u, v, u_threshold, v_threshold);

    pairs = pairs + 1;

    return (pairs);
}

// xicor ===============================================================================================================

// [[Rcpp::export]]
arma::vec C_xicor(arma::vec xvec, arma::vec yvec, bool compute_pval = true, int seed = 0) {
    arma::vec res = actionet::xicor(std::move(xvec), std::move(yvec), compute_pval, seed);

    return (res);
}

// [[Rcpp::export]]
Rcpp::List C_XICOR(arma::mat& X, arma::mat& Y, bool compute_pval = true, int seed = 0, int thread_no = 0) {
    arma::field<arma::mat> out = actionet::XICOR(X, Y, compute_pval, seed, thread_no);

    Rcpp::List res;
    res["XI"] = out(0);
    res["Z"] = out(1);

    return (res);
}
