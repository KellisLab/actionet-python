// Rcpp interface for `action` module
// Organized by module header in th order imported.
#include "actionet_r_config.h"

// aa ==================================================================================================================

//' Runs archetypal analysis (AA)
//'
//' @param A Input matrix.
//' @param W0 Matrix with <em>k</em> columns representing initial archetypes.
//' @param max_it Maximum number of iterations.
//' @param tol Convergence tolerance.
//'
//' @return Field with matrices <b>C</b> (<b>A.n_cols</b> x <em>k</em>) and <b>H</b> (<em>k</em> x <b>A.n_cols</b>).
//'
//' @examples
//' S_r = t(reducedDims(ace)$ACTION)
//' SPA.out = runSPA(S_r, 10)
//' W0 = S_r[, SPA.selected_cols]
//' AA.out = runAA(S_r, W0)
//' H = AA.out$H
//' cell.assignments = apply(H, 2, which.max)
// [[Rcpp::export]]
Rcpp::List C_runAA(arma::mat& A, arma::mat& W0, int max_it = 100, double tol = 1e-6) {
    arma::field<arma::mat> res = actionet::runAA(A, W0, max_it, tol);

    Rcpp::List out;
    out["C"] = res(0);
    out["H"] = res(1);
    out["W"] = A * res(0);

    return out;
}

// action_decomp =======================================================================================================

//' Run ACTION decomposition algorithm
//'
//' @param S_r Input matrix. Usually a reduced representation of the raw data.
//' @param k_min Minimum number of archetypes (>= 2) to search for, and the beginning of the search range.
//' @param k_max Maximum number of archetypes (<= <b>S_r.n_cols</b>) to search for, and the end of the search range.
//' @param normalization Normalization method to apply on <b>S_r</b> before running ACTION.
//' @param max_it Maximum number of iterations for <code>runAA()</code>.
//' @param tol Convergence tolerance for <code>runAA()</code>.
//' @param thread_no Number of CPU threads to use. If 0, number is automatically determined.
//'
//' @return A named list with entries 'C' and 'H', each a list for different values of k
//'
//' @examples
//' ACTION.out = runACTION(S_r, k_max = 10)
//' H8 = ACTION.out$H[[8]]
//' cell.assignments = apply(H8, 2, which.max)
// [[Rcpp::export]]
Rcpp::List C_decompACTION(arma::mat& S_r, int k_min = 2, int k_max = 30, int max_it = 100, double tol = 1e-16,
                        int thread_no = 0) {
    actionet::ResACTION trace = actionet::decompACTION(S_r, k_min, k_max, max_it, tol, thread_no);

    Rcpp::List res;

    Rcpp::List C(k_max);
    for (int i = k_min; i <= k_max; i++) {
        arma::mat cur_C = trace.C[i];
        C[i - 1] = cur_C;
    }
    res["C"] = C;

    Rcpp::List H(k_max);
    for (int i = k_min; i <= k_max; i++) {
        arma::mat cur_H = trace.H[i];
        H[i - 1] = cur_H;
    }
    res["H"] = H;

    return res;
}

// [[Rcpp::export]]
Rcpp::List C_runACTION(arma::mat& S_r, int k_min = 2, int k_max = 30, int max_it = 100, double tol = 1e-16,
                     double spec_th = -3, int min_obs = 3, int thread_no = 0) {
    arma::field<arma::mat> action_out =
        actionet::runACTION(S_r, k_min, k_max, max_it, tol, spec_th, min_obs, thread_no);

    Rcpp::List out;
    out["H_stacked"] = action_out(0);
    out["C_stacked"] = action_out(1);
    out["H_merged"] = action_out(2);
    out["C_merged"] = action_out(3);
    out["assigned_archetypes"] = arma::vec(action_out(4)) + 1; // Shift index

    return out;
}

// action_post =========================================================================================================

//' Filter and aggregate multi-level archetypes
//'
//' @param C_trace Field containing C matrices. Output of <code>runACTION()</code> in <code>ResACTION["C"]</code>.
//' @param H_trace Field containing H matrices. Output of <code>runACTION()</code> in <code>ResACTION["H"]</code>.
//' @param spec_th Minimum threshold (as z-score) to filter archetypes by specificity.
//' @param min_obs Minimum number of observations assigned to an archetypes needed to retain that archetype.
//'
//' @return A named list: \itemize{
//' \item selected_archs: List of final archetypes that passed the
//' filtering/pruning step.
//' \item C_stacked,H_stacked: Horizontal/Vertical
//' concatenation of filtered C and H matrices, respectively.
//' }
//'
//' @examples
//' S = logcounts(sce)
//' reduction.out = reduce(S, reduced_dim = 50)
//' S_r = reduction.out$S_r
//' ACTION.out = runACTION(S_r, k_max = 10)
//' reconstruction.out = reconstruct_archetypes(S, ACTION.out$C, ACTION.out$H)
// [[Rcpp::export]]
Rcpp::List C_collectArchetypes(const Rcpp::List& C_trace, const Rcpp::List& H_trace,
                             double spec_th = -3, int min_obs = 3) {
    int n_list = H_trace.size();
    arma::field<arma::mat> C_trace_vec(n_list + 1);
    arma::field<arma::mat> H_trace_vec(n_list + 1);
    for (int i = 0; i < n_list; i++) {
        if (Rf_isNull(H_trace[i])) {
            continue;
        }
        C_trace_vec[i + 1] = (Rcpp::as<arma::mat>(C_trace[i]));
        H_trace_vec[i + 1] = (Rcpp::as<arma::mat>(H_trace[i]));
    }

    actionet::ResCollectArch results =
        actionet::collectArchetypes(C_trace_vec, H_trace_vec, spec_th, min_obs);

    Rcpp::List out_list;
    out_list["selected_archs"] = results.selected_archs + 1;
    out_list["C_stacked"] = results.C_stacked;
    out_list["H_stacked"] = results.H_stacked;

    return out_list;
}

//' Identify and merge redundant archetypes into a representative subset
//'
//' @param S_r Reduced data matrix from which archetypes were found.
//' @param C_stacked Concatenated (and filtered) <code>C</code> (<b>S_r.n</b> x <em>n</em>) matrix.
//' Output of <code>collectArchetypes()</code> in <code>ResCollectArch["C_stacked"]</code>.
//' @param H_stacked Concatenated (and filtered) <code>H</code> (<b>S_r.n</b> x <em>n</em>) matrix.
//' Output of <code>collectArchetypes()</code> in <code>ResCollectArch["H_stacked"]</code>.
//' @param normalization Normalization method to apply to <b>S_r</b>.
//' @param thread_no Number of CPU threads to use. If 0, number is automatically determined.
//'
//' @return A named list: \itemize{
//' \item archetype_groups: Equivalent classes of archetypes (non-redundant)
//' \item C_merged,H_merged: C and H matrices of merged archetypes
//' \item sample_assignments: Assignment of samples/cells to merged archetypes
//' }
//' @examples
//' prune.out = collectArchetypes(ACTION.out$C, ACTION.out$H)
//'	G = buildNetwork(prune.out$H_stacked)
//' unification.out = mergeArchetypes(G, S_r, prune.out$C_stacked, prune.out$H_stacked)
//' cell.clusters = unification.out$sample_assignments
// [[Rcpp::export]]
Rcpp::List
    C_mergeArchetypes(arma::mat& S_r, arma::mat& C_stacked, arma::mat& H_stacked, int thread_no = 0) {
    actionet::ResMergeArch results =
        actionet::mergeArchetypes(S_r, C_stacked, H_stacked, thread_no);

    Rcpp::List out_list;
    out_list["selected_archetypes"] = results.selected_archetypes + 1;
    out_list["C_merged"] = results.C_merged;
    out_list["H_merged"] = results.H_merged;
    out_list["assigned_archetypes"] = results.assigned_archetypes + 1;

    return out_list;
}

// reduce_kernel =======================================================================================================

//' Compute reduced kernel matrix
//'
//' @param S Input matrix (<em>vars</em> x <em>obs</em>).
//' May be <code>arma::mat</code> or <code>arma::sp_mat</code>.
//' @param dim Number of singular vectors to estimate. Passed to <code>runSVD()</code>.
//' @param svd_alg Singular value decomposition algorithm. See to <code>runSVD()</code> for options.
//' @param max_it Maximum number of iterations. Passed to <code>runSVD()</code>.
//' @param seed Random seed.
//' @param verbose Print status messages.
//'
//' @return Field with 5 elements:
//' - 0: <code>arma::mat</code> Reduced kernel matrix.
//' - 1: <code>arma::vec</code> Singular values.
//' - 2: <code>arma::mat</code> Left singular vectors.
//' - 3: <code>arma::mat</code> <b>A</b> perturbation matrix.
//' - 4: <code>arma::mat</code> <b>B</b> perturbation matrix.
//'
//' @examples
//' S = logcounts(sce)
//' reduction.out = reduce(S, reduced_dim = 50)
//' S_r = reduction.out$S_r
// [[Rcpp::export]]
Rcpp::List C_reduceKernelSparse(arma::sp_mat& S, int k = 50, int svd_alg = 0, int max_it = 0, int seed = 0,
                                bool verbose = true) {
    arma::field<arma::mat> reduction =
        actionet::reduceKernel(S, k, svd_alg, max_it, seed, verbose);

    Rcpp::List res;
    res["S_r"] = reduction(0);
    res["sigma"] = reduction(1);
    res["V"] = reduction(2);
    res["A"] = reduction(3);
    res["B"] = reduction(4);

    return res;
}

// [[Rcpp::export]]
Rcpp::List C_reduceKernelDense(arma::mat& S, int k = 50, int svd_alg = 0, int max_it = 0, int seed = 0,
                               bool verbose = true) {
    arma::field<arma::mat> reduction =
        actionet::reduceKernel(S, k, svd_alg, max_it, seed, verbose);

    Rcpp::List res;
    res["S_r"] = reduction(0);
    res["sigma"] = reduction(1);
    res["V"] = reduction(2);
    res["A"] = reduction(3);
    res["B"] = reduction(4);

    return res;
}

// simplex_regression ==================================================================================================

//' Solves min_{X} (|| AX - B ||) s.t. simplex constraint
//'
//' @param A Input matrix <em>A</em> in <em>AX - B</em>.
//' @param B Inout matrix <em>B</em> in <em>AX - B</em>.
//' @param computeXtX Return <em>Xt(X)</em>
//'
//' @return X Solution
//'
//' @examples
//' C = ACTION.out$C[[10]]
//' A = S_r %*% C
//' B = S_r
//' H = runSimplexRegression(A, B)
// [[Rcpp::export]]
arma::mat C_runSimplexRegression(arma::mat& A, arma::mat& B, bool computeXtX = false) {
    arma::mat X = actionet::runSimplexRegression(A, B, computeXtX);

    return X;
}

// spa =================================================================================================================

//' Run successive projections algorithm (SPA) to solve separable NMF
//'
//' @param A Input matrix.
//' @param k Number of candidate vertices to solve for.
//'
//' @return A named list with entries 'selected_cols' and 'norms'
//' @examples
//' H = runSPA(S_r, 10)
// [[Rcpp::export]]
Rcpp::List C_runSPA(arma::mat& A, int k) {
    actionet::ResSPA res = actionet::runSPA(A, k);

    // Shift index
    arma::uvec selected_cols = res.selected_cols;
    arma::vec cols(k);
    for (int i = 0; i < k; i++) {
        cols[i] = selected_cols[i] + 1;
    }

    Rcpp::List out;
    out["selected_cols"] = cols;
    out["norms"] = res.column_norms;

    return out;
}
