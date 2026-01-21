#' Gene expression imputation using network diffusion.
#'
#' @param ace ACTIONetExperiment object containing output of 'run.ACTIONet()'.
#' @param genes The list of genes to perform imputation on.
#' @param features_use A vector of features of length NROW(ace) or the name of a column of rowData(ace) containing the genes given in 'genes'.
#' @param alpha Depth of diffusion between (0, 1).
#' The larger it is, the deeper the diffusion, which results in less nonzeros (default = 0.85).
#' @param thread_no Number of parallel threads
#' @param diffusion_it Number of diffusion iterations (default = 5)
#' @param assay_name Slot in the ace object with normalized counts.
#'
#' @return Imputed gene expression matrix. Column names are set with imputed genes names and rows are cells.
#'
#' @examples
#' imputed.genes <- impute.genes.using.ACTIONet(ace, c("CD14", "CD19", "CD3G"))
#' plot.ACTIONet.gradient(ace, imputed.genes[, 1])

#' @export
imputeFeatures <- function(
    ace,
    features,
    algorithm = c("actionet", "pca"),
    features_use = NULL,
    alpha = 0.85,
    norm_method = "pagerank_sym",
    thread_no = 0,
    max_it = 5,
    assay_name = "logcounts",
    reduction_slot = "action",
    net_slot = "actionet") {
    algorithm <- match.arg(algorithm)

    features_use <- .get_features(ace, features_use = features_use, allow_empty = FALSE)
    matched_feat <- intersect(unique(features), features_use)
    idx_feat <- match(matched_feat, features_use)

    if (length(idx_feat) == 0) {
        err <- sprintf("No 'features' in 'features_use'")
        stop(err)
    }

    .validate_ace(ace, allow_se_like = FALSE, allow_null = FALSE, obj_name = "ace", return_elem = FALSE, error_on_fail = TRUE)

    X0 <- .ace_or_assay(
        ace,
        assay_name = assay_name,
        allow_se_like = FALSE,
        return_elem = TRUE
    )[idx_feat, , drop = FALSE]


    if (algorithm == "pca") {
        pc_smooth <- smoothKernel(
            ace = ace,
            norm_method = norm_method,
            alpha = alpha,
            max_it = max_it,
            reduction_slot = reduction_slot,
            net_slot = net_slot,
            thread_no = thread_no,
            return_raw = TRUE
        )

        H <- pc_smooth$H
        W <- pc_smooth$SVD.out$u
        W <- W[idx_feat, , drop = FALSE]

        out <- W %*% Matrix::t(H)
        out[out < 0] <- 0
        # } else if (algorithm == "action") { # TODO: Fix this!! We need to also impute C. What alpha values?
        #     if (!("archetype_footprint" %in% names(colMaps(ace))) | (force_reimpute == TRUE)) {
        #         H <- networkDiffusion(
        #             obj = ace,
        #             scores = colMaps(ace)[["H_merged"]],
        #             norm_method = norm_method,
        #             alpha = alpha,
        #             thread_no = thread_no,
        #             max_it = max_it,
        #             net_slot = net_slot
        #         )
        #     } else {
        #         H <- ace$archetype_footprint
        #     }
        #     C <- colMaps(ace)$C_merged
        #     W <- as.matrix(X0 %*% C)
        #     out <- W %*% Matrix::t(H)
    } else {
        out <- networkDiffusion(
            obj = ace,
            scores = Matrix::t(X0),
            norm_method = norm_method,
            alpha = alpha,
            thread_no = thread_no,
            approx = TRUE,
            max_it = max_it,
            net_slot = net_slot,
            return_raw = TRUE
        )
        out <- Matrix::t(out)
    }

    # Re-scale expression of features
    m1 <- apply(X0, 1, max)
    m2 <- apply(out, 1, max)
    ratio <- m1 / m2
    ratio[m2 == 0] <- 1
    D <- Matrix::Diagonal(NROW(out), ratio)
    out <- Matrix::t(as.matrix(D %*% out))

    colnames(out) <- matched_feat
    rownames(out) <- colnames(ace)
    return(out)
}


#' Imputing expression of genes by interpolating over archetype profile
#'
#' @param ace ACTIONet output
#' @param genes List of genes to impute
#' @param features_use A vector of features of length NROW(ace) or the name of a column of rowData(ace) containing the genes given in 'genes'.
#'
#' @return A matrix of imputed expression values
#'
#' @examples
#' expression_imputed <- impute.genes.using.archetype(ace, genes)
#' @export
impute.genes.using.archetypes <- function(ace, genes, features_use = NULL) {
    features_use <- .get_feature_vec(ace, features_use = features_use)
    matched_feat <- intersect(unique(genes), features_use)
    idx_feat <- match(matched_feat, features_use)

    Z <- rowMaps(ace)[["archetype_gene_profile"]][idx_feat, , drop = FALSE]
    H <- Matrix::t(colMaps(ace)[["H_merged"]])

    expression_imputed <- Matrix::t(Z %*% H)
    colnames(expression_imputed) <- matched_feat

    return(expression_imputed)
}


#' Imputing expression specificity of genes by interpolating over archetype profile
#'
#' @param ace ACTIONet output
#' @param genes List of genes to impute
#' @param features_use A vector of features of length NROW(ace) or the name of a column of rowData(ace) containing the genes given in 'genes'.
#'
#' @return A matrix of imputed expression values
#'
#' @examples
#' expression_imputed <- impute.genes.using.archetype(ace, genes)
#' @export
impute.specific.genes.using.archetypes <- function(ace, genes, features_use = NULL) {
    features_use <- .get_feature_vec(ace, features_use = features_use)
    matched_feat <- intersect(unique(genes), features_use)
    idx_feat <- match(matched_feat, features_use)

    Z <- log1p(rowMaps(ace)[["arch_feat_spec"]][idx_feat, , drop = FALSE])
    H <- Matrix::t(colMaps(ace)[["H_merged"]])

    expression_imputed <- Matrix::t(Z %*% H)
    colnames(expression_imputed) <- matched_feat

    return(expression_imputed)
}
