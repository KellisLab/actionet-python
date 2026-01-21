#' Interpolates cell scores from archetype enrichment matrix
#'
#' @param ace ACTIONet output object
#' @param enrichment_mat Enrichment matrix with rows corresponding to archetypes and columns to an arbitrary annotation
#' @param normalize If TRUE, enrichment matrix will be first doubly-normalized
#'
#' @return Enrichment map of size cell x annotation
#'
#' @examples
#'
#' data("curatedMarkers_human") # pre-packaged in ACTIONet
#' marker_set <- curatedMarkers_human$Blood$PBMC$Monaco2019.12celltypes$marker.genes
#' arch.annot <- annotate.archetypes.using.markers(ace, markers = markers)
#' enrichment.mat <- arch.annot$enrichment
#' cell.enrichment.mat <- map.cell.scores.from.archetype.enrichment(ace, enrichment.mat)
#' cell.assignments <- colnames(cell.enrichment.mat)[apply(cell.enrichment.mat, 1, which.max)]
#' @export
map.cell.scores.from.archetype.enrichment <- function(ace,
                                                      enrichment_mat,
                                                      normalize = FALSE,
                                                      H.slot = "H_merged") {
  cell.scores.mat <- colMaps(ace)[[H.slot]]

  if (nrow(enrichment_mat) != ncol(cell.scores.mat)) {
    print("Flipping enrichment matrix")
    enrichment_mat <- Matrix::t(enrichment_mat)
  }

  if (normalize == TRUE) {
    enrichment.scaled <- doubleNorm(enrichment_mat)
  } else {
    enrichment.scaled <- enrichment_mat
    enrichment.scaled[enrichment.scaled < 0] <- 0
    if (max(enrichment.scaled) > 50) {
      enrichment.scaled <- log1p(enrichment.scaled)
    }
  }

  cell.enrichment.mat <- cell.scores.mat %*% enrichment.scaled
  colnames(cell.enrichment.mat) <- colnames(enrichment_mat)
  rownames(cell.enrichment.mat) <- colnames(ace)

  return(cell.enrichment.mat)
}

scoreCells <- function(ace, markers, algorithm = "gmm2", pre_imputation_algorithm = "none", gene_scaling_method = 0,
                       pre_alpha = 0.15, post_alpha = 0.9, network_normalization_method = "pagerank_sym", diffusion_it = 5, thread_no = 0, features_use = NULL, TFIDF_prenorm = 1, assay_name = "logcounts", net_slot = "actionet", specificity_slot = "arch_feat_spec", H_slot = "H_merged") {
  if (!(net_slot %in% names(colNets(ace)))) {
    warning(sprintf("net_slot does not exist in colNets(ace)."))
    return()
  } else {
    G <- colNets(ace)[[net_slot]]
  }

  features_use <- .get_feature_vec(ace, features_use)
  marker_mat_full <- .preprocess_annotation_markers(markers, features_use)
  mask <- Matrix::rowSums(abs(marker_mat_full)) != 0
  marker_mat <- marker_mat_full[mask, ]

  if (pre_imputation_algorithm == "none") {
    S <- assays(ace)[[assay_name]]
    sub_S <- S[mask, ]
  } else {
    sub_S <- Matrix::t(imputeFeatures(ace, rownames(marker_mat), assay_name = assay_name, thread_no = thread_no, alpha = pre_alpha, diffusion_it = diffusion_it, net_slot = net_slot, algorithm = pre_imputation_algorithm))
  }
  sub_S <- as(sub_S, "matrix")

  network_normalization_code <- 0
  if (network_normalization_method == "pagerank_sym") {
    network_normalization_code <- 2
  }
  if (algorithm == "gmm2") {
    marker_stats <- aggregate_genesets_mahalanobis_2gmm(G, sub_S, marker_mat, network_normalization_method = network_normalization_code, expression_normalization_method = TFIDF_prenorm, gene_scaling_method = gene_scaling_method, pre_alpha = pre_alpha, post_alpha = post_alpha)
  } else if (algorithm == "arch2") {
    marker_stats <- aggregate_genesets_mahalanobis_2archs(G, sub_S, marker_mat, network_normalization_method = network_normalization_code, expression_normalization_method = TFIDF_prenorm, gene_scaling_method = gene_scaling_method, pre_alpha = pre_alpha, post_alpha = post_alpha)
  } else {
    warning(sprintf("Algorithm %s not found. Reverting back to gmm2", algorithm))
    marker_stats <- aggregate_genesets_mahalanobis_2gmm(G, sub_S, sub_marker_mat, network_normalization_method = network_normalization_method, expression_normalization_method = TFIDF_prenorm, gene_scaling_method = gene_scaling_method, pre_alpha = pre_alpha, post_alpha = post_alpha)
  }

  colnames(marker_stats) <- colnames(marker_mat)
  marker_stats[!is.finite(marker_stats)] <- 0
  annots <- colnames(marker_mat)[apply(marker_stats, 1, which.max)]
  conf <- apply(marker_stats, 1, max)

  out <- list(Label = annots, Confidence = conf, Enrichment = marker_stats)

  return(out)
}
