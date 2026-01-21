#' Infer cell annotations from imputed gene expression for all cells.
#'
#' @param ace ACTIONetExperiment object
#' @param markers A named list of marker genes.
#' @param features_use A vector of features of length NROW(ace) or the name of a column of rowData(ace) containing the genes given in 'markers'.
#' @param alpha_val Random-walk parameter for gene imputation.
#' @param thread_no Number of parallel threads used for gene imputation.
#' @param net_slot Name of slot in colNets(ace) containing the network to use for gene expression imputation (default="actionet").
#' @param assay_name Name of assay for which to impute gene expression (default="logcounts").
#' @return A named list: \itemize{
#' \item Label: Inferred cell type labels
#' \item Confidence: Confidence of inferred labels
#' \item Enrichment: Cell type score matrix.
#' }
#'
#' @examples
#' data("curatedMarkers_human") # pre-packaged in ACTIONet
#' markers <- curatedMarkers_human$Blood$PBMC$Monaco2019.12celltypes$marker.genes
#' annots <- annotate.cells.using.markers(ace, markers = markers)
#' plot.ACTIONet(ace, annots$Label, annots$Confidence)
#' @export
annotateCells <- function(
    ace,
    markers,
    method = c("vision", "actionet"),
    features_use = NULL,
    assay_name = "logcounts",
    net_slot = "actionet",
    norm_method = c("pagerank", "pagerank_sym"),
    alpha = 0.85,
    max_it = 5,
    approx = TRUE,
    ignore_baseline = FALSE,
    use_enrichment = TRUE,
    use_lpa = FALSE,
    thread_no = 0) {
  method <- match.arg(method)
  norm_method <- match.arg(norm_method)
  norm_method <- ifelse(norm_method == "pagerank_sym", 2, 0)

  .validate_ace(ace, allow_se_like = FALSE, return_elem = FALSE, error_on_fail = TRUE)
  X <- .encode_markers(ace, markers = markers, features_use = features_use, obj_name = "ace")
  S <- .validate_assay(ace, assay_name = assay_name, matrix_type = "sparse", force_type = TRUE, error_on_fail = TRUE, return_elem = TRUE)
  G <- .validate_net(ace, net_slot = net_slot, matrix_type = "sparse", force_type = TRUE, return_elem = TRUE)

  if (method == "vision") {
    marker_stats <- C_computeFeatureStatsVision(
      G = G,
      S = S,
      X = X,
      norm_method = norm_method,
      alpha = alpha,
      max_it = max_it,
      approx = approx,
      thread_no = thread_no
    )
  } else if (method == "actionet") {
    marker_stats <- C_computeFeatureStats(
      G = G,
      S = S,
      X = X,
      norm_method = norm_method,
      alpha = alpha,
      max_it = max_it,
      approx = approx,
      thread_no = thread_no,
      ignore_baseline = ignore_baseline
    )
  }

  colnames(marker_stats) <- colnames(X)
  marker_stats[!is.finite(marker_stats)] <- 0
  enrichment <- marker_stats

  if (use_enrichment) {
    Gn <- Matrix::t(C_normalizeGraph(G, norm_method = 1))
    marker_stats[marker_stats < 0] <- 0
    logPvals <- C_computeGraphLabelEnrichment(Gn, marker_stats)

    labels <- colnames(X)[apply(logPvals, 1, which.max)]
    confidence <- apply(logPvals, 1, max)
  } else {
    labels <- colnames(X)[apply(marker_stats, 1, which.max)]
    confidence <- apply(marker_stats, 1, max)
  }

  out <- list(labels = labels, confidence = confidence, enrichment = enrichment)

  if (use_lpa) {
    out[["labels_corrected"]] <- propagateLabels(
      ace,
      labels = labels,
      which_fixed = NULL,
      algorithm = "LPA",
      net_slot = net_slot,
      thread_no = thread_no
    )
  }

  return(out)
}

.preprocess_annotation_markers <- function(markers, feature_set) {
  if (is.matrix(markers) || ACTIONetExperiment:::is.sparseMatrix(markers)) {
    common_features <- sort(unique(intersect(feature_set, rownames(markers))))
    row_idx <- match(common_features, rownames(markers))
    X <- markers[row_idx, ]
    rownames(X) <- common_features
  } else if (is.list(markers)) {
    if (is.null(names(markers))) {
      names(markers) <- sapply(1:length(markers), function(i) {
        msg <- sprintf("Annotation %d", i)
        print(msg)
      })
    }
    X <- do.call(cbind, lapply(markers, function(gs) {
      genes <- unlist(strsplit(gs, "[+]|[-]"))
      if (length(grep("[+|-]$", gs)) != 0) {
        x <- as.numeric(grepl("[+]", gs)) - as.numeric(grepl("[-]", gs))
        genes <- as.character(sapply(gs, function(gene) substr(gene, 1, stringr::str_length(gene) - 1)))
      } else {
        x <- rep(1, length(genes))
        genes <- gs
      }
      common.genes <- sort(unique(intersect(genes, feature_set)))
      idx1 <- match(common.genes, genes)
      idx2 <- match(common.genes, feature_set)

      v <- sparseMatrix(i = idx2, j = rep(1, length(idx2)), x = x[idx1], dims = c(length(feature_set), 1))
      return(v)
    }))
    colnames(X) <- names(markers)
    rownames(X) <- feature_set
  } else if (is.data.frame(markers) || (length(which(is(markers) == "DFrame")) != 0)) { # marker, cell type, [weight]
    if (ncol(markers) == 2) {
      markers$weight <- 1
    }
    UL <- sort(unique(markers[, 2]))
    X <- do.call(cbind, lapply(UL, function(nn) {
      idx <- which(markers[, 2] == nn)
      genes <- markers[idx, 1]
      x <- markers[idx, 3]

      common.genes <- sort(unique(intersect(genes, feature_set)))
      idx1 <- match(common.genes, genes)
      idx2 <- match(common.genes, feature_set)

      v <- sparseMatrix(i = idx2, j = rep(1, length(idx2)), x = x[idx1], dims = c(length(feature_set), 1))
      return(v)
    }))
    colnames(X) <- UL
    rownames(X) <- feature_set
  }

  return(X)
}


#' Annotate archetypes using prior cell annotations
#' (It uses t-test on the archetype footprint matrix (H))
#'
#' @param ace ACTIONet output object
#' @param labels Annotation of interest (clusters, celltypes, etc.) to test enrichment
#'
#' @return A named list: \itemize{
#' \item Label: Inferred archetype labels
#' \item Confidence: Confidence of inferred labels
#' \item Enrichment: Full enrichment matrix
#' }
#'
#' @examples
#' arch.annot <- annotate.archetypes.using.labels(ace, sce$celltypes)
#' @export
annotate.archetypes.using.labels <- function(ace,
                                             labels,
                                             archetype.slot = "H_merged", algorithm = "ttest") {
  Labels <- .preprocess_annotation_labels(labels, ace)

  if (is.matrix(ace) | ACTIONetExperiment:::is.sparseMatrix(ace)) {
    profile <- as.matrix(ace)
  } else {
    profile <- Matrix::t(colMaps(ace)[[archetype.slot]])
  }
  Annot <- names(Labels)[match(sort(unique(Labels)), Labels)]

  if (algorithm == "wilcox") {
    wilcox.out <- presto::wilcoxauc(profile, Annot[Labels])
    Enrichment <- do.call(cbind, split(-log10(wilcox.out$pval) * sign(wilcox.out$auc - 0.5), wilcox.out$group))
  } else {
    # Using t-statistics
    Enrichment <- sapply(Annot, function(label) {
      mask <- names(Labels) == label
      class.profile <- profile[, mask]
      null.profile <- profile[, !mask]

      N.class <- sum(mask)
      N.null <- sum(!mask)

      if ((N.class < 3) | (N.null < 3)) {
        return(rep(0, nrow(profile)))
      }

      mu.class <- ACTIONetExperiment:::fastRowMeans(class.profile)
      mu.null <- ACTIONetExperiment:::fastRowMeans(null.profile)

      sigma_sq.class <- apply(class.profile, 1, var)
      sigma_sq.null <- apply(null.profile, 1, var)


      delta.mean <- mu.class - mu.null
      t.stat <- delta.mean / sqrt((sigma_sq.class / N.class) + (sigma_sq.null / N.null))
      return(t.stat)
    })
  }

  Enrichment[is.na(Enrichment)] <- 0
  archetypeLabels <- Annot[apply(Enrichment, 1, which.max)]
  Labels.confidence <- apply(Enrichment, 1, max)

  rownames(Enrichment) <- paste("A", 1:nrow(Enrichment), "-", archetypeLabels,
    sep = ""
  )

  out <- list(
    Label = archetypeLabels,
    Confidence = Labels.confidence,
    Enrichment = Enrichment
  )

  return(out)
}


#' Annotate clusters using known marker genes
#' (It uses permutation test on cluster specificity scores)
#'
#' @param ace ACTIONet output object
#' @param markers A list of lists (each a set of markers for a given cell type)
#' @param rand_sample_no Number of random permutations (default=1000)
#'
#' @return A named list: \itemize{
#' \item Label: Inferred archetype labels
#' \item Confidence: Confidence of inferred labels
#' \item Enrichment: Full enrichment matrix
#' }
#'
#' @examples
#' data("curatedMarkers_human") # pre-packaged in ACTIONet
#' markers <- curatedMarkers_human$Blood$PBMC$Monaco2019.12celltypes$marker.genes
#' arch.annot <- annotate.archetypes.using.markers(ace, markers = markers)
#' @export
annotate.archetypes.using.markers <- function(ace,
                                              markers,
                                              features_use = NULL,
                                              significance_slot = "arch_feat_spec") {
  features_use <- .get_feature_vec(ace, features_use)
  marker_mat <- .preprocess_annotation_markers(markers, features_use)

  marker_stats <- Matrix::t(assess.geneset.enrichment.from.archetypes(ace, marker_mat)$logPvals)
  colnames(marker_stats) <- colnames(marker_mat)

  marker_stats[!is.finite(marker_stats)] <- 0
  annots <- colnames(marker_mat)[apply(marker_stats, 1, which.max)]
  conf <- apply(marker_stats, 1, max)

  out <- list(
    Label = annots,
    Confidence = conf,
    Enrichment = marker_stats
  )

  return(out)
}


annotateArchetypes <- function(ace, markers = NULL, labels = NULL, scores = NULL, archetype_slot = "H_merged", archetype_specificity_slot = "arch_feat_spec") {
  annotations.count <- is.null(markers) + is.null(labels) + is.null(scores)
  if (annotations.count != 2) {
    stop("Exactly one of the `markers`, `labels`, or `scores` can be provided.")
  }

  if (!is.null(markers)) {
    features_use <- .get_feature_vec(ace, NULL)
    marker_mat <- as(.preprocess_annotation_markers(markers, features_use), "sparseMatrix")

    archetype_feat_spec <- as.matrix(rowMaps(ace)[[archetype_specificity_slot]])
    colnames(archetype_feat_spec) <- paste("A", 1:ncol(archetype_feat_spec), sep = "")

    archetype_enrichment <- Matrix::t(assess_enrichment(archetype_feat_spec, marker_mat)$logPvals)
    rownames(archetype_enrichment) <- colnames(archetype_feat_spec)
    colnames(archetype_enrichment) <- colnames(marker_mat)
  } else if (!is.null(labels)) {
    X1 <- as.matrix(colMaps(ace)[[archetype_slot]])
    colnames(X1) <- paste("A", 1:ncol(X1), sep = "")

    if (length(labels) == 1) {
      l2 <- colData(ace)[[labels]]
    } else {
      l2 <- labels
    }
    f2 <- factor(l2)
    X2 <- model.matrix(~ .0 + f2)

    xi.out <- XICOR(X1, X2)
    Z_pos <- xi.out$Z
    Z_pos[Z_pos < 0] <- 0
    dir <- sign(cor(X1, X2))
    archetype_enrichment <- dir * Z_pos

    rownames(archetype_enrichment) <- colnames(X1)
    colnames(archetype_enrichment) <- levels(f2)
  } else if (!is.null(scores)) {
    X1 <- as.matrix(colMaps(ace)[[archetype_slot]])
    colnames(X1) <- paste("A", 1:ncol(X1), sep = "")

    if (length(scores) == 1) {
      X2 <- as.matrix(colMaps(ace)[[scores]])
    } else {
      X2 <- as.matrix(scores)
    }

    xi.out <- XICOR(X1, X2)
    Z_pos <- xi.out$Z
    Z_pos[Z_pos < 0] <- 0
    dir <- sign(cor(X1, X2))
    archetype_enrichment <- dir * Z_pos

    rownames(archetype_enrichment) <- colnames(X1)
    colnames(archetype_enrichment) <- colnames(X2)
  }
  archetype_enrichment[!is.finite(archetype_enrichment)] <- 0
  annots <- colnames(archetype_enrichment)[apply(archetype_enrichment, 1, which.max)]
  conf <- apply(archetype_enrichment, 1, max)

  out <- list(
    Label = annots,
    Confidence = conf,
    Enrichment = archetype_enrichment
  )

  return(out)
}


annotateClusters <- function(ace, markers = NULL, labels = NULL, scores = NULL, cluster_name = "cluster") {
  annotations.count <- is.null(markers) + is.null(labels) + is.null(scores)
  if (annotations.count != 2) {
    stop("Exactly one of the `markers`, `labels`, or `scores` can be provided.")
  }

  if (!is.null(markers)) {
    features_use <- .get_feature_vec(ace, NULL)
    marker_mat <- as(.preprocess_annotation_markers(markers, features_use), "sparseMatrix")

    scores <- as.matrix(rowMaps(ace)[[sprintf("%s_feat_spec", cluster_name)]])
    cluster_enrichment <- Matrix::t(assess_enrichment(scores, marker_mat)$logPvals)
    rownames(cluster_enrichment) <- colnames(scores)
    colnames(cluster_enrichment) <- colnames(marker_mat)
  } else if (!is.null(labels)) {
    l1 <- colData(ace)[[cluster_name]]
    if (length(labels) == 1) {
      l2 <- colData(ace)[[labels]]
    } else {
      l2 <- labels
    }

    f1 <- factor(l1)
    f2 <- factor(l2)

    X1 <- model.matrix(~ .0 + f1)
    X2 <- model.matrix(~ .0 + f2)

    xi.out <- XICOR(X1, X2)
    Z_pos <- xi.out$Z
    Z_pos[Z_pos < 0] <- 0
    dir <- sign(cor(X1, X2))
    cluster_enrichment <- dir * Z_pos

    rownames(cluster_enrichment) <- levels(f1)
    colnames(cluster_enrichment) <- levels(f2)
  } else if (!is.null(scores)) {
    if (length(scores) == 1) {
      X2 <- as.matrix(colMaps(ace)[[scores]])
    } else {
      X2 <- as.matrix(scores)
    }

    l1 <- colData(ace)[[cluster_name]]
    f1 <- factor(l1)
    X1 <- model.matrix(~ .0 + f1)

    xi.out <- XICOR(X1, X2)
    Z_pos <- xi.out$Z
    Z_pos[Z_pos < 0] <- 0
    dir <- sign(cor(X1, X2))
    cluster_enrichment <- dir * Z_pos

    rownames(cluster_enrichment) <- levels(l1)
    colnames(cluster_enrichment) <- colnames(X2)
  }
  cluster_enrichment[!is.finite(cluster_enrichment)] <- 0
  annots <- colnames(cluster_enrichment)[apply(cluster_enrichment, 1, which.max)]
  conf <- apply(cluster_enrichment, 1, max)

  out <- list(
    Label = annots,
    Confidence = conf,
    Enrichment = cluster_enrichment
  )

  return(out)
}

projectArchs <- function(ace, archtype_scores, archetype_slot = "H_merged", normalize = TRUE) {
  cell.enrichment.mat <- map.cell.scores.from.archetype.enrichment(
    ace = ace,
    enrichment_mat = archtype_scores,
    normalize = normalize,
    H.slot = archetype_slot
  )
  cell.annotations <- colnames(cell.enrichment.mat)[apply(
    cell.enrichment.mat, 1,
    which.max
  )]

  Labels <- colnames(cell.enrichment.mat)[apply(cell.enrichment.mat, 1, which.max)]
  Labels.confidence <- apply(cell.enrichment.mat, 1, max)

  res <- list(
    Label = Labels,
    Confidence = Labels.confidence,
    Enrichment = cell.enrichment.mat
  )

  return(res)
}

#' Annotate clusters using prior cell annotations
#' (It uses Fisher's exact test for computing overlaps -- approximate HGT is used)
#'
#' @param ace ACTIONet output object
#' @param clusters Cluster
#' @param labels Annotation of interest (clusters, celltypes, etc.) to test enrichment
#'
#' @return A named list: \itemize{
#' \item Labels: Inferred archetype labels
#' \item Labels.confidence: Confidence of inferred labels
#' \item Enrichment: Full enrichment matrix
#' }
#'
#' @examples
#' arch.annot <- annotate.clusters.using.labels(ace, ace$clusters, sce$celltypes)
#' @export
annotate.clusters.using.labels <- function(ace,
                                           clusters,
                                           labels) {
  clusters <- .preprocess_annotation_labels(clusters, ace)
  Labels <- .preprocess_annotation_labels(labels, ace)

  pop.size <- length(Labels)
  pos.size <- table(Labels)

  logPvals <- sapply(sort(unique(clusters)), function(i) {
    idx <- which(clusters == i)
    sample.size <- length(idx)
    success.size <- sapply(sort(unique(Labels)), function(i) {
      sum(Labels[idx] == i)
    })

    logPval <- HGT_tail(
      population.size = pop.size,
      success.count = pos.size,
      sample.size = sample.size,
      observed.success = success.size
    )

    return(logPval)
  })


  cl.Annot <- names(clusters)[match(sort(unique(clusters)), clusters)]
  Annot <- names(Labels)[match(sort(unique(Labels)), Labels)]

  colnames(logPvals) <- cl.Annot
  rownames(logPvals) <- Annot

  clusterLabels <- Annot[apply(logPvals, 2, which.max)]

  cellLabels <- match(clusterLabels[clusters], Annot)
  names(cellLabels) <- clusterLabels[clusters]

  res <- list(
    Labels = clusterLabels,
    cellLabels = cellLabels,
    Enrichment = logPvals
  )

  return(res)
}


#' Annotate clusters using known marker genes
#' (It uses permutation test on cluster specificity scores)
#'
#' @param ace ACTIONet output object
#' @param marker.genes A list of lists (each a set of markers for a given cell type)
#' @param specificity.slot.name An entry in the rowMaps(ace), precomputed using computeFeatureSpecificity() function
#' @param rand.sample.no Number of random permutations (default=1000)
#'
#' @return A named list: \itemize{
#' \item Labels: Inferred archetype labels
#' \item Labels.confidence: Confidence of inferred labels
#' \item Enrichment: Full enrichment matrix
#' }
#'
#' @examples
#' data("curatedMarkers_human") # pre-packaged in ACTIONet
#' marker.genes <- curatedMarkers_human$Blood$PBMC$Monaco2019.12celltypes$marker.genes
#' ace <- computeFeatureSpecificity(ace, ace$clusters, "cluster_specificity_scores")
#' arch.annot <- annotate.clusters.using.markers(ace, marker.genes = marker.genes, specificity.slot.name = "cluster_specificity_scores")
#' @export
annotate.clusters.using.markers <- function(ace,
                                            marker.genes,
                                            specificity.slot.name,
                                            rand.sample.no = 1000) {
  if (!(specificity.slot.name %in% names(rowMaps(ace)))) {
    message(sprintf("%s does not exist in rowMaps(ace)", specificity.slot.name))
  }

  if (is.matrix(marker.genes) | ACTIONetExperiment:::is.sparseMatrix(marker.genes)) {
    marker.genes <- apply(marker.genes, 2, function(x) {
      rownames(marker.genes)[x >
        0]
    })
  }

  specificity.panel <- Matrix::t(as.matrix(log1p(rowMaps(ace)[[specificity.slot.name]])))

  GS.names <- names(marker.genes)
  if (is.null(GS.names)) {
    GS.names <- sapply(1:length(GS.names), function(i) sprintf("Celltype %s", i))
  }

  markers.table <- do.call(rbind, lapply(names(marker.genes), function(celltype) {
    genes <- marker.genes[[celltype]]

    if (length(genes) == 0) {
      err <- sprintf("No markers left.\n")
      stop(err, call. = FALSE)
    }

    signed.count <- sum(sapply(genes, function(gene) grepl("\\+$|-$", gene)))
    is.signed <- signed.count > 0

    if (!is.signed) {
      df <- data.frame(
        Gene = genes,
        Direction = +1,
        Celltype = celltype,
        stringsAsFactors = FALSE
      )
    } else {
      pos.genes <- (as.character(sapply(
        genes[grepl("+", genes, fixed = TRUE)],
        function(gene) stringr::str_replace(gene, stringr::fixed("+"), "")
      )))
      neg.genes <- (as.character(sapply(
        genes[grepl("-", genes, fixed = TRUE)],
        function(gene) stringr::str_replace(gene, stringr::fixed("-"), "")
      )))

      df <- data.frame(
        Gene = c(pos.genes, neg.genes),
        Direction = c(rep(+1, length(pos.genes)), rep(-1, length(neg.genes))),
        Celltype = celltype,
        stringsAsFactors = FALSE
      )
    }
  }))

  markers.table <- markers.table[markers.table$Gene %in% colnames(specificity.panel), ]

  if (dim(markers.table)[1] == 0) {
    err <- sprintf("No markers left.\n")
    stop(err, call. = FALSE)
  }

  specificity.panel <- specificity.panel[, markers.table$Gene]

  IDX <- split(1:dim(markers.table)[1], markers.table$Celltype)

  print("Computing significance scores")
  set.seed(0)
  Z <- sapply(IDX, function(idx) {
    markers <- (as.character(markers.table$Gene[idx]))
    directions <- markers.table$Direction[idx]
    mask <- markers %in% colnames(specificity.panel)

    A <- as.matrix(specificity.panel[, markers[mask]])
    sgn <- as.numeric(directions[mask])
    stat <- A %*% sgn

    rand.stats <- sapply(1:rand.sample.no, function(i) {
      rand.samples <- sample.int(dim(specificity.panel)[2], sum(mask))
      rand.A <- as.matrix(specificity.panel[, rand.samples])
      rand.stat <- rand.A %*% sgn
    })

    cell.zscores <- as.numeric((stat - apply(rand.stats, 1, mean)) / apply(
      rand.stats,
      1, sd
    ))

    return(cell.zscores)
  })

  Z[is.na(Z)] <- 0
  Labels <- colnames(Z)[apply(Z, 1, which.max)]

  Labels.conf <- apply(Z, 1, max)

  names(Labels) <- rownames(specificity.panel)
  names(Labels.conf) <- rownames(specificity.panel)
  rownames(Z) <- rownames(specificity.panel)

  out <- list(
    Label = Labels,
    Confidence = Labels.conf,
    Enrichment = Z
  )

  return(out)
}


#' Annotate arbitary feature score matrix using known marker genes
#' (It uses permutation test on cluster specificity scores)
#'
#' @param profile An arbitrary matrix with rows corresponding to features and columns to any given annotation/grouping of cells
#' @param markers A list of lists (each a set of markers for a given cell type)
#'
#' @return A named list: \itemize{
#' \item Labels: Inferred archetype labels
#' \item Labels.confidence: Confidence of inferred labels
#' \item Enrichment: Full enrichment matrix
#' }
#'
#' @export
annotate.profile.using.markers <- function(profile, markers) {
  marker_mat <- .preprocess_annotation_markers(markers, rownames(profile))

  enrichment.out <- assess_enrichment(profile, marker_mat)

  X <- enrichment.out$logPvals
  rownames(X) <- colnames(marker_mat)
  colnames(X) <- colnames(profile)

  return(X)
}
