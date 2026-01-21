#' @export
getFeatureAbundance <- function(
    obj,
    features,
    nonzero = FALSE,
    labels = NULL,
    labels_use = NULL,
    features_use = NULL,
    metric = c("fraction", "percent", "ratio", "counts"),
    assay_name = "counts") {
  metric <- match.arg(metric, several.ok = TRUE)[1]

  if (length(features) == 1 && features == "all") {
    using_all <- TRUE
  } else {
    if (length(features) == 0) {
      stop("'features' must be 'all' or a subset of 'features_use'")
    }
    features_use <- .get_features(obj, features_use = features_use, allow_empty = FALSE, features_name = "features_use")
    using_all <- all(features_use %in% features)
  }

  X <- .ace_or_assay(obj, assay_name = assay_name, allow_se_like = TRUE, return_elem = TRUE)
  .warn_dgt(X)

  if (nonzero) {
    X <- X > 0
    if (!is.matrix(X)) {
      X <- as(X, "dMatrix")
    }
  }

  cs_mat <- Matrix::colSums(X)
  if (using_all) {
    if (metric != "counts") {
      stop("'all' features only valid for metric 'counts'")
    }
    cs_mm <- cs_mat
  } else {
    idx_feat <- which(features_use %in% features)

    if (length(idx_feat) < 1) {
      stop("No matching features")
    }

    cs_mm <- Matrix::colSums(X[idx_feat, , drop = FALSE])
    # if (!is.matrix(X)) {
    #   cs_mm <- C_fastSpMatViewSum(
    #     X = X,
    #     idx = idx_feat,
    #     dim = 2,
    #     threads = .get_num_threads(.SYS_THREADS_DEF, 0)
    #   )
    # } else {
    #   cs_mm <- Matrix::colSums(X[idx_feat, , drop = FALSE])
    # }
  }

  if (!is.null(labels)) {
    label.split <- .validate_vector_attr(
      obj,
      attr = labels,
      groups_use = labels_use,
      return_type = "split",
      keep_factor = TRUE,
      attr_name = "labels",
      return_elem = TRUE
    )

    if (metric %in% c("fraction", "percent")) {
      out <- lapply(label.split, function(idx) {
        x <- cs_mm[idx] / cs_mat[idx]
        if (metric == "percent") {
          x <- 100 * x
        }
        return(x)
      })
    } else if (metric == "ratio") {
      out <- lapply(label.split, function(idx) {
        x <- cs_mm[idx] / (cs_mat[idx] - cs_mm[idx])
      })
    } else {
      out <- lapply(label.split, function(idx) {
        x <- cs_mm[idx]
      })
    }
  } else {
    if (metric %in% c("fraction", "percent")) {
      out <- cs_mm / cs_mat
      if (metric == "percent") {
        out <- 100 * out
      }
    } else if (metric == "ratio") {
      out <- cs_mm / (cs_mat - cs_mm)
    } else {
      out <- cs_mm
    }
    return(out)
  }
}

#' @export
getMitoAbundance <- function(
    obj,
    labels = NULL,
    labels_use = NULL,
    features_use = NULL,
    id_type = c("gene_name", "ensembl_id"),
    species = c("hsapiens", "mmusculus", "human", "mouse"),
    protein_coding = FALSE,
    metric = c("fraction", "percent", "ratio", "counts"),
    assay_name = "counts") {
  feats_mito <- .get_mito_feats(id_type = id_type, species = species, protein_coding = protein_coding)

  out <- getFeatureAbundance(
    obj,
    features = feats_mito,
    labels = labels,
    labels_use = labels_use,
    features_use = features_use,
    metric = metric
  )
  return(out)
}

.get_mito_feats <- function(
    id_type = c("gene_name", "ensembl_id"),
    species = c("hsapiens", "mmusculus", "human", "mouse"),
    protein_coding = FALSE) {
  id_type <- tolower(id_type)
  id_type <- match.arg(id_type, several.ok = TRUE)[1]

  species <- tolower(species)
  species <- match.arg(species, several.ok = TRUE)[1]

  if (species == "human") {
    species <- "hsapiens"
  }

  if (species == "mouse") {
    species <- "mmusculus"
  }

  feats_tbl <- int_feats_mito[int_feats_mito[["species"]] == species, , drop = FALSE]

  if (protein_coding) {
    feats_tbl <- feats_tbl[feats_tbl[["feature_type"]] == "protein_coding", , drop = FALSE]
  }

  out <- feats_tbl[[id_type]]
  return(out)
}
