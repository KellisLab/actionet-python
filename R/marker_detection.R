#' @export
findMarkers <- function(
    obj,
    labels,
    labels_use = NULL,
    top_genes = 50,
    features_use = NULL,
    features_keep = NULL,
    assay_name = "logcounts",
    thread_no = 0,
    result = c("table", "ranks", "scores"),
    to_return = c("data.frame", "df", "list")) {
  result <- match.arg(result)
  to_return <- match.arg(to_return)

  features_use <- .get_features(obj, features_use = features_use, allow_empty = FALSE)

  specificity <- computeFeatureSpecificity(
    obj = obj,
    labels = labels,
    labels_use = labels_use,
    assay_name = assay_name,
    thread_no = thread_no,
    return_raw = TRUE
  )

  feat_spec <- specificity[["upper_significance"]] - specificity[["lower_significance"]]
  feat_spec[feat_spec < 0] <- 0
  rownames(feat_spec) <- features_use

  if (!is.null(features_keep)) {
    feat_spec <- feat_spec[rownames(feat_spec) %in% features_keep, , drop = FALSE]
  }

  # if (most_specific == TRUE) {
  #   W <- select.top.k.features(
  #     feat_spec,
  #     top_features = top_genes,
  #     normalize = FALSE,
  #     reorder_columns = FALSE
  #   )
  #   feat_spec_top <- apply(W, 2, function(v) rownames(W)[order(v, decreasing = TRUE)][1:top_genes])
  # }

  if (result == "table") {
    out <- sapply(colnames(feat_spec), function(x) {
      v <- feat_spec[, x]
      v <- sort(v, decreasing = TRUE)
      if (!is.null(top_genes)) {
        v <- head(v, top_genes)
      }
      names(v)
    })
  } else if (result == "ranks") {
    out <- apply(feat_spec, 2, function(v) {
      rank(dplyr::desc(v), ties.method = "max")
    })
  } else if (result == "scores") {
    out <- feat_spec
  }

  out <- data.frame(out)
  if (to_return == "list") {
    out <- as.list(out)
    if (result != "table") {
      out <- lapply(out, function(x) {
        names(x) <- features_use
        return(x)
      })
    }
  }
  return(out)
}
