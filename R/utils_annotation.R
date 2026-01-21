.encode_markers <- function(
    obj,
    markers,
    features_use = NULL,
    obj_name = "obj",
    features_name = "features_use") {
  if (any(class(markers) == "list")) {
    mn <- names(markers)
  } else {
    mn <- colnames(markers)
  }

  if (is.null(mn) || any(is.na(mn))) {
    err <- sprintf("'markers' contains unnamed entries")
    stop(err)
  }

  if (any(duplicated(mn))) {
    err <- sprintf("'markers' contains duplicated labels")
    stop(err)
  }

  features_use <- .get_features(obj, features_use = features_use, allow_empty = FALSE)
  if (any(class(markers) %in% c("list", "data.frame"))) {
    X <- sapply(markers, function(x) {
      as.numeric(features_use %in% x)
    })
    X <- as(X, "CsparseMatrix")
  } else if (is.matrix(markers) || ACTIONetExperiment:::is.sparseMatrix(markers)) {
    if (any(!is.finite(markers))) {
      err <- sprintf("'markers' contains non-numeric values")
      stop(err)
    }

    if (NROW(markers) != NROW(obj)) {
      err <- sprintf("NROW(%s) does not match NROW(markers)", obj_name)
      stop(err)
    }
    X <- as(markers, "dMatrix")
    X[X != 0] <- 1
  } else {
    err <- sprintf("'markers' must be one of: 'list', 'data.frame', 'matrix', 'sparseMatrix'")
    stop(err)
  }

  rownames(X) <- features_use

  cs_zero <- (Matrix::colSums(X) == 0)
  if (any(cs_zero)) {
    if (all(cs_zero)) {
      err <- sprintf("No markers in '%s'", features_name)
      stop(err)
    }

    dropped <- names(which(cs_zero))
    for (k in dropped) {
      wrn <- sprintf("Label '%s' has no markers", k)
      message(wrn)
    }
  }

  return(X)
}
