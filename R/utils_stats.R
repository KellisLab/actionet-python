.create_design_formula <- function(vars) {
  fml <- Reduce(function(x, y) paste(x, y, sep = " + "), vars)
  fml <- paste("~", fml)
  fml <- as.formula(fml)
  return(fml)
}

.make_design_mat <- function(design, data = NULL, remove_intercept = FALSE) {
  if (is(design, "formula")) {
    if (is.null(data)) {
      stop("'data' is missing.")
    }
    tf <- terms(design, keep.order = TRUE)
    if (remove_intercept) {
      attr(tf, "intercept") <- 0
    }
    design_mat <- stats::model.matrix(
      object = tf,
      data = data
    )
  } else if (is.matrix(design)) {
    design_mat <- design
  }
  return(design_mat)
}
