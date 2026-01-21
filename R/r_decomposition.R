#' @export
runSVD <- function(
    X,
    k = 30,
    algorithm = c("irlb", "halko", "feng"),
    max_it = NULL,
    seed = 0,
    verbose = TRUE) {
  algorithm <- match.arg(algorithm)
  algorithm <- switch(algorithm,
    "irlb" = 0,
    "halko" = 1,
    "feng" = 2
  )

  if (is.null(max_it)) {
    max_it <- ifelse(algorithm == 0, 1000, 5)
  }

  X <- .validate_matrix(X)
  if (is.matrix(X)) {
    out <- C_runSVDDense(A = X, k = k, max_it = max_it, seed = seed, algorithm = algorithm, verbose = verbose)
  } else {
    out <- C_runSVDSparse(A = X, k = k, max_it = max_it, seed = seed, algorithm = algorithm, verbose = verbose)
  }

  return(out)
}
