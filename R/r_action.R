#' Compute the reduced kernel matrix and decomposition
#'
#' @param obj ACTIONetExperiment, SummarizedExperiment, SingleCellExperiment or matrix.
#' @param k Dimension of reduced kernel matrix. Number of singular vectors to estimate. Passed to <code>runSVD()</code>.
#' @param algorithm Singular value decomposition algorithm. Passed to <code>runSVD()</code>.
#' @param max_it Number of SVD iterations. If `NULL`: 1000 for "ilrb", 5 otherwise.
#' @param seed Random seed.
#' @param verbose Print status messages.
#' @param assay_name Name of assay to reduce.
#' @param reduction_slot Slot of colMaps(ace) in which to stored reduced kernel matrix. Also the prefix of slots to in which to store output components of the decomposition. (ACTIONetExperiment output only).
#' @param return_raw Return raw output regardless of 'obj' type.
#'
#' @return ACTIONetExperiment object with reduction in colMaps(ace).
#'
#' @export
reduceKernel <- function(
    obj,
    k = 50,
    algorithm = c("irlb", "halko", "feng"),
    max_it = NULL,
    seed = 0,
    verbose = TRUE,
    assay_name = "logcounts",
    reduction_slot = "action",
    return_raw = FALSE) {

  is_ace <- .is_se_like(obj)

  if (is_ace && !return_raw) {
    obj <- .validate_ace(obj, as_ace = TRUE, allow_se_like = TRUE, fix_dimnames = TRUE, return_elem = TRUE, error_on_fail = TRUE)
  }

  X <- .ace_or_assay(
    obj,
    assay_name = assay_name,
    sparse_type = "CsparseMatrix",
    allow_se_like = TRUE,
    return_elem = TRUE
  )

  algorithm <- match.arg(algorithm, several.ok = TRUE)[1]
  algorithm <- switch(algorithm,
    "irlb" = 0,
    "halko" = 1,
    "feng" = 2
  )

  if (is.null(max_it)) {
    max_it <- ifelse(algorithm == 0, 1000, 5)
  }

  if (is.matrix(X)) {
    out <- C_reduceKernelDense(X, k = k, svd_alg = algorithm, max_it = max_it, seed = seed, verbose = verbose)
  } else {
    out <- C_reduceKernelSparse(X, k = k, svd_alg = algorithm, max_it = max_it, seed = seed, verbose = verbose)
  }

  if (is_ace && !return_raw) {
    if (any(class(obj) != "ACTIONetExperiment")) {
      obj <- as(obj, "ACTIONetExperiment")
    }

    S_r <- out$S_r
    colnames(S_r) <- colnames(obj)
    rownames(S_r) <- paste0("dim_", seq_len(NROW(S_r)))
    colMaps(obj)[[reduction_slot]] <- Matrix::t(S_r)
    colMapTypes(obj)[[reduction_slot]] <- "reduction"

    V <- out$V
    colnames(V) <- paste0("V", seq_len(NCOL(V)))
    rowMaps(obj)[[sprintf("%s_V", reduction_slot)]] <- V
    rowMapTypes(obj)[[sprintf("%s_V", reduction_slot)]] <- "internal"

    A <- out$A
    colnames(A) <- paste0("A", seq_len(NCOL(A)))
    rowMaps(obj)[[sprintf("%s_A", reduction_slot)]] <- A
    rowMapTypes(obj)[[sprintf("%s_A", reduction_slot)]] <- "internal"

    B <- out$B
    colnames(B) <- paste0("B", seq_len(NCOL(B)))
    colMaps(obj)[[sprintf("%s_B", reduction_slot)]] <- B
    colMapTypes(obj)[[sprintf("%s_B", reduction_slot)]] <- "internal"

    metadata(obj)[[sprintf("%s_sigma", reduction_slot)]] <- out$sigma

    return(obj)
  }

  return(out)
}


#' @export
decompACTION <- function(
    X,
    k_min,
    k_max,
    max_it = 100,
    tol = 1e-16,
    thread_no = 0) {
  if (k_min < 2) {
    stop("'k_min' must be >=2")
  }

  if (k_max < k_min) {
    stop("'k_max' must be >='k_min'")
  }

  .validate_matrix(X, var_name = "X", matrix_type = "dense", force_type = FALSE, return_elem = FALSE)

  out <- C_decompACTION(
    S_r = X,
    k_min = k_min,
    k_max = k_max,
    max_it = max_it,
    tol = tol,
    thread_no = thread_no
  )

  return(out)
}


#' @export
collectArchetypes <- function(C_trace,
                              H_trace,
                              specificity_th = -3,
                              min_cells_per_arch = 2) {
  out <- C_collectArchetypes(
    C_trace = C_trace,
    H_trace = H_trace,
    spec_th = specificity_th,
    min_obs = min_cells_per_arch
  )

  return(out)
}


#' @export
mergeArchetypes <- function(
    S_r,
    C_stacked,
    H_stacked,
    thread_no = 0) {
  if (!all(dim(C_stacked) == rev(dim(H_stacked)))) {
    err <- sprintf("Dimensions of `C_stacked` and transposed `H_stacked` do not match.\n")
    stop(err)
  }

  out <- C_mergeArchetypes(
    S_r = S_r,
    C_stacked = C_stacked,
    H_stacked = H_stacked,
    thread_no = thread_no
  )

  return(out)
}
