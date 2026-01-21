#' Perform batch correction on `ACTIONetExperiment` and `SingleCellExperiment` objects.

#' @export
correctBatchEffectFastMNN <- function(
    ace,
    batch_attr = NULL,
    assay_name = NULL,
    reduced_dim = 50,
    MNN_k = 20,
    reduction_out = "MNN",
    BPPARAM = SerialParam()) {
  ACTIONetExperiment:::.check_and_load_package(c("scran", "SingleCellExperiment", "batchelor", "BiocParallel"))
  ace <- .validate_ace(ace, as_ace = TRUE, allow_se_like = TRUE, fix_dimnames = TRUE, return_elem = TRUE)
  S <- .validate_assay(ace, assay_name = assay_name, return_elem = TRUE)
  batch_vec <- .validate_vector_attr(
    ace,
    attr = batch_attr,
    return_type = "data",
    attr_name = "batch_attr",
    obj_name = "ace",
    return_elem = TRUE
  )

  IDX <- split(seq_len(length(batch_vec)), f = batch_vec)
  merge_order <- order(sapply(IDX, function(idx) length(idx)), decreasing = TRUE)

  set.seed(0)
  mnn.out <- batchelor::fastMNN(
    S,
    batch = batch_vec,
    k = MNN_k,
    d = reduced_dim,
    auto.merge = FALSE,
    merge.order = merge_order,
    cos.norm = FALSE,
    BPPARAM = BPPARAM
  )

  S_r <- SingleCellExperiment::reducedDims(mnn.out)[["corrected"]]
  rownames(S_r) <- colnames(ace)
  colnames(S_r) <- sapply(seq_len(dim(S_r)[2]), function(i) sprintf("PC%d", i))

  colMaps(ace)[[reduction_out]] <- S_r
  colMapTypes(ace)[[reduction_out]] <- "reduction"

  V <- rowData(mnn.out)[["rotation"]]
  colnames(V) <- paste0("V", seq_len(NCOL(V)))
  rowMaps(ace)[[sprintf("%s_V", reduction_out)]] <- V
  rowMapTypes(ace)[[sprintf("%s_V", reduction_out)]] <- "internal"

  invisible(gc())

  return(ace)
}

# TODO: Allow design_mat to be multiple input types.
#' @export
correctBatchEffect <- function(
    ace,
    batches = NULL,
    design = NULL,
    reduction_slot = "action",
    corrected_suffix = "orth",
    assay_name = "logcounts") {
  if (!is.null(batches) && !is.null(design)) {
    err <- sprintf("Only one of 'batches' or 'design' must be specified")
    stop(err)
  } else {
    if (!is.null(batches)) {
      batches <- .validate_vector_attr(
        ace,
        attr = batches,
        return_type = "data",
        keep_factor = TRUE,
        attr_name = "batches",
        obj_name = "ace",
        return_elem = TRUE
      )
      batches <- as.factor(batches)
      if (length(levels(batches)) < 2) {
        err <- sprintf("'batches' must have >=2 levels")
        stop(err)
      }
      design_mat <- model.matrix(~ 0 + batches)
    } else {
      design_mat <- .make_design_mat(design = design, data = colData(ace), remove_intercept = TRUE)
      if (NROW(design_mat) != NCOL(ace)) {
        err <- sprintf("Size of 'design' does not match 'NCOL(ace)'")
      }
    }
  }

  ace <- .validate_ace(ace, allow_se_like = FALSE, fix_dimnames = TRUE, return_elem = TRUE, error_on_fail = TRUE)
  S <- .validate_assay(ace, assay_name = assay_name, error_on_fail = TRUE, return_elem = TRUE)
  S_r <- .validate_map(ace, map_slot = reduction_slot, matrix_type = "dense", force_type = TRUE, return_elem = TRUE)

  B <- .validate_map(ace, map_slot = sprintf("%s_B", reduction_slot), matrix_type = "dense", force_type = TRUE, return_elem = TRUE, row = FALSE)
  V <- .validate_map(ace, map_slot = sprintf("%s_V", reduction_slot), matrix_type = "dense", force_type = TRUE, return_elem = TRUE, row = TRUE)
  A <- .validate_map(ace, map_slot = sprintf("%s_A", reduction_slot), matrix_type = "dense", force_type = TRUE, return_elem = TRUE, row = TRUE)
  sigma <- S4Vectors::metadata(ace)[[sprintf("%s_sigma", reduction_slot)]]

  if (length(sigma) != NCOL(S_r)) {
    err <- sprintf("Size of 'sigma' in 'metadata(ace)' does not match reduction\nRecommend re-running 'reduceKernel()'")
    stop(err)
  }

  if (is.matrix(S)) {
    out <- C_orthogonalizeBatchEffect_full(
      S = S,
      old_S_r = S_r,
      old_V = V,
      old_A = A,
      old_B = B,
      old_sigma = sigma,
      design = design_mat
    )
  } else {
    out <- C_orthogonalizeBatchEffect(
      S = S,
      old_S_r = S_r,
      old_V = V,
      old_A = A,
      old_B = B,
      old_sigma = sigma,
      design = design_mat
    )
  }
  S_r <- out$S_r
  colnames(S_r) <- colnames(ace)
  rownames(S_r) <- sapply(seq_len(NROW(S_r)), function(i) sprintf("Dim%d", i))
  name_Sr <- sprintf("%s_%s", reduction_slot, corrected_suffix)
  colMaps(ace)[[name_Sr]] <- Matrix::t(S_r)
  colMapTypes(ace)[[name_Sr]] <- "reduction"


  V <- out$V
  colnames(V) <- sapply(seq_len(dim(V)[2]), function(i) sprintf("V%d", i))
  name_V <- sprintf("%s_V_%s", reduction_slot, corrected_suffix)
  rowMaps(ace)[[name_V]] <- V
  rowMapTypes(ace)[[name_V]] <- "internal"


  A <- out$A
  colnames(A) <- sapply(seq_len(dim(A)[2]), function(i) sprintf("A%d", i))
  name_A <- sprintf("%s_A_%s", reduction_slot, corrected_suffix)
  rowMaps(ace)[[name_A]] <- A
  rowMapTypes(ace)[[name_A]] <- "internal"


  B <- out$B
  colnames(B) <- sapply(seq_len(dim(B)[2]), function(i) sprintf("B%d", i))
  name_B <- sprintf("%s_B_%s", reduction_slot, corrected_suffix)
  colMaps(ace)[[name_B]] <- B
  colMapTypes(ace)[[name_B]] <- "internal"

  name_sigma <- sprintf("%s_sigma_%s", reduction_slot, corrected_suffix)
  S4Vectors::metadata(ace)[[name_sigma]] <- out$sigma

  return(ace)
}
