#' Computes feature specificity scores for each cluster.
#' @export
computeFeatureSpecificity <- function(
    obj,
    labels,
    labels_use = NULL,
    assay_name = "logcounts",
    map_out_prefix = "cluster",
    return_lower = FALSE,
    thread_no = 0,
    return_raw = FALSE) {
  is_ace <- .validate_ace(obj, allow_se_like = TRUE, error_on_fail = FALSE, return_elem = FALSE)
  X <- .ace_or_assay(obj = obj, assay_name = assay_name, allow_se_like = TRUE, obj_name = "obj", return_elem = TRUE)

  labels <- .validate_vector_attr(
    obj,
    attr = labels,
    groups_use = labels_use,
    return_type = "data",
    attr_name = "labels",
    return_elem = TRUE
  )

  na_mask <- is.na(labels)
  if (any(na_mask)) {
    labels <- labels[!na_mask]
    X <- X[, !na_mask, drop = FALSE]
  }

  obs_factor <- factor(labels)
  obs_labels <- as.numeric(obs_factor)
  obs_keys <- levels(obs_factor)

  # Compute gene specificity for each cluster
  if (is.matrix(X)) {
    out <- C_computeFeatureSpecificityDense(X, labels = obs_labels, thread_no = thread_no)
  } else {
    out <- C_computeFeatureSpecificitySparse(X, labels = obs_labels, thread_no = thread_no)
  }

  out <- lapply(out, function(scores) {
    colnames(scores) <- obs_keys
    rownames(scores) <- rownames(obj)
    return(scores)
  })

  if (is_ace && !return_raw) {
    spec_slot_out <- sprintf("%s_feat_spec", map_out_prefix)
    rowMaps(obj)[[spec_slot_out]] <- out[["upper_significance"]]
    rowMapTypes(obj)[[spec_slot_out]] <- "reduction"

    if (return_lower) {
      spec_slot_lower <- sprintf("%s_feat_spec_lower", map_out_prefix)
      rowMaps(obj)[[spec_slot_lower]] <- out[["lower_significance"]]
      rowMapTypes(obj)[[spec_slot_lower]] <- "reduction"
    }

    return(obj)
  }

  return(out)
}


#' Computes feature specificity scores for each archetype.
#' @export
archetypeFeatureSpecificity <- function(
    ace,
    assay_name = "logcounts",
    map_slot = "archetype_footprint",
    map_out_prefix = "archetype",
    thread_no = 0,
    return_raw = FALSE) {
  .validate_ace(ace, allow_se_like = FALSE, error_on_fail = TRUE, return_elem = FALSE)
  X <- .validate_assay(ace = ace, assay_name = assay_name)

  H <- .validate_map(
    ace = ace,
    map_slot = map_slot,
    matrix_type = "dense",
    force_type = TRUE
  )
  H <- Matrix::t(H)

  if (is.matrix(X)) {
    out <- C_archetypeFeatureSpecificityDense(X, H = H, thread_no = thread_no)
  } else {
    out <- C_archetypeFeatureSpecificitySparse(X, H = H, thread_no = thread_no)
  }

  out <- lapply(out, function(scores) {
    colnames(scores) <- paste("A", seq_len(NCOL(scores)), sep = "")
    rownames(scores) <- rownames(ace)
    return(scores)
  })


  if (!return_raw) {
    prof_slot_out <- sprintf("%s_feat_profile", map_out_prefix)
    rowMaps(ace)[[prof_slot_out]] <- out[["archetypes"]]
    rowMapTypes(ace)[[prof_slot_out]] <- "internal"

    spec_slot_out <- sprintf("%s_feat_spec", map_out_prefix)
    rowMaps(ace)[[spec_slot_out]] <- out[["upper_significance"]]
    rowMapTypes(ace)[[spec_slot_out]] <- "reduction"

    return(ace)
  }

  return(out)
}
