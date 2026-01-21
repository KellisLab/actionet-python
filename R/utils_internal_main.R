.run.layoutNetwork <- function(ace,
                               initial_coordinates,
                               algorithm = "umap",
                               min_dist = 1.0,
                               spread = 1.0,
                               gamma = 1.0,
                               n_epochs = 250,
                               learning_rate = 1,
                               net_slot = "actionet",
                               seed = 0,
                               thread_no = 0,
                               return_raw = FALSE) {
  algorithm <- match.arg(tolower(algorithm), c(
    "umap", "tumap",
    "largevis", "pacmap"
  ))

  .validate_ace(ace, allow_null = FALSE, return_elem = FALSE)

  G <- .validate_net(
    ace,
    net_slot = net_slot,
    matrix_type = "sparse",
    force_type = TRUE,
  )

  vis.out <- layoutNetwork(
    G = G,
    initial_position = initial_coordinates,
    method = algorithm,
    min_dist = min_dist,
    spread = spread,
    gamma = gamma,
    n_epochs = n_epochs,
    learning_rate = learning_rate,
    seed = seed,
    thread_no = thread_no
  )

  if (return_raw == TRUE) {
    return(vis.out)
  } else {
    colMaps(ace)[["ACTIONred"]] <- initial_coordinates[, 1:3]
    colMapTypes(ace)[["ACTIONred"]] <- "embedding"

    X <- vis.out$coordinates
    colnames(X) <- c("x", "y")
    rownames(X) <- colnames(ace)
    colMaps(ace)$ACTIONet2D <- X
    colMapTypes(ace)[["umap_2d_actionet"]] <- "embedding"

    X <- vis.out$coordinates_3D
    colnames(X) <- c("x", "y", "z")
    rownames(X) <- colnames(ace)
    colMaps(ace)$ACTIONet3D <- X
    colMapTypes(ace)[["ACTIONet3D"]] <- "embedding"

    X <- vis.out$colors
    colnames(X) <- c("r", "g", "b")
    rownames(X) <- colnames(ace)
    colMaps(ace)$colors_actionet <- X
    colMapTypes(ace)[["colors_actionet"]] <- "embedding"

    return(ace)
  }
}


#' Prune nonspecific and/or unreliable archetypes
.run.collectArchetypes <- function(ace,
                                   C_trace,
                                   H_trace,
                                   specificity_th = -3,
                                   min_cells_per_arch = 2) {
  .validate_ace(ace, allow_null = FALSE, return_elem = FALSE)

  pruning.out <- .collectArchetypes(
    C_trace = C_trace,
    H_trace = H_trace,
    specificity_th = specificity_th,
    min_cells_per_arch = min_cells_per_arch
  )

  colMaps(ace)[["H_stacked"]] <- Matrix::t(as(pruning.out$H_stacked, "sparseMatrix"))
  colMapTypes(ace)[["H_stacked"]] <- "internal"

  colMaps(ace)[["C_stacked"]] <- as(pruning.out$C_stacked, "sparseMatrix")
  colMapTypes(ace)[["C_stacked"]] <- "internal"

  return(ace)
}


#' Identiy equivalent classes of archetypes and group them together
.run.mergeArchetypes <- function(ace,
                                 reduction_slot = "action",
                                 C_stacked_slot = "C_stacked",
                                 H_stacked_slot = "H_stacked",
                                 normalization = 0,
                                 merged_suffix = "merged",
                                 footprint_slot_name = "assigned_archetype",
                                 thread_no = 0,
                                 return_raw = FALSE) {
  .validate_ace(ace, allow_null = FALSE, return_elem = FALSE)

  S_r <- .validate_map(
    ace = ace,
    map_slot = reduction_slot,
    matrix_type = "dense",
    force_type = TRUE
  )
  S_r <- Matrix::t(S_r)

  C_stacked <- .validate_map(
    ace = ace,
    map_slot = C_stacked_slot,
    matrix_type = "dense",
    force_type = TRUE
  )

  H_stacked <- .validate_map(
    ace = ace,
    map_slot = H_stacked_slot,
    matrix_type = "dense",
    force_type = TRUE
  )
  H_stacked <- Matrix::t(H_stacked)

  unification.out <- .mergeArchetypes(
    S_r = S_r,
    C_stacked = C_stacked,
    H_stacked = H_stacked,
    normalization = normalization,
    thread_no = thread_no
  )

  if (return_raw == TRUE) {
    return(unification.out)
  } else {
    Ht_merged <- as(Matrix::t(unification.out$H_merged), "sparseMatrix")
    colMaps(ace)[[sprintf("H_%s", merged_suffix)]] <- Ht_merged
    colMapTypes(ace)[[sprintf("H_%s", merged_suffix)]] <- "internal"

    colMaps(ace)[[sprintf("C_%s", merged_suffix)]] <- as(unification.out$C_merged, "sparseMatrix")
    colMapTypes(ace)[[sprintf("C_%s", merged_suffix)]] <- "internal"

    colData(ace)[[footprint_slot_name]] <- c(unification.out$assigned_archetype)

    return(ace)
  }
}

#' @export
smoothKernel <- function(
    ace,
    norm_method = "pagerank",
    alpha = 0.85,
    max_it = 5,
    reduction_slot = "action",
    net_slot = "actionet",
    thread_no = 0,
    return_raw = FALSE) {
  .validate_ace(ace, allow_se_like = FALSE, return_elem = FALSE, error_on_fail = TRUE)

  vars <- list(
    V = ACTIONetExperiment::rowMaps(ace)[[sprintf("%s_V", reduction_slot)]],
    A = ACTIONetExperiment::rowMaps(ace)[[sprintf("%s_A", reduction_slot)]],
    B = ACTIONetExperiment::colMaps(ace)[[sprintf("%s_B", reduction_slot)]],
    sigma = S4Vectors::metadata(ace)[[sprintf("%s_sigma", reduction_slot)]]
  )

  if (any(sapply(vars, is.null))) {
    nullvars <- sprintf("%s_%s", reduction_slot, names(vars)[which(sapply(vars, is.null))])
    err <- sprintf("'%s' missing from 'ace'. Did you run 'reduceKernel()'?", paste(nullvars, collapse = ","))
    stop(err)
  }

  S_r <- .validate_map(
    ace,
    map_slot = reduction_slot,
    matrix_type = "dense",
    force_type = TRUE,
  )

  G <- .validate_net(
    ace,
    net_slot = net_slot,
    matrix_type = "sparse",
    force_type = TRUE,
  )

  V <- vars$V
  A <- vars$A
  B <- vars$B
  sigma <- vars$sigma

  U <- as.matrix(S_r %*% Matrix::Diagonal(length(sigma), 1.0 / sigma))
  SVD.out <- C_perturbedSVD(V, sigma, U, -A, B)
  V.smooth <- networkDiffusion(
    obj = G,
    scores = SVD.out$v,
    norm_method = norm_method,
    alpha = alpha,
    thread_no = thread_no,
    max_it = max_it
  )

  H <- V.smooth %*% diag(SVD.out$d)

  if (return_raw == TRUE) {
    out <- list(U = U, SVD.out = SVD.out, V.smooth = V.smooth, H = H)
    return(out)
  } else {
    W <- SVD.out$u
    rownames(W) <- rownames(ace)
    smooth_red_name <- sprintf("%s_smooth", reduction_slot)
    smooth_U_name <- sprintf("%s_U", reduction_slot)
    rowMaps(ace)[[smooth_U_name]] <- W
    colMaps(ace)[[smooth_red_name]] <- H
    rowMapTypes(ace)[[smooth_U_name]] <- colMapTypes(ace)[[smooth_red_name]] <- "internal"
    return(ace)
  }
}
