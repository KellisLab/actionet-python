#' Run main ACTIONet pipeline
#'
#' @param ace `ACTIONetExperiment` containing an appropriate reduction in 'reduction_slot'.
#' @param k_min Minimum depth of decompositions. (default=2)
#' @param k_max Maximum depth of decompositions. (default=30)
#' @param assay_name Name of assay to be used. (default='logcounts')
#' @param reduction_slot Slot in colMaps(ace) containing reduced kernel. (default='ACTION')
#' @param net_slot_out  Name of slot in colMaps(ace) to store ACTIONet adjacency matrix. (default='ACTIONet')
#' @param min_obs Minimum number of observations required to construct an archetype. (default=2)
#' @param max_it Maximum number of iterations for ACTION algorithm. (default=50)
#' @param spec_th Defines the stringency of pruning nonspecific archetypes.
#' The larger the value, the more archetypes will be filtered out. (default=-3)
#' @param network_metric Distance metric with which to compute cell-to-cell similarity during network construction. Options are 'jsd' (Jensen-Shannon divergence), L2-norm ('l2'), and inner product ('ip'). (default='jsd')
#' @param network_algorithm Algorithm to use for network construction. Options are k-nearest neighbors ('knn') and k*-nearest neighbors ('k*nn'). (default='k*nn')
#' @param network_density Density factor of ACTIONet graph. (default=1)
#' @param mutual_edges_only Whether to enforce edges to be mutually-nearest-neighbors. (default=TRUE)
#' @param layout_method Algorithm for computing plot layout. Options are UMAP ("umap") or t-UMAP ("tumap"). (default="umap")
#' @param layout_epochs Number of epochs for SGD algorithm. (default=250)
#' @param layout_parallel Run layout construction using multiple cores. May result in marginally different outputs across runs due to parallelization-induced randomization. (default=TRUE)
#' @param compute_specificity_parallel Run feature specificity enrichment using multiple cores. Setting this to `TRUE` on large datasets may cause an out of memory crash. (default=FALSE)
#' @param thread_no Number of parallel threads. (default=0)
#' @param seed Seed for random initialization. (default=0)
#'
#' @return An ACTIONetExperiment object
#'
#' @examples
#' ace <- runACTIONet(ace)
#' @export
runACTIONet <- function(ace,
                        k_min = 2,
                        k_max = 30,
                        assay_name = "logcounts",
                        reduction_slot = "action",
                        net_slot_out = "actionet",
                        min_obs = 2,
                        max_it = 50,
                        spec_th = -3,
                        network_metric = "jsd",
                        network_algorithm = "k*nn",
                        network_density = 1,
                        mutual_edges_only = TRUE,
                        layout_method = c("umap", "tumap", "largevis"),
                        layout_epochs = 100,
                        layout_spread = 1.0,
                        layout_min_dist = 1.0,
                        layout_3d = TRUE,
                        layout_parallel = TRUE,
                        compute_specificity_parallel = FALSE,
                        thread_no = 0,
                        seed = 0) {
  ace <- .validate_ace(ace, as_ace = TRUE, allow_se_like = TRUE, return_elem = TRUE)

  layout_method <- tolower(layout_method)
  layout_method <- match.arg(layout_method, several.ok = FALSE)

  .validate_assay(ace, assay_name = assay_name, return_elem = FALSE)
  .validate_map(ace, map_slot = reduction_slot, return_elem = FALSE)

  ace <- runACTION(
    ace,
    k_min = k_min,
    k_max = k_max,
    reduction_slot = reduction_slot,
    prenormalize = TRUE,
    min_obs = min_obs,
    max_it = max_it,
    spec_th = spec_th,
    thread_no = thread_no
  )

  # Build ACTIONet
  ace <- buildNetwork(
    ace,
    algorithm = network_algorithm,
    distance_metric = network_metric,
    density = network_density,
    thread_no = thread_no,
    mutual_edges_only = mutual_edges_only,
    map_slot = "H_stacked",
    net_slot_out = net_slot_out
  )

  ace <- networkCentrality(
    ace,
    label_attr = "assigned_archetype",
    algorithm = "local_coreness",
    thread_no = thread_no,
    net_slot = net_slot_out,
    attr_out = "node_centrality"
  )

  # Smooth archetype footprints
  ace <- networkDiffusion(
    ace,
    scores = colMaps(ace)[["H_merged"]],
    norm_method = "pagerank",
    thread_no = thread_no,
    net_slot = net_slot_out,
    map_slot_out = "archetype_footprint"
  )

  # Use archetypal reduction as initial coordinates for uwot
  # Need to reduce it to 3D coordinate space.
  # red.out <- runSVD(
  #   X = scale(colMaps(ace)[["archetype_footprint"]]),
  #   k = 3,
  #   seed = seed,
  #   verbose = FALSE
  # )
  # initial_coordinates <- scale(red.out$u)

  slot_layout <- sprintf("%s_%s", layout_method, net_slot_out)
  initial_coordinates <- colMaps(ace)[["archetype_footprint"]]

  layout_args <- list(
    method = layout_method,
    n_components = 2,
    min_dist = layout_min_dist,
    spread = layout_spread,
    n_epochs = layout_epochs,
    net_slot = net_slot_out,
    seed = seed,
    thread_no = ifelse(layout_parallel, thread_no, 1)
  )
  layout_args$map_slot_out <- sprintf("%s_%dd_%s", layout_args$method, layout_args$n_components, layout_args$net_slot)

  ace <- do.call(
    layoutNetwork,
    c(
      list(
        obj = ace,
        initial_coordinates = scale(initial_coordinates)
      ),
      layout_args
    )
  )

  if (layout_3d) {
    # Warm-start 3D layout using 2D embedding
    initial_coordinates <- cbind(
      colMaps(ace)[[layout_args$map_slot_out]],
      initial_coordinates[, 3]
    )

    layout_args$n_components <- 3
    layout_args$n_epochs <- layout_epochs / 2
    layout_args$map_slot_out <- sprintf("%s_%dd_%s", layout_args$method, layout_args$n_components, layout_args$net_slot)

    ace <- do.call(
      layoutNetwork,
      c(
        list(
          obj = ace,
          initial_coordinates = scale(initial_coordinates)
        ),
        layout_args
      )
    )

    ace <- computeNodeColors(
      ace,
      embedding_slot = layout_args$map_slot_out,
      color_slot_out = sprintf("colors_%s", layout_args$net_slot),
      thread_no = thread_no
    )
  }

  # Compute gene specificity for each archetype
  ace <- archetypeFeatureSpecificity(
    ace = ace,
    assay_name = assay_name,
    map_slot = "archetype_footprint",
    thread_no = ifelse(compute_specificity_parallel, thread_no, 1),
    return_raw = FALSE
  )

  return(ace)
}

#' @export
runACTION <- function(
    ace,
    k_min = 2,
    k_max = 30,
    reduction_slot = "action",
    prenormalize = TRUE,
    min_obs = 2,
    max_it = 50,
    tol = 1e-100,
    spec_th = -3,
    thread_no = 0,
    merged_suffix = "merged",
    archetype_slot_out = "assigned_archetype") {
  ace <- .validate_ace(
    ace,
    as_ace = TRUE,
    allow_se_like = TRUE,
    return_elem = TRUE
  )

  S_r <- .validate_map(
    ace = ace,
    map_slot = reduction_slot,
    matrix_type = "dense",
    force_type = TRUE,
    return_elem = TRUE
  )

  if (prenormalize) {
    S_r <- normalize.matrix(
      S_r,
      dim = 1, # obs are stored rows in colMaps()
      scale_param = NULL,
      trans_func = NULL
    )
  }

  out <- C_runACTION(
    S_r = Matrix::t(S_r),
    k_min = k_min,
    k_max = k_max,
    max_it = max_it,
    tol = tol,
    spec_th = spec_th,
    min_obs = min_obs,
    thread_no = thread_no
  )

  colMaps(ace)[["H_stacked"]] <- Matrix::t(as(out$H_stacked, "sparseMatrix"))
  colMapTypes(ace)[["H_stacked"]] <- "internal"

  colMaps(ace)[["C_stacked"]] <- as(out$C_stacked, "sparseMatrix")
  colMapTypes(ace)[["C_stacked"]] <- "internal"

  colMaps(ace)[[sprintf("H_%s", merged_suffix)]] <- as(Matrix::t(out$H_merged), "sparseMatrix")
  colMapTypes(ace)[[sprintf("H_%s", merged_suffix)]] <- "internal"

  colMaps(ace)[[sprintf("C_%s", merged_suffix)]] <- as(out$C_merged, "sparseMatrix")
  colMapTypes(ace)[[sprintf("C_%s", merged_suffix)]] <- "internal"

  colData(ace)[[archetype_slot_out]] <- c(out$assigned_archetype)
  return(ace)
}
