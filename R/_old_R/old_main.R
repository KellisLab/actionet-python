#' Reconstructs the ACTIONet graph with the new parameters (uses prior decomposition)
#'
#' @param ace ACTIONetExperiment object.
#' @param network_density Density factor of ACTIONet graph. (default=1)
#' @param network_metric Distance metric with which to compute cell-to-cell similarity during network construction. Options are 'jsd' (Jensen-Shannon divergence), L2-norm ('l2'), and inner product ('ip'). (default='jsd')
#' @param algorithm Algorithm to use for network construction. Options are k-nearest neighbors ('knn') and k*-nearest neighbors ('k*nn'). (default='k*nn')
#' @param mutual_edges_only Whether to enforce edges to be mutually-nearest-neighbors. (default=TRUE)
#' @param layout_compactness A value between 0-100, indicating the compactness of ACTIONet layout (default=50).
#' @param layout_epochs Number of epochs for SGD algorithm. (default=250)
#' @param layout_algorithm Algorithm for computing plot layout. Options are UMAP ("umap") or t-UMAP ("tumap"). (default="umap")
#' @param layout_parallel Run layout construction using multiple cores. May result in marginally different outputs across runs due to parallelization-induced randomization. (default=TRUE)
#' @param thread_no Number of parallel threads. (default=0)
#' @param reduction_slot Slot in colMaps(ace) containing reduced kernel. (default='ACTION')
#' @param net_slot_out Name of slot in colMaps(ace) to store ACTIONet adjacency matrix. (default='ACTIONet')
#' @param seed Seed for random initialization. (default=0)
#'
#' @return ace Updated ace object
#'
#' @examples
#' plot.ACTIONet(ace)
#' ace.updated <- rebuildACTIONet(ace, network_density = 0.1)
#' plot.ACTIONet(ace.updated)
#' @export
rebuildACTIONet <- function(ace,
                            network_density = 1,
                            network_metric = "jsd",
                            algorithm = "k*nn",
                            mutual_edges_only = TRUE,
                            layout_spread = 1.0,
                            layout_min_dist = 1.0,
                            layout_gamma = 1.0,
                            layout_epochs = 100,
                            layout_algorithm = c("umap", "tumap"),
                            layout_parallel = TRUE,
                            thread_no = 0,
                            reduction_slot = "action",
                            net_slot_out = "actionet",
                            H_stacked_slot = "H_stacked",
                            reduction_normalization = 1,
                            seed = 0) {
  set.seed(seed)
  layout_algorithm <- tolower(layout_algorithm)
  layout_algorithm <- match.arg(layout_algorithm, several.ok = FALSE)

  .validate_ace(ace, return_elem = FALSE)
  .validate_map(ace, map_slot = reduction_slot, return_elem = FALSE)

  if (!(sprintf("%s_normalized", reduction_slot) %in% names(colMaps(ace)))) {
    ace <- normalize.reduction(ace, reduction_slot = reduction_slot, reduction_normalization = reduction_normalization)
  }
  reduction_slot <- sprintf("%s_normalized", reduction_slot)

  # re-Build ACTIONet
  H_stacked <- .validate_map(
    ace = ace,
    map_slot = H_stacked_slot,
    matrix_type = "dense",
    force_type = TRUE
  )
  H_stacked <- Matrix::t(H_stacked)

  G <- buildNetwork(
    H = H_stacked,
    algorithm = algorithm,
    distance_metric = network_metric,
    density = network_density,
    thread_no = thread_no,
    mutual_edges_only = mutual_edges_only
  )
  colNets(ace)[[net_slot_out]] <- G

  # Layout ACTIONet
  ace <- rerunLayout(
    ace = ace,
    algorithm = layout_algorithm,
    n_epochs = layout_epochs,
    spread = layout_spread,
    min_dist = layout_min_dist,
    gamma = layout_gamma,
    net_slot = net_slot_out,
    thread_no = ifelse(layout_parallel, thread_no, 1),
    seed = seed
  )

  return(ace)
}


#' Rerun layout on the ACTIONet graph with new parameters
#'
#' @param ace ACTIONetExperiment object.
#' @param layout_compactness A value between 0-100, indicating the compactness of ACTIONet layout (default=50).
#' @param layout_epochs Number of epochs for SGD algorithm (default=250).
#' @param layout_algorithm Algorithm for computing plot layout. Options are UMAP ("umap") or t-UMAP ("tumap"). (default="umap")
#' @param network_density Density factor of ACTIONet graph (default=1).
#' @param mutual_edges_only Whether to enforce edges to be mutually-nearest-neighbors (default=TRUE).
#' @param thread_no Number of parallel threads (default=0).
#' @param reduction_slot Slot in colMaps(ace) containing reduced kernel (default='ACTION').
#' @param net_slot Slot in colMaps(ace) containing ACTIONet adjacency matrix (default='ACTIONet').
#' @param seed Seed for random initialization (default=0).
#'
#' @return ace Updated ace object
#'
#' @examples
#' plot.ACTIONet(ace)
#' ace.updated <- rerunLayout(ace, compactness = 20)
#' plot.ACTIONet(ace.updated)
#' @export
rerunLayout <- function(ace,
                        initial_coordinates = NULL,
                        algorithm = c("umap", "tumap"),
                        min_dist = 1.0,
                        spread = 1.0,
                        gamma = 1.0,
                        n_epochs = 100,
                        learning_rate = 1.0,
                        net_slot = "actionet",
                        seed = 0,
                        thread_no = 0) {
  algorithm <- tolower(algorithm)
  algorithm <- match.arg(algorithm, several.ok = FALSE)

  if (is.null(initial_coordinates)) {
    red.out <- runSVD_full(scale(ace$archetype_footprint), k = 3, algorithm = 0, max_it = 1000, verbose = 0, seed = 0)
    initial_coordinates <- scale(red.out$u[, 1:3])
  }

  ace <- .run.layoutNetwork(
    ace = ace,
    initial_coordinates = initial_coordinates,
    algorithm = algorithm,
    min_dist = min_dist,
    spread = spread,
    gamma = gamma,
    n_epochs = n_epochs,
    learning_rate = learning_rate,
    net_slot = net_slot,
    thread_no = thread_no,
    seed = seed
  )

  return(ace)
}


#' @export
rerun.archetype.unification <- function(ace,
                                        backbone_density = 0.5,
                                        resolution = 1.0,
                                        min_cluster_size = 3,
                                        footprint_alpha = 0.85,
                                        compute_specificity_parallel = FALSE,
                                        assay_name = "logcounts",
                                        reduction_slot = "action",
                                        C_stacked_slot = "C_stacked",
                                        H_stacked_slot = "H_stacked",
                                        net_slot = "actionet",
                                        merged_suffix = "merged",
                                        reduction_normalization = 1,
                                        thread_no = 0) {
  .validate_ace(ace, return_elem = FALSE)

  G <- .validate_net(
    ace = ace,
    net_slot = net_slot,
    matrix_type = "sparse",
    force_type = FALSE
  )

  if (!(sprintf("%s_normalized", reduction_slot) %in% names(colMaps(ace)))) {
    ace <- normalize.reduction(ace, reduction_slot = reduction_slot, reduction_normalization = reduction_normalization)
  }
  reduction_slot <- sprintf("%s_normalized", reduction_slot)

  ace <- .run.mergeArchetypes(
    ace = ace,
    reduction_slot = reduction_slot,
    C_stacked_slot = C_stacked_slot,
    H_stacked_slot = H_stacked_slot,
    normalization = 0,
    merged_suffix = merged_suffix,
    thread_no = thread_no
  )

  colData(ace)[["node_centrality"]] <- networkCentrality(
    obj = G,
    label_attr = colData(ace)[["assigned_archetype"]],
    algorithm = "local_coreness"
  )

  archetype_footprint <- networkDiffusion(
    obj = G,
    scores = colMaps(ace)[[sprintf("H_%s", merged_suffix)]],
    norm_method = "pagerank",
    alpha = footprint_alpha,
    thread_no = thread_no,
    max_it = 5,
    tol = 1e-8,
    net_slot = NULL
  )
  colMaps(ace)$archetype_footprint <- archetype_footprint

  # Compute gene specificity for each archetype
  ace <- archetypeFeatureSpecificity(
    ace = ace,
    assay_name = assay_name,
    map_slot = "archetype_footprint",
    thread_no = ifelse(compute_specificity_parallel, thread_no, 1),
    return_raw = FALSE
  )

  ace <- constructBackbone(ace, backbone_density = backbone_density)

  return(ace)
}
