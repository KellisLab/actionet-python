#' A wrapper function For HDBSCAN algorithm applied to an ACE object
#'
#' @param ace Input results to be clustered
#' minPoints, minClusterSize HDBSCAN parameters (default = 30,30)
#' archetype.slot Slot of archeypte to use for clustering (default='H_merged');
#'
#' @return clusters
#'
#' @examples
#' clusters <- HDBSCAN.clustering(ace)
#' plot.ACTIONet(ace, clusters)
#' @export
HDBSCAN.clustering <- function(ace,
                               minPoints = 30,
                               minClusterSize = 30,
                               archetype.slot = "H_merged") {
  X <- as.matrix(colMaps(ace)[[archetype.slot]])

  out_list <- run_HDBSCAN(
    X = X,
    minPoints = minPoints,
    minClusterSize = minClusterSize
  )

  return(out_list)
}
