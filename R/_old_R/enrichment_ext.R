#' Performs geneset enrichment analysis on archetypes
#'
#' @param ace ACTIONetExperiment (ACE) output object
#' @param associations Either a genes x pathways membership matrix, or a set of genesets
#' @param L Maximum length of the top-ranked genes to consider
#'
#' @return Matrix pathway x cell type/states
#'
#' @examples
#' data("gProfilerDB_human")
#' associations <- gProfilerDB_human$SYMBOL$WP
#' Geneset.enrichments <- assess.geneset.enrichment.from.archetypes(ace, associations)
#' @export
assess.geneset.enrichment.from.archetypes <- function(ace,
                                                      associations,
                                                      min.counts = 0,
                                                      specificity.slot = "arch_feat_spec") {
  scores <- rowMaps(ace)[[specificity.slot]]
  if (max(scores) > 100) {
    scores <- log1p(scores)
  }

  if (is.list(associations)) {
    associations <- sapply(associations, function(gs) {
      as.numeric(rownames(scores) %in%
        gs)
    })
    rownames(associations) <- rownames(scores)
  }
  common.features <- intersect(rownames(associations), rownames(scores))

  rows <- match(common.features, rownames(associations))
  associations <- as(associations[rows, ], "dMatrix")
  scores <- scores[common.features, ]

  enrichment.out <- assess_enrichment(scores, associations)

  rownames(enrichment.out$logPvals) <- colnames(associations)
  rownames(enrichment.out$thresholds) <- colnames(associations)
  enrichment.out$scores <- scores

  return(enrichment.out)
}
