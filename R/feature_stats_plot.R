#' @export
plotFeatureDist <- function(
    obj,
    labels,
    features = "all",
    labels_use = NULL,
    nonzero = FALSE,
    metric = c("counts", "fraction", "percent", "ratio"),
    log_trans = c("none", "log", "log2", "log10"),
    features_use = NULL,
    assay_name = "counts",
    palette = NULL,
    x_label = NULL,
    y_label = NULL,
    plot_title = NULL,
    to_return = c("plot", "data")) {
  metric <- match.arg(metric)
  log_trans <- match.arg(log_trans)
  to_return <- match.arg(to_return)

  if (log_trans != "none" && metric != "counts") {
    err <- sprintf(" log_trans (%s) only valid for metric 'counts'", log_trans)
    stop(err)
  }

  feature_stats <- getFeatureAbundance(
    obj,
    features = features,
    nonzero = nonzero,
    labels = labels,
    labels_use = labels_use,
    features_use = features_use,
    metric = metric,
    assay_name = assay_name
  )

  if (is.list(feature_stats)) {
    df <- lapply(seq_len(length(feature_stats)), function(l) {
      data.frame(label = names(feature_stats)[l], stat = feature_stats[[l]])
    })
    df <- do.call(rbind, df)
    df$label <- factor(df$label, levels = names(feature_stats))
  } else {
    df <- data.frame(label = "all", stat = feature_stats)
  }

  if (log_trans != "none") {
    df$stat[df$stat == 0] <- 1
    df$stat <- get(log_trans)(df$stat)
    y_label <- ifelse(is.null(y_label), ggplot2::ylab(sprintf("%s(%s)", log_trans, metric)), y_label)
  } else {
    y_label <- ifelse(is.null(y_label), ggplot2::ylab(sprintf("%s", metric)), y_label)
  }

  if (to_return == "data") {
    return(df)
  }

  df$x <- df$fill <- df$label
  df$y <- df$stat
  p <- .plot_gg_violin(
    df,
    x_label = x_label,
    y_label = y_label,
    plot_title = plot_title,
    palette = palette
  )
  return(p)
}

#' @export
plotMitoDist <- function(
    obj,
    labels,
    id_type = c("gene_name", "ensembl_id"),
    species = c("hsapiens", "mmusculus", "human", "mouse"),
    protein_coding = FALSE,
    labels_use = NULL,
    log_trans = c("none", "log", "log2", "log10"),
    metric = c("counts", "fraction", "percent", "ratio"),
    features_use = NULL,
    assay_name = "counts",
    palette = NULL,
    x_label = NULL,
    y_label = NULL,
    plot_title = NULL,
    to_return = c("plot", "data")) {
  metric <- match.arg(metric)
  if (is.null(plot_title)) {
    plot_title <- sprintf("Mitochondrial features (%s)", metric)
  }

  feats_mito <- .get_mito_feats(id_type = id_type, species = species, protein_coding = protein_coding)
  out <- plotFeatureDist(
    obj,
    labels = labels,
    features = feats_mito,
    labels_use = labels_use,
    nonzero = FALSE,
    log_trans = log_trans,
    metric = metric,
    features_use = features_use,
    assay_name = assay_name,
    palette = palette,
    x_label = x_label,
    y_label = y_label,
    plot_title = plot_title,
    to_return = to_return
  )

  return(out)
}

.plot_gg_violin <- function(df, x_label = NULL, y_label = NULL, plot_title = NULL, palette = NULL) {
  require(ggplot2)
  if (is.null(x_label)) {
    x_label <- element_blank()
  }
  if (is.null(y_label)) {
    y_label <- element_blank()
  }
  if (is.null(palette)) {
    palette <- CPal_default[seq.int(unique(df$x))]
  }

  all_labels <- sort(unique(df$x))

  p <- ggplot(df, aes(x = x, y = y, fill = fill)) +
    geom_violin(draw_quantiles = c(0.25, 0.5, 0.75), scale = "width") +
    scale_fill_manual(values = palette) +
    scale_x_discrete(
      labels = all_labels,
      position = "bottom"
    ) +
    labs(x = x_label, y = y_label) +
    theme(
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
      panel.border = element_rect(color = "black", fill = NA),
      legend.position = "none"
    )

  if (!is.null(plot_title)) {
    p <- p + ggtitle(plot_title) + theme(plot.title = element_text(hjust = 0.5))
  }

  return(p)
}
