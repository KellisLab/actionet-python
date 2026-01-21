#' @export
buildNetwork <- function(
    obj,
    algorithm = "k*nn",
    distance_metric = "jsd",
    density = 1.0,
    thread_no = 0,
    mutual_edges_only = TRUE,
    M = 16,
    ef_construction = 200, # metric "knn" only
    ef = 200, # metric "knn" only
    k = 10, # metric "knn" only
    map_slot = "H_stacked",
    net_slot_out = "actionet",
    return_raw = FALSE) {
  is_ace <- .validate_ace(obj, error_on_fail = FALSE, return_elem = FALSE)

  H <- .ace_or_map(
    obj = obj,
    map_slot = map_slot,
    matrix_type = "dense",
    force_type = TRUE,
    transpose_map = TRUE,
    return_elem = TRUE
  )

  G <- C_buildNetwork(
    H = H,
    algorithm = algorithm,
    distance_metric = distance_metric,
    density = density,
    thread_no = thread_no,
    M = M,
    ef_construction = ef_construction,
    ef = ef,
    mutual_edges_only = mutual_edges_only,
    k = k
  )

  if (is_ace && !return_raw) {
    colNets(obj)[[net_slot_out]] <- G
    return(obj)
  }
  return(G)
}

#' @export
networkDiffusion <- function(
    obj,
    scores, ## `scores` must be castable to dense matrix.
    norm_method = c("pagerank", "pagerank_sym"),
    alpha = 0.85,
    thread_no = 0,
    approx = TRUE,
    max_it = 5,
    tol = 1e-8,
    net_slot = "actionet",
    map_slot_out = NULL,
    return_raw = FALSE) {
  norm_method <- tolower(norm_method)
  norm_method <- match.arg(norm_method, several.ok = TRUE)[1]

  is_ace <- .validate_ace(obj, error_on_fail = FALSE, return_elem = FALSE)

  if (!is.matrix(scores)) {
    scores <- Matrix::as.matrix(scores)
  }

  if (NROW(scores) != NCOL(obj)) {
    err <- sprintf("`length(scores)` must equal `NCOL(obj)`.\n")
    stop(err)
  }

  G <- .ace_or_net(
    obj = obj,
    net_slot = net_slot,
    matrix_type = "sparse",
    force_type = TRUE,
    obj_name = "obj"
  )

  if (alpha >= 1) {
    stop("`alpha` => 1")
  } else if (alpha < 0) {
    stop("`alpha` < 0")
  }
  X <- C_computeNetworkDiffusion(
    G = G,
    X0 = scores,
    alpha = alpha,
    max_it = max_it,
    thread_no = thread_no,
    approx = approx,
    norm_method = ifelse(norm_method == "pagerank_sym", 2, 0),
    tol = tol
  )

  if (is_ace && !return_raw) {
    if (is.null(map_slot_out)) {
      map_slot_out <- sprintf("%s_%s", norm_method, net_slot)
    }
    colMaps(obj)[[map_slot_out]] <- X
    return(obj)
  }

  return(X)
}


#' @export
networkCentrality <- function(
    obj,
    label_attr = NULL,
    algorithm = c("coreness", "pagerank", "local_coreness", "local_pagerank"),
    alpha = 0.9,
    max_it = 5,
    tol = 1e-8,
    thread_no = 0,
    net_slot = "actionet",
    attr_out = NULL,
    return_raw = FALSE) {
  algorithm <- tolower(algorithm)
  algorithm <- match.arg(algorithm, several.ok = TRUE)[1]

  is_ace <- .validate_ace(obj, error_on_fail = FALSE, return_elem = FALSE)

  G <- .ace_or_net(
    obj = obj,
    net_slot = net_slot,
    matrix_type = "sparse",
    sparse_type = "CsparseMatrix",
    force_type = TRUE,
    obj_name = "obj"
  )

  if (algorithm %in% c("pagerank", "local_pagerank")) {
    if (is.null(label_attr)) {
      err <- sprintf("'label_attr' cannot be 'NULL' if 'algorithm=%s'.\n", algorithm)
      stop(err)
    }

    if (alpha >= 1) {
      alpha <- 0.99
      warning("alpha=0.99")
    } else if (alpha < 0) {
      alpha <- 0
      warning("alpha=0")
    }
  }

  if (!is.null(label_attr)) {
    label_attr <- .validate_attr(obj, attr = label_attr, obj_name = "obj", attr_name = "label_attr", return_elem = TRUE)
    assignments <- as.numeric(factor(label_attr))
  }

  if (algorithm == "coreness") {
    centrality <- C_computeCoreness(G)
  } else if (algorithm == "pagerank") {
    centrality <- networkDiffusion(
      obj = G,
      scores = rep(1 / NCOL(G), NCOL(G)),
      norm_method = "pagerank",
      alpha = alpha,
      thread_no = thread_no,
      max_it = max_it,
      tol = tol,
      net_slot = NULL
    )
  } else if (algorithm == "local_coreness") {
    centrality <- C_computeArchetypeCentrality(G, assignments)
  } else if (algorithm == "local_pagerank") {
    design.mat <- stats::model.matrix(~ 0 + as.factor(assignments))
    design.mat <- scale(design.mat, center = FALSE, scale = Matrix::colSums(design.mat))
    scores <- networkDiffusion(
      obj = G,
      scores = design.mat,
      norm_method = "pagerank",
      alpha = alpha,
      thread_no = thread_no,
      max_it = max_it,
      tol = tol,
      net_slot = NULL
    )
    scores <- apply(scores, 2, function(x) x / max(x))
    centrality <- apply(scores, 1, max)
  }
  centrality <- c(centrality)

  if (is_ace && !return_raw) {
    if (is.null(attr_out)) {
      attr_out <- sprintf("%s_%s", algorithm, net_slot)
    }
    colData(obj)[[attr_out]] <- centrality
    return(obj)
  }

  return(centrality)
}


#' @export
propagateLabels <- function(
    obj,
    labels,
    which_fixed = NULL,
    algorithm = c("LPA"),
    lambda = 0,
    iters = 3,
    sig_th = 3,
    net_slot = "actionet",
    thread_no = 0) {
  algorithm <- match.arg(algorithm, several.ok = TRUE)[1]

  G <- .ace_or_net(
    obj = obj,
    net_slot = net_slot,
    matrix_type = "sparse",
    force_type = TRUE,
    obj_name = "obj"
  )

  lf <- .validate_vector_attr(
    obj,
    attr = labels,
    return_type = "data",
    attr_name = "labels",
    return_elem = TRUE
  )
  lf <- factor(lf)
  label_num <- as.numeric(lf)
  keys <- levels(lf)
  label_num[is.na(label_num)] <- -1

  if (algorithm == "LPA") {
    new_labels <- C_runLPA(
      G = G,
      labels = label_num,
      lambda = lambda,
      iters = iters,
      sig_threshold = sig_th,
      fixed_labels_ = which_fixed,
      thread_no = thread_no
    )
    new_labels[new_labels == -1] <- NA
    new_labels <- keys[new_labels]
  } else {
    stop("Invalid algorithm")
  }

  return(new_labels)
}


#' @export
correct.cell.labels <- function(
    ace,
    label_attr,
    algorithm = "LPA",
    iters = 3,
    lambda = 0,
    sig_th = 3,
    net_slot = "actionet",
    thread_no = 0) {
  initial_labels <- ACTIONetExperiment::get.data.or.split(ace, attr = label_attr, to_return = "data")
  labels <- propagateLabels(
    obj = ace,
    label_attr = initial_labels,
    fixed_samples = NULL,
    algorithm = algorithm,
    lambda = lambda,
    iters = iters,
    sig_th = sig_th,
    net_slot = net_slot,
    thread_no = thread_no
  )

  return(labels)
}

#' @export
infer.missing.cell.labels <- function(
    ace,
    label_attr,
    algorithm = "LPA",
    iters = 3,
    lambda = 0,
    sig_th = 3,
    net_slot = "actionet",
    thread_no = 0) {
  initial_labels <- ACTIONetExperiment::get.data.or.split(ace, attr = label_attr, to_return = "data")
  fixed_samples <- which(!is.na(initial_labels))

  labels <- propagateLabels(
    obj = ace,
    label_attr = initial_labels,
    fixed_samples = fixed_samples,
    algorithm = algorithm,
    lambda = lambda,
    iters = iters,
    sig_th = sig_th,
    net_slot = net_slot,
    thread_no = thread_no
  )

  return(labels)
}


networkAutocorrelation <- function(
    obj,
    scores = NULL,
    algorithm = c("geary", "moran", "categorical"),
    score_normalization_method = 1L,
    perm_no = 0,
    thread_no = 0L,
    net_slot = "actionet") {
  algorithm <- tolower(algorithm)
  algorithm <- match.arg(algorithm, several.ok = TRUE)[1]

  G <- .ace_or_net(
    obj = obj,
    net_slot = net_slot,
    obj_name = "obj"
  )

  if (!ACTIONetExperiment::is.sparseMatrix(G)) {
    G <- as(G, "sparseMatrix")
  }

  if (algorithm == "geary") {
    out <- autocorrelation_Geary(G = G, scores = scores, normalization_method = score_normalization_method, perm_no = perm_no, thread_no = thread_no)
  } else if (algorithm == "moran") {
    out <- autocorrelation_Moran(G = G, scores = scores, normalization_method = score_normalization_method, perm_no = perm_no, thread_no = thread_no)
  } else if (algorithm == "categorical") {
    if (is.character(scores)) {
      label_type <- "char"
      annotations.factor <- factor(scores)
      annotations <- as.numeric(annotations.factor)
    } else if (is.factor(scores)) {
      label_type <- "factor"
      annotations.factor <- scores
      annotations <- as.numeric(annotations.factor)
    } else {
      annotations <- scores
    }
    out <- assess.categorical.autocorrelation(A = G, labels = annotations, perm.no = perm_no)
  }

  return(out)
}
