#' @export
layoutNetwork <- function(
    obj,
    initial_coordinates = NULL,
    net_slot = "actionet",
    assay_name = "logcounts",
    method = c("umap", "tumap", "largevis"),
    n_components = 2,
    spread = 1.0,
    min_dist = 1.0,
    n_epochs = NULL,
    learning_rate = 1.0,
    repulsion_strength = 1.0,
    negative_sample_rate = 3.0,
    approx_pow = TRUE,
    pcg_rand = TRUE,
    batch = TRUE,
    grain_size = 1,
    seed = 0,
    thread_no = 0,
    verbose = TRUE,
    a = NULL,
    b = NULL,
    opt_method = c("adam", "sgd"),
    alpha = learning_rate,
    beta1 = 0.5,
    beta2 = 0.9,
    eps = 1e-7,
    map_slot_out = NULL,
    return_raw = FALSE) {
    force(alpha)
    method <- tolower(method)
    method <- match.arg(method, several.ok = TRUE)[1]

    opt_method <- tolower(opt_method)
    opt_method <- match.arg(opt_method, several.ok = TRUE)[1]

    is_ace <- .validate_ace(obj, error_on_fail = FALSE, return_elem = FALSE)
    G <- .ace_or_net(
        obj = obj,
        net_slot = net_slot,
        matrix_type = "sparse",
        sparse_type = "CsparseMatrix",
        force_type = TRUE,
        obj_name = "ace"
    )

    if (!is.null(initial_coordinates)) {
        if (is.matrix(initial_coordinates)) {
            initial_coordinates <- .validate_matrix(
                x = initial_coordinates,
                matrix_type = "dense",
                force_type = TRUE,
                return_elem = TRUE
            )
        } else if (is_ace) {
            initial_coordinates <- .validate_map(
                obj,
                map_slot = initial_coordinates,
                matrix_type = "dense",
                force_type = TRUE,
                return_elem = TRUE
            )
        } else {
            if (is_ace) {
                err <- sprintf("'initial_coordinates' must be type 'matrix' or entry in 'colMaps(obj)' for 'obj' type '%s'", class(obj))
            } else {
                err <- sprintf("'initial_coordinates' must be type 'matrix' for 'obj' type '%s'", class(obj))
            }
            stop(err)
        }
    } else {
        if (!is_ace) {
            err <- sprintf("'initial_coordinates' cannot be NULL for 'obj' type '%s'", class(obj))
            stop(err)
        } else {
            msg <- sprintf("Computing initial coordinates from assay '%s'", assay_name)
            message(msg)
            svd.out <- runSVD(
                X = .validate_assay(obj, assay_name = assay_name, return_elem = TRUE),
                k = base::max(3, n_components),
                seed = seed,
                verbose = verbose
            )
            initial_coordinates <- scale(svd.out$v)
        }
    }

    if (NROW(initial_coordinates) != NCOL(obj)) {
        err <- sprintf("'NROW(initial_coordinates)' (%d) does not match 'NCOL(obj)' (%d)", NROW(initial_coordinates), NCOL(obj))
        stop(err)
    }

    if (n_components < 2) {
        err <- sprintf("'n_components' (%d) must be >= 2", n_components)
        stop(err)
    }

    if (NCOL(initial_coordinates) < n_components) {
        err <- sprintf("'NCOL(initial_coordinates)' must be >= 'n_components' (%d)", n_components)
        stop(err)
    }

    if (is.null(a) || is.null(b)) {
        a <- b <- 0
    }

    embedding <- C_layoutNetwork(
        G = G,
        initial_coordinates = initial_coordinates,
        method = method,
        n_components = n_components,
        spread = spread,
        min_dist = min_dist,
        n_epochs = ifelse(is.null(n_epochs), 0, n_epochs),
        learning_rate = learning_rate,
        repulsion_strength = repulsion_strength,
        negative_sample_rate = negative_sample_rate,
        approx_pow = approx_pow,
        pcg_rand = pcg_rand,
        batch = batch,
        grain_size = grain_size,
        seed = seed,
        thread_no = thread_no,
        verbose = verbose,
        a = a,
        b = b,
        opt_method = opt_method,
        alpha = alpha,
        beta1 = beta1,
        beta2 = beta2,
        eps = eps
    )

    if (is_ace && !return_raw) {
        rownames(embedding) <- colnames(obj)
        if (is.null(map_slot_out)) {
            map_slot_out <- sprintf("%s_%dd_%s", method, n_components, net_slot)
        }
        colMaps(obj)[[map_slot_out]] <- embedding
        colMapTypes(obj)[[map_slot_out]] <- "embedding"
        return(obj)
    }
    return(embedding)
}

#' @export
computeNodeColors <- function(
    obj,
    embedding_slot = "umap_3d_actionet",
    color_slot_out = NULL,
    thread_no = 1,
    return_raw = FALSE) {
    is_ace <- .validate_ace(obj, error_on_fail = FALSE, return_elem = FALSE)

    coordinates <- .ace_or_map(
        obj = obj,
        map_slot = embedding_slot,
        matrix_type = "dense",
        force_type = TRUE,
        transpose_map = FALSE,
        return_elem = TRUE
    )

    colors <- C_computeNodeColors(coordinates, thread_no)

    if (is_ace && !return_raw) {
        if (is.null(color_slot_out)) {
            color_slot_out <- sprintf("colors_%s", embedding_slot)
        }
        colMaps(obj)[[color_slot_out]] <- colors
        return(obj)
    }
    return(colors)
}
