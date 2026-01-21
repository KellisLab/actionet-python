.SYS_THREADS_DEF <- parallel::detectCores()

.get_num_threads <- function(
    max_threads = 0,
    thread_no = 0) {
    max_threads <- ifelse(max_threads > 0, base::min(max_threads, .SYS_THREADS_DEF), .SYS_THREADS_DEF)

    if (thread_no <= 0) {
        threads_use <- max_threads
    } else if (thread_no > 1) {
        threads_use <- base::min(thread_no, max_threads)
    } else {
        threads_use <- 1
    }
    return(threads_use)
}
