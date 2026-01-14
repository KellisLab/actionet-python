// Pybind11 interface for `tools` module
// Organized by module header in the order imported.

#include "wp_utils.h"
#include "libactionet.hpp"

namespace py = pybind11;

// svd =================================================================================================================

py::dict run_svd_sparse(py::object A, int k = 30, int max_it = 0, int seed = 0,
                        int algorithm = 0, bool verbose = true) {
    arma::sp_mat A_sp = scipy_to_arma_sparse(A);
    arma::field<arma::mat> res = actionet::runSVD(A_sp, k, max_it, seed, algorithm, verbose);

    py::dict out;
    out["u"] = arma_mat_to_numpy(res(0));
    out["d"] = arma_mat_to_numpy(res(1));
    out["v"] = arma_mat_to_numpy(res(2));
    return out;
}

// =====================================================================================================================

void init_tools(py::module_ &m) {
    m.def("run_svd_sparse", &run_svd_sparse, "Run SVD (sparse)",
          py::arg("A"), py::arg("k") = 30, py::arg("max_it") = 0, py::arg("seed") = 0,
          py::arg("algorithm") = 0, py::arg("verbose") = true);
}
