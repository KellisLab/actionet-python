// Pybind11 interface for `visualization` module
// Organized by module header in the order imported.

#include "wp_utils.h"
#include "libactionet.hpp"

namespace py = pybind11;

// generate_layout =====================================================================================================

py::array_t<double> layout_network(py::object G, py::array_t<double> initial_coords,
                                    std::string method = "umap", unsigned int n_components = 2,
                                    float spread = 1.0, float min_dist = 1.0,
                                    unsigned int n_epochs = 0, int seed = 0,
                                    int thread_no = 0, bool verbose = true) {
    arma::sp_mat G_sp = scipy_to_arma_sparse(G);
    arma::mat init_mat = numpy_to_arma_mat(initial_coords);

    arma::mat coords = actionet::layoutNetwork(
        G_sp, init_mat, method, n_components, spread, min_dist,
        n_epochs, 1.0f, 1.0f, 5.0f, false, true, true, 1,
        seed, thread_no, verbose, 0.0f, 0.0f, "adam", -1.0f, 0.5f, 0.9f, 1e-7f
    );

    return arma_mat_to_numpy(coords);
}

// =====================================================================================================================

void init_visualization(py::module_ &m) {
    m.def("layout_network", &layout_network, "Layout network (UMAP/t-SNE)",
          py::arg("G"), py::arg("initial_coords"), py::arg("method") = "umap",
          py::arg("n_components") = 2, py::arg("spread") = 1.0, py::arg("min_dist") = 1.0,
          py::arg("n_epochs") = 0, py::arg("seed") = 0,
          py::arg("thread_no") = 0, py::arg("verbose") = true);
}
