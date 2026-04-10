// Pybind11 interface for `visualization` module
// Organized by module header in the order imported.

#include "wp_utils.h"
#include "libactionet.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {
std::vector<float> optional_float_vector(py::object values, const char* name) {
    if (values.is_none()) {
        return {};
    }

    py::array_t<double, py::array::forcecast> arr =
        values.cast<py::array_t<double, py::array::forcecast>>();
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error(std::string("'") + name + "' must be a 1D array");
    }

    const auto* ptr = static_cast<const double*>(buf.ptr);
    const size_t size = static_cast<size_t>(buf.shape[0]);
    std::vector<float> out(size);
    for (size_t i = 0; i < size; ++i) {
        out[i] = static_cast<float>(ptr[i]);
    }
    return out;
}
} // namespace

// generate_layout =====================================================================================================

py::array_t<double> layout_network(py::object G, py::array_t<double> initial_coords,
                                    std::string method = "umap", unsigned int n_components = 2,
                                    float spread = 1.0, float min_dist = 1.0,
                                    unsigned int n_epochs = 0, int seed = 0,
                                    int thread_no = 0, bool verbose = true,
                                    float learning_rate = 1.0f, float repulsion_strength = 1.0f,
                                    float negative_sample_rate = 5.0f, bool approx_pow = false,
                                    bool pcg_rand = true, std::string rng_type = "",
                                    bool batch = true, unsigned int grain_size = 1,
                                    float a = 0.0f, float b = 0.0f,
                                    std::string opt_method = "adam", float alpha = -1.0f,
                                    float beta1 = 0.5f, float beta2 = 0.9f, float eps = 1e-7f,
                                    py::object ai = py::none(), py::object aj = py::none()) {
    arma::sp_mat G_sp = scipy_to_arma_sparse(G);
    arma::mat init_mat = numpy_to_arma_mat(initial_coords);

    const std::size_t requested_threads = thread_no > 0 ? static_cast<std::size_t>(thread_no) : 0;
    OptimizerArgs opt_args(opt_method, alpha == -1.0f ? learning_rate : alpha, beta1, beta2, eps);
    UwotArgs uwot_args(
        method, n_components, spread, min_dist, n_epochs, learning_rate,
        repulsion_strength, negative_sample_rate, approx_pow, pcg_rand,
        batch, seed, requested_threads, grain_size, verbose, opt_args, rng_type
    );

    if (a != 0.0f || b != 0.0f) {
        uwot_args.set_ab(a, b);
    }

    if (!ai.is_none()) {
        uwot_args.ai = optional_float_vector(std::move(ai), "ai");
    }
    if (!aj.is_none()) {
        uwot_args.aj = optional_float_vector(std::move(aj), "aj");
    }

    arma::mat coords;
    {
        py::gil_scoped_release release;
        coords = actionet::layoutNetwork(G_sp, init_mat, std::move(uwot_args));
    }

    return arma_mat_to_numpy(coords);
}

py::array_t<double> compute_node_colors(py::array_t<double> coordinates, int thread_no = 1) {
    arma::mat coords_mat = numpy_to_arma_mat(coordinates);
    arma::mat colors;
    {
        py::gil_scoped_release release;
        colors = actionet::computeNodeColors(coords_mat, thread_no);
    }
    return arma_mat_to_numpy(colors);
}

// =====================================================================================================================

void init_visualization(py::module_ &m) {
    m.def("layout_network", &layout_network, "Layout network (uwot methods)",
          py::arg("G"), py::arg("initial_coords"), py::arg("method") = "umap",
          py::arg("n_components") = 2, py::arg("spread") = 1.0, py::arg("min_dist") = 1.0,
          py::arg("n_epochs") = 0, py::arg("seed") = 0,
          py::arg("thread_no") = 0, py::arg("verbose") = true,
          py::arg("learning_rate") = 1.0f, py::arg("repulsion_strength") = 1.0f,
          py::arg("negative_sample_rate") = 5.0f, py::arg("approx_pow") = false,
          py::arg("pcg_rand") = true, py::arg("rng_type") = "",
          py::arg("batch") = true, py::arg("grain_size") = 1,
          py::arg("a") = 0.0f, py::arg("b") = 0.0f,
          py::arg("opt_method") = "adam", py::arg("alpha") = -1.0f,
          py::arg("beta1") = 0.5f, py::arg("beta2") = 0.9f, py::arg("eps") = 1e-7f,
          py::arg("ai") = py::none(), py::arg("aj") = py::none());

    m.def("compute_node_colors", &compute_node_colors, "Compute node colors from coordinates",
          py::arg("coordinates"), py::arg("thread_no") = 1);
}
