// Pybind11 interface for backed HDF5 matrix operators.

#include "wp_utils.h"
#include "libactionet.hpp"
#include "io/backed_h5ad/backed_sparse_matrix_operator.hpp"
#include "io/backed_h5ad/backed_dense_matrix_operator.hpp"
#include "io/backed_h5ad/create_backed_operator.hpp"

#include <algorithm>
#include <memory>
#include <vector>

namespace py = pybind11;

namespace {
    py::dict svd_to_dict(const actionet::SVDResult& res) {
        py::dict out;
        out["u"] = arma_mat_to_numpy_c(res.U);
        out["d"] = arma_vec_to_numpy(res.sigma);
        out["v"] = arma_mat_to_numpy_c(res.V);
        return out;
    }

    std::vector<double> optional_row_scale(py::object row_scale_factors) {
        if (row_scale_factors.is_none()) {
            return {};
        }

        py::array_t<double, py::array::forcecast> arr =
            row_scale_factors.cast<py::array_t<double, py::array::forcecast>>();
        py::buffer_info buf = arr.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("row_scale_factors must be a 1D array");
        }

        const auto* ptr = static_cast<const double*>(buf.ptr);
        return std::vector<double>(ptr, ptr + static_cast<size_t>(buf.shape[0]));
    }

    // Copy a 1-D int64 numpy array into an arma::uvec.  Consolidates the
    // ``col_indices``/``row_indices`` copy loops that were previously
    // duplicated inline in ``backed_take_columns``.
    arma::uvec int64_array_to_uvec(const py::array_t<int64_t>& arr) {
        auto buf = arr.request();
        if (buf.ndim != 1) {
            throw std::runtime_error(
                "expected a 1D int64 array for backed index conversion");
        }
        const auto n = static_cast<arma::uword>(buf.size);
        arma::uvec out(n);
        const auto* ptr = static_cast<const int64_t*>(buf.ptr);
        for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
            if (ptr[i] < 0) {
                throw std::runtime_error(
                    "backed index arrays must be non-negative");
            }
            out(i) = static_cast<arma::uword>(ptr[i]);
        }
        return out;
    }
} // namespace

std::shared_ptr<actionet::MatrixOperator> create_backed_operator(
    const std::string& file_path,
    const std::string& group_path,
    int chunk_size,
    py::object row_scale_factors,
    bool apply_log1p,
    double log_scale,
    size_t io_target_chunk_bytes,
    double io_target_chunk_fraction_of_cap,
    int n_threads) {
    return actionet::createBackedOperator(
        file_path,
        group_path,
        static_cast<arma::uword>(std::max(1, chunk_size)),
        optional_row_scale(std::move(row_scale_factors)),
        apply_log1p,
        log_scale,
        io_target_chunk_bytes,
        io_target_chunk_fraction_of_cap,
        n_threads
    );
}

py::dict run_svd_backed_operator(std::shared_ptr<actionet::MatrixOperator> op,
                                 int k = 30, int max_it = 0, int seed = 0,
                                 int algorithm = actionet::ALG_HALKO, bool verbose = true) {
    if (!op) {
        throw std::runtime_error("run_svd_backed_operator: operator is null");
    }
    validate_python_svd_algorithm(algorithm, "run_svd_backed_operator");
    actionet::SVDResult res;
    {
        py::gil_scoped_release release;
        res = actionet::runSVD_Operator(*op, k, max_it, seed, algorithm, verbose);
    }
    return svd_to_dict(res);
}

py::dict reduce_kernel_backed_operator(std::shared_ptr<actionet::MatrixOperator> op,
                                       int k = 50, int svd_alg = actionet::ALG_HALKO,
                                       int max_it = 0, int seed = 0, bool verbose = true) {
    if (!op) {
        throw std::runtime_error("reduce_kernel_backed_operator: operator is null");
    }
    validate_python_svd_algorithm(svd_alg, "reduce_kernel_backed_operator");
    actionet::KernelReductionResult res;
    {
        py::gil_scoped_release release;
        res = actionet::reduceKernel_Operator(*op, k, svd_alg, max_it, seed, verbose);
    }
    return kernel_result_to_dict(res);
}

py::dict reduce_kernel_from_svd_backed_operator(std::shared_ptr<actionet::MatrixOperator> op,
                                                py::array_t<double> u, py::object d,
                                                py::array_t<double> v, bool verbose = true) {
    if (!op) {
        throw std::runtime_error("reduce_kernel_from_svd_backed_operator: operator is null");
    }

    actionet::SVDResult svd;
    svd.U = numpy_to_arma_mat(u);
    svd.sigma = parse_sigma(d);
    svd.V = numpy_to_arma_mat(v);
    actionet::KernelReductionResult res;
    {
        py::gil_scoped_release release;
        res = actionet::reduceKernelFromSVD_Operator(*op, svd, verbose);
    }
    return kernel_result_to_dict(res);
}

void init_io(py::module_ &m) {
    py::class_<actionet::MatrixOperator, std::shared_ptr<actionet::MatrixOperator>>(m, "MatrixOperator");

    py::class_<actionet::BackedSparseMatrixOperator, actionet::MatrixOperator,
               std::shared_ptr<actionet::BackedSparseMatrixOperator>>(m, "BackedSparseMatrixOperator")
        .def(py::init<const std::string&, const std::string&, arma::uword,
                      const std::vector<double>&, bool, double, size_t, double, int>(),
             py::arg("file_path"),
             py::arg("group_path") = "/X",
             py::arg("chunk_size") = 4096,
             py::arg("row_scale_factors") = std::vector<double>{},
             py::arg("apply_log1p") = false,
             py::arg("log_scale") = 1.0,
             py::arg("io_target_chunk_bytes") = 0,
             py::arg("io_target_chunk_fraction_of_cap") = 0.5,
             py::arg("n_threads") = 0)
        .def_property_readonly("shape", [](const actionet::BackedSparseMatrixOperator& op) {
            return py::make_tuple(op.rows(), op.cols());
        })
        .def_property_readonly("is_csr", &actionet::BackedSparseMatrixOperator::isCSR)
        .def_property_readonly("file_path", &actionet::BackedSparseMatrixOperator::filePath)
        .def_property_readonly("group_path", &actionet::BackedSparseMatrixOperator::groupPath);

    py::class_<actionet::BackedDenseMatrixOperator, actionet::MatrixOperator,
               std::shared_ptr<actionet::BackedDenseMatrixOperator>>(m, "BackedDenseMatrixOperator")
        .def(py::init<const std::string&, const std::string&, arma::uword,
                      const std::vector<double>&, bool, double, size_t, int>(),
             py::arg("file_path"),
             py::arg("group_path") = "/X",
             py::arg("chunk_size") = 4096,
             py::arg("row_scale_factors") = std::vector<double>{},
             py::arg("apply_log1p") = false,
             py::arg("log_scale") = 1.0,
             py::arg("slab_byte_budget") = 256ULL * 1024 * 1024,
             py::arg("n_threads") = 0)
        .def_property_readonly("shape", [](const actionet::BackedDenseMatrixOperator& op) {
            return py::make_tuple(op.rows(), op.cols());
        })
        .def_property_readonly("file_path", &actionet::BackedDenseMatrixOperator::filePath)
        .def_property_readonly("group_path", &actionet::BackedDenseMatrixOperator::groupPath)
        .def_property_readonly("effective_chunk_size", &actionet::BackedDenseMatrixOperator::effectiveChunkSize);

    m.def("create_backed_operator", &create_backed_operator,
          "Create an HDF5-backed matrix operator (auto-detects sparse vs dense)",
          py::arg("file_path"),
          py::arg("group_path") = "/X",
          py::arg("chunk_size") = 4096,
          py::arg("row_scale_factors") = py::none(),
          py::arg("apply_log1p") = false,
          py::arg("log_scale") = 1.0,
          py::arg("io_target_chunk_bytes") = 0,
          py::arg("io_target_chunk_fraction_of_cap") = 0.5,
          py::arg("n_threads") = 0);

    m.def("run_svd_backed_operator", &run_svd_backed_operator,
          "Run SVD with a MatrixOperator-backed input",
          py::arg("op"), py::arg("k") = 30, py::arg("max_it") = 0, py::arg("seed") = 0,
          py::arg("algorithm") = actionet::ALG_HALKO, py::arg("verbose") = true);

    m.def("reduce_kernel_backed_operator", &reduce_kernel_backed_operator,
          "Reduce kernel with a MatrixOperator-backed input",
          py::arg("op"), py::arg("k") = 50, py::arg("svd_alg") = actionet::ALG_HALKO,
          py::arg("max_it") = 0, py::arg("seed") = 0, py::arg("verbose") = true);

    m.def("reduce_kernel_from_svd_backed_operator", &reduce_kernel_from_svd_backed_operator,
          "Reduce kernel from precomputed SVD with a MatrixOperator-backed input",
          py::arg("op"), py::arg("u"), py::arg("d"), py::arg("v"), py::arg("verbose") = true);

    m.def("backed_take_columns",
          [](std::shared_ptr<actionet::MatrixOperator> op,
             py::array_t<int64_t> col_indices_arr,
             py::object row_indices_obj,
             bool prefer_sparse) -> py::object {
              if (!op) {
                  throw std::runtime_error("backed_take_columns: operator is null");
              }

              arma::uvec col_indices = int64_array_to_uvec(col_indices_arr);

              arma::uvec row_indices;
              if (!row_indices_obj.is_none()) {
                  py::array_t<int64_t> row_arr = row_indices_obj.cast<py::array_t<int64_t>>();
                  row_indices = int64_array_to_uvec(row_arr);
              }

              if (prefer_sparse) {
                  arma::sp_mat result;
                  {
                      py::gil_scoped_release release;
                      result = dispatch_backed_op(
                          op,
                          "backed_take_columns",
                          [&](auto& concrete_op) {
                              return concrete_op.takeColumnsSparse(col_indices, row_indices);
                          });
                  }
                  return arma_sparse_to_scipy(result);
              } else {
                  arma::mat result;
                  {
                      py::gil_scoped_release release;
                      result = dispatch_backed_op(
                          op,
                          "backed_take_columns",
                          [&](auto& concrete_op) {
                              return concrete_op.takeColumnsDense(col_indices, row_indices);
                          });
                  }
                  return py::cast<py::object>(arma_mat_to_numpy(result));
              }
          },
          "Extract columns from a backed matrix operator",
          py::arg("op"),
          py::arg("col_indices"),
          py::arg("row_indices") = py::none(),
          py::arg("prefer_sparse") = false);
}
