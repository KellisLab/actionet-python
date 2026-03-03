// Pybind11 interface for backed HDF5 matrix operators.

#include "wp_utils.h"
#include "libactionet.hpp"
#include "io/backed_h5ad/backed_sparse_matrix_operator.hpp"

#include <algorithm>
#include <memory>
#include <vector>

namespace py = pybind11;

namespace {
    py::dict svd_to_dict(const actionet::SVDResult& res) {
        py::dict out;
        out["u"] = arma_mat_to_numpy(res.U);
        out["d"] = arma_vec_to_numpy(res.sigma);
        out["v"] = arma_mat_to_numpy(res.V);
        return out;
    }

    py::dict kernel_to_dict(const actionet::KernelReductionResult& res) {
        py::dict out;
        out["S_r"] = arma_mat_to_numpy(res.S_r);
        out["sigma"] = arma_vec_to_numpy(res.sigma);
        out["U"] = arma_mat_to_numpy(res.U);
        out["A"] = arma_mat_to_numpy(res.A);
        out["B"] = arma_mat_to_numpy(res.B);
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
} // namespace

std::shared_ptr<actionet::BackedSparseMatrixOperator> create_backed_operator(
    const std::string& file_path,
    const std::string& group_path,
    int chunk_size,
    py::object row_scale_factors,
    bool apply_log1p) {
    return std::make_shared<actionet::BackedSparseMatrixOperator>(
        file_path,
        group_path,
        static_cast<arma::uword>(std::max(1, chunk_size)),
        optional_row_scale(std::move(row_scale_factors)),
        apply_log1p
    );
}

py::dict run_svd_backed_operator(std::shared_ptr<actionet::MatrixOperator> op,
                                 int k = 30, int max_it = 0, int seed = 0,
                                 int algorithm = ALG_HALKO, bool verbose = true) {
    if (!op) {
        throw std::runtime_error("run_svd_backed_operator: operator is null");
    }
    actionet::SVDResult res = actionet::runSVD_Operator(*op, k, max_it, seed, algorithm, verbose);
    return svd_to_dict(res);
}

py::dict reduce_kernel_backed_operator(std::shared_ptr<actionet::MatrixOperator> op,
                                       int k = 50, int svd_alg = ALG_HALKO,
                                       int max_it = 0, int seed = 0, bool verbose = true) {
    if (!op) {
        throw std::runtime_error("reduce_kernel_backed_operator: operator is null");
    }
    actionet::KernelReductionResult res =
        actionet::reduceKernel_Operator(*op, k, svd_alg, max_it, seed, verbose);
    return kernel_to_dict(res);
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
    actionet::KernelReductionResult res = actionet::reduceKernelFromSVD_Operator(*op, svd, verbose);
    return kernel_to_dict(res);
}

void init_io(py::module_ &m) {
    py::class_<actionet::MatrixOperator, std::shared_ptr<actionet::MatrixOperator>>(m, "MatrixOperator");

    py::class_<actionet::BackedSparseMatrixOperator, actionet::MatrixOperator,
               std::shared_ptr<actionet::BackedSparseMatrixOperator>>(m, "BackedSparseMatrixOperator")
        .def(py::init<const std::string&, const std::string&, arma::uword,
                      const std::vector<double>&, bool>(),
             py::arg("file_path"),
             py::arg("group_path") = "/X",
             py::arg("chunk_size") = 4096,
             py::arg("row_scale_factors") = std::vector<double>{},
             py::arg("apply_log1p") = false)
        .def_property_readonly("shape", [](const actionet::BackedSparseMatrixOperator& op) {
            return py::make_tuple(op.rows(), op.cols());
        })
        .def_property_readonly("is_csr", &actionet::BackedSparseMatrixOperator::isCSR)
        .def_property_readonly("file_path", &actionet::BackedSparseMatrixOperator::filePath)
        .def_property_readonly("group_path", &actionet::BackedSparseMatrixOperator::groupPath);

    m.def("create_backed_operator", &create_backed_operator,
          "Create an HDF5-backed sparse matrix operator",
          py::arg("file_path"),
          py::arg("group_path") = "/X",
          py::arg("chunk_size") = 4096,
          py::arg("row_scale_factors") = py::none(),
          py::arg("apply_log1p") = false);

    m.def("run_svd_backed_operator", &run_svd_backed_operator,
          "Run SVD with a MatrixOperator-backed input",
          py::arg("op"), py::arg("k") = 30, py::arg("max_it") = 0, py::arg("seed") = 0,
          py::arg("algorithm") = ALG_HALKO, py::arg("verbose") = true);

    m.def("reduce_kernel_backed_operator", &reduce_kernel_backed_operator,
          "Reduce kernel with a MatrixOperator-backed input",
          py::arg("op"), py::arg("k") = 50, py::arg("svd_alg") = ALG_HALKO,
          py::arg("max_it") = 0, py::arg("seed") = 0, py::arg("verbose") = true);

    m.def("reduce_kernel_from_svd_backed_operator", &reduce_kernel_from_svd_backed_operator,
          "Reduce kernel from precomputed SVD with a MatrixOperator-backed input",
          py::arg("op"), py::arg("u"), py::arg("d"), py::arg("v"), py::arg("verbose") = true);
}
