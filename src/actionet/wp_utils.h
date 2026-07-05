// Pybind11 interface utilities for ACTIONet
// Conversion functions between Python and C++ data structures

#ifndef WP_UTILS_H
#define WP_UTILS_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Armadillo includes (libactionet uses Armadillo)
#include "armadillo"
#include "decomposition/matrix_operator.hpp"
#include "network/build_network_core.hpp"

namespace py = pybind11;

// Convert NumPy array to Armadillo dense matrix
arma::mat numpy_to_arma_mat(py::array_t<double, py::array::c_style | py::array::forcecast> arr);

// Convert SciPy sparse matrix to Armadillo sparse matrix
arma::sp_mat scipy_to_arma_sparse(py::object scipy_sparse);

// Convert Armadillo dense matrix to NumPy array (Fortran-order, single memcpy)
py::array_t<double> arma_mat_to_numpy(const arma::mat& mat);

// Convert Armadillo dense matrix to C-contiguous NumPy array.
// Costlier than arma_mat_to_numpy (element-wise transpose), but the result
// is optimal for subsequent h5py/HDF5 writes which expect row-major data.
py::array_t<double> arma_mat_to_numpy_c(const arma::mat& mat);

// Convert Armadillo sparse matrix to SciPy CSC sparse matrix (direct internal copy)
py::object arma_sparse_to_scipy(const arma::sp_mat& sp_mat);

// Convert libactionet CSRGraph to SciPy CSR sparse matrix.
py::object csr_graph_to_scipy(const actionet::CSRGraph& graph);

// Convert NumPy vector to Armadillo vector
arma::vec numpy_to_arma_vec(py::array_t<double, py::array::c_style | py::array::forcecast> arr);

// Convert Armadillo vector to NumPy array
py::array_t<double> arma_vec_to_numpy(const arma::vec& vec);

/// @brief Parse a flexible singular-value argument (1D, Nx1, or 1xN) into arma::vec.
inline arma::vec parse_sigma(py::object d) {
    py::array_t<double, py::array::forcecast> d_arr = d.cast<py::array_t<double, py::array::forcecast>>();
    py::buffer_info d_buf = d_arr.request();
    auto* ptr = static_cast<double*>(d_buf.ptr);
    if (d_buf.ndim == 1) {
        return arma::vec(ptr, static_cast<arma::uword>(d_buf.shape[0]), true, true);
    }
    if (d_buf.ndim == 2 && (d_buf.shape[0] == 1 || d_buf.shape[1] == 1)) {
        return arma::vec(ptr, static_cast<arma::uword>(d_buf.shape[0] * d_buf.shape[1]), true, true);
    }
    throw std::runtime_error("Expected singular values `d` to be a 1D vector or Nx1/1xN array");
}

#endif // WP_UTILS_H
