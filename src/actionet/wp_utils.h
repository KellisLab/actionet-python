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

// Convert Armadillo dense matrix to NumPy array
py::array_t<double> arma_mat_to_numpy(const arma::mat& mat);

// Convert Armadillo sparse matrix to SciPy sparse matrix
py::object arma_sparse_to_scipy(const arma::sp_mat& sp_mat);

// Convert libactionet CSRGraph to SciPy CSR sparse matrix.
py::object csr_graph_to_scipy(const actionet::CSRGraph& graph);

// Convert NumPy vector to Armadillo vector
arma::vec numpy_to_arma_vec(py::array_t<double, py::array::c_style | py::array::forcecast> arr);

// Convert Armadillo vector to NumPy array
py::array_t<double> arma_vec_to_numpy(const arma::vec& vec);

/// @brief MatrixOperator wrapper around a Python object exposing shape/matvec/rmatvec.
///
/// The Python object must provide:
///   - .shape  : tuple (m, n)
///   - .matvec(x: ndarray) -> ndarray  (compute y = A * x,   len(x)=n, len(y)=m)
///   - .rmatvec(x: ndarray) -> ndarray (compute y = A' * x,  len(x)=m, len(y)=n)
///
/// Threading / GIL:
///   Each call to matvec/rmatvec acquires the Python GIL.  This means the
///   operator path is inherently single-threaded from C++.  PRIMME is configured
///   in single-threaded mode for the operator path to avoid deadlock.  If multi-
///   threaded PRIMME were needed, the Python callback would have to release the
///   GIL internally (e.g. via nogil in Cython or py::gil_scoped_release).
class PythonMatrixOperator final : public actionet::MatrixOperator {
public:
    explicit PythonMatrixOperator(py::object op);

    arma::uword rows() const override { return rows_; }
    arma::uword cols() const override { return cols_; }

    void matvec(const arma::vec& x, arma::vec& y) const override;
    void rmatvec(const arma::vec& x, arma::vec& y) const override;

private:
    py::object op_;
    arma::uword rows_;
    arma::uword cols_;
};

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
