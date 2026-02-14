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

namespace py = pybind11;

// Convert NumPy array to Armadillo dense matrix
arma::mat numpy_to_arma_mat(py::array_t<double, py::array::c_style | py::array::forcecast> arr);

// Convert SciPy sparse matrix to Armadillo sparse matrix
arma::sp_mat scipy_to_arma_sparse(py::object scipy_sparse);

// Convert Armadillo dense matrix to NumPy array
py::array_t<double> arma_mat_to_numpy(const arma::mat& mat);

// Convert Armadillo sparse matrix to SciPy sparse matrix
py::object arma_sparse_to_scipy(const arma::sp_mat& sp_mat);

// Convert NumPy vector to Armadillo vector
arma::vec numpy_to_arma_vec(py::array_t<double, py::array::c_style | py::array::forcecast> arr);

// Convert Armadillo vector to NumPy array
py::array_t<double> arma_vec_to_numpy(const arma::vec& vec);

// MatrixOperator wrapper around a Python object exposing shape/matvec/rmatvec.
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

#endif // WP_UTILS_H
