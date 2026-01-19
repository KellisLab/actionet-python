// Pybind11 interface utilities for ACTIONet
// Conversion functions between Python and C++ data structures

#ifndef WP_UTILS_H
#define WP_UTILS_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Armadillo includes (libactionet uses Armadillo)
#include "armadillo"

namespace py = pybind11;

// Convert NumPy array to Armadillo dense matrix
arma::mat numpy_to_arma_mat(py::array_t<double> arr);

// Convert SciPy sparse matrix to Armadillo sparse matrix
arma::sp_mat scipy_to_arma_sparse(py::object scipy_sparse);

// Convert Armadillo dense matrix to NumPy array
py::array_t<double> arma_mat_to_numpy(const arma::mat& mat);

// Convert Armadillo sparse matrix to SciPy sparse matrix
py::object arma_sparse_to_scipy(const arma::sp_mat& sp_mat);

// Convert NumPy vector to Armadillo vector
arma::vec numpy_to_arma_vec(py::array_t<double> arr);

// Convert Armadillo vector to NumPy array
py::array_t<double> arma_vec_to_numpy(const arma::vec& vec);

#endif // WP_UTILS_H
