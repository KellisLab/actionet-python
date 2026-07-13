// Pybind11 interface utilities for ACTIONet
// Conversion functions between Python and C++ data structures

#ifndef WP_UTILS_H
#define WP_UTILS_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Armadillo includes (libactionet uses Armadillo)
#include "armadillo"
#include "action/reduce_kernel.hpp"
#include "decomposition/svd_main.hpp"
#include "decomposition/matrix_operator.hpp"
#include "io/backed_h5ad/backed_dense_matrix_operator.hpp"
#include "io/backed_h5ad/backed_sparse_matrix_operator.hpp"
#include "network/build_network_core.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;

/// @brief Validate SVD algorithm IDs accepted by the Python extension.
///
/// The Python API exposes only IRLB and Halko. Keep this guard in pybind
/// entry points so private ``actionet._core`` calls cannot bypass the public
/// Python wrapper's string validation.
inline void validate_python_svd_algorithm(int algorithm, const char* context) {
    if (algorithm == actionet::ALG_IRLB || algorithm == actionet::ALG_HALKO) {
        return;
    }
    throw std::runtime_error(
        std::string(context) + ": unsupported SVD algorithm id " +
        std::to_string(algorithm) + "; valid IDs: 0 (IRLB), 1 (Halko)");
}

// Convert NumPy array to Armadillo dense matrix
arma::mat numpy_to_arma_mat(py::array_t<double, py::array::c_style | py::array::forcecast> arr);

// Convert SciPy sparse matrix to Armadillo sparse matrix
arma::sp_mat scipy_to_arma_sparse(py::object scipy_sparse);

// Convert Armadillo dense matrix to NumPy array (Fortran-order, single memcpy)
py::array_t<double> arma_mat_to_numpy(const arma::mat& mat);

// Convert Armadillo dense matrix to C-contiguous NumPy array.
// Uses a cache-blocked (32x32) tiled transpose so both reads and writes hit
// L1; peak memory is unchanged vs a raw copy.  Optimal for downstream
// h5py/HDF5 writes which expect row-major data.
py::array_t<double> arma_mat_to_numpy_c(const arma::mat& mat);

// Convert Armadillo sparse matrix to SciPy CSC sparse matrix (direct internal copy)
py::object arma_sparse_to_scipy(const arma::sp_mat& sp_mat);

// Convert libactionet CSRGraph to SciPy CSR sparse matrix.
py::object csr_graph_to_scipy(const actionet::CSRGraph& graph);

// Convert NumPy vector to Armadillo vector
arma::vec numpy_to_arma_vec(py::array_t<double, py::array::c_style | py::array::forcecast> arr);

// Convert Armadillo vector to NumPy array
py::array_t<double> arma_vec_to_numpy(const arma::vec& vec);

/// @brief Copy an Armadillo integer index vector (uword) into a NumPy array of
/// dtype ``T``.  Each call site picks the Python-visible dtype (typically
/// ``int`` for compactness or ``int64_t`` for full-range indices); the loop
/// body is trivially the same.  Kept in the header so all wrapper TUs share
/// a single implementation.
template <typename T>
py::array_t<T> arma_uvec_to_numpy(const arma::uvec& vec) {
    py::array_t<T> arr(vec.n_elem);
    auto buf = arr.request();
    auto* ptr = static_cast<T*>(buf.ptr);
    for (arma::uword i = 0; i < vec.n_elem; ++i) {
        ptr[i] = static_cast<T>(vec(i));
    }
    return arr;
}

/// @brief Copy an Armadillo signed integer vector (sword / ivec) into a NumPy
/// array of dtype ``T``.  Companion to ``arma_uvec_to_numpy``.
template <typename T>
py::array_t<T> arma_ivec_to_numpy(const arma::ivec& vec) {
    py::array_t<T> arr(vec.n_elem);
    auto buf = arr.request();
    auto* ptr = static_cast<T*>(buf.ptr);
    for (arma::uword i = 0; i < vec.n_elem; ++i) {
        ptr[i] = static_cast<T>(vec(i));
    }
    return arr;
}

/// @brief Pack (S_r, sigma, U, A, B) into an arma::field<arma::mat> in Plan-02
/// public layout: {S_r (cells x k), sigma, U (genes x k), A, B}.
arma::field<arma::mat> pack_reduction_field(const arma::mat& S_r, const arma::vec& sigma,
                                            const arma::mat& U, const arma::mat& A,
                                            const arma::mat& B);

/// @brief Unpack a 5-element arma::field<arma::mat> in the Plan-02 layout
/// to a Python dict with keys ("S_r","sigma","U","A","B").
py::dict kernel_field_to_dict(const arma::field<arma::mat>& reduction);

/// @brief Convert a KernelReductionResult struct to a Python dict with keys
/// ("S_r","sigma","U","A","B"). Used by reduce_kernel / orthogonalize APIs.
py::dict kernel_result_to_dict(const actionet::KernelReductionResult& res);

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

/// @brief Dispatch a callable to the concrete backed operator subtype.
///
/// ``op_base`` must be a ``std::shared_ptr<actionet::MatrixOperator>`` that
/// actually points to a ``BackedSparseMatrixOperator`` or
/// ``BackedDenseMatrixOperator``.  ``fn`` is invoked as ``fn(*concrete_op)``
/// with the appropriately-typed reference; use a generic (auto&) lambda so the
/// same body compiles against both sparse and dense operators.  Throws with a
/// ``context``-prefixed message if the pointer is null or the concrete type is
/// neither of the two supported backed operator classes.
template <typename F>
auto dispatch_backed_op(const std::shared_ptr<actionet::MatrixOperator>& op_base,
                        const char* context, F&& fn) {
    if (!op_base) {
        throw std::runtime_error(std::string(context) + ": operator is null");
    }
    if (auto* sparse_op = dynamic_cast<actionet::BackedSparseMatrixOperator*>(op_base.get())) {
        return fn(*sparse_op);
    }
    if (auto* dense_op = dynamic_cast<actionet::BackedDenseMatrixOperator*>(op_base.get())) {
        return fn(*dense_op);
    }
    throw std::runtime_error(std::string(context) + ": unsupported operator type");
}

#endif // WP_UTILS_H
