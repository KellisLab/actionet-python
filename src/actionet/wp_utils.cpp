// Pybind11 interface utilities for ACTIONet
// Conversion functions between Python and C++ data structures

#include "wp_utils.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename SizeT>
py::ssize_t checked_py_size(SizeT value, const char* name) {
    if (value > static_cast<SizeT>(std::numeric_limits<py::ssize_t>::max())) {
        throw std::runtime_error(std::string(name) + " exceeds Python array limits");
    }
    return static_cast<py::ssize_t>(value);
}

template <typename IndexT>
bool fits_sparse_index_type(const arma::sp_mat& sp_mat) {
    constexpr arma::uword index_max = static_cast<arma::uword>(std::numeric_limits<IndexT>::max());
    return sp_mat.n_rows <= index_max && sp_mat.n_cols <= index_max && sp_mat.n_nonzero <= index_max;
}

template <typename IndexT>
bool fits_sparse_index_type(const actionet::CSRGraph& graph) {
    constexpr actionet::CSROffset index_max =
        static_cast<actionet::CSROffset>(std::numeric_limits<IndexT>::max());
    return graph.n <= index_max && graph.nnz() <= index_max;
}

py::object arma_sparse_to_scipy_legacy(const arma::sp_mat& sp_mat) {
    py::module_ scipy_sparse = py::module_::import("scipy.sparse");

    std::vector<double> data;
    std::vector<py::ssize_t> rows;
    std::vector<py::ssize_t> cols;

    data.reserve(static_cast<size_t>(sp_mat.n_nonzero));
    rows.reserve(static_cast<size_t>(sp_mat.n_nonzero));
    cols.reserve(static_cast<size_t>(sp_mat.n_nonzero));

    for (arma::sp_mat::const_iterator it = sp_mat.begin(); it != sp_mat.end(); ++it) {
        data.push_back(*it);
        rows.push_back(static_cast<py::ssize_t>(it.row()));
        cols.push_back(static_cast<py::ssize_t>(it.col()));
    }

    py::array_t<double> data_arr(data.size());
    py::array_t<py::ssize_t> rows_arr(rows.size());
    py::array_t<py::ssize_t> cols_arr(cols.size());

    if (!data.empty()) {
        std::memcpy(data_arr.mutable_data(), data.data(), data.size() * sizeof(double));
        std::memcpy(rows_arr.mutable_data(), rows.data(), rows.size() * sizeof(py::ssize_t));
        std::memcpy(cols_arr.mutable_data(), cols.data(), cols.size() * sizeof(py::ssize_t));
    }

    return scipy_sparse.attr("coo_matrix")(
        py::make_tuple(
            data_arr,
            py::make_tuple(
                rows_arr,
                cols_arr
            )
        ),
        py::make_tuple(sp_mat.n_rows, sp_mat.n_cols)
    ).attr("tocsr")();
}

template <typename IndexT>
py::object arma_sparse_to_scipy_csr_impl(const arma::sp_mat& sp_mat) {
    py::module_ scipy_sparse = py::module_::import("scipy.sparse");

    const py::ssize_t n_rows = checked_py_size(sp_mat.n_rows, "Sparse matrix row count");
    const py::ssize_t n_cols = checked_py_size(sp_mat.n_cols, "Sparse matrix column count");
    const py::ssize_t nnz = checked_py_size(sp_mat.n_nonzero, "Sparse matrix nnz");

    py::array_t<double> data(nnz);
    py::array_t<IndexT> indices(nnz);
    py::array_t<IndexT> indptr(n_rows + 1);

    double* data_ptr = data.mutable_data();
    IndexT* indices_ptr = indices.mutable_data();
    IndexT* indptr_ptr = indptr.mutable_data();

    std::fill(indptr_ptr, indptr_ptr + n_rows + 1, IndexT{0});

    for (arma::sp_mat::const_iterator it = sp_mat.begin(); it != sp_mat.end(); ++it) {
        ++indptr_ptr[static_cast<py::ssize_t>(it.row()) + 1];
    }

    for (py::ssize_t row = 0; row < n_rows; ++row) {
        indptr_ptr[row + 1] += indptr_ptr[row];
    }

    std::vector<IndexT> next_offset(static_cast<size_t>(n_rows));
    std::copy(indptr_ptr, indptr_ptr + n_rows, next_offset.begin());

    for (arma::sp_mat::const_iterator it = sp_mat.begin(); it != sp_mat.end(); ++it) {
        const size_t row = static_cast<size_t>(it.row());
        const IndexT dest = next_offset[row]++;
        indices_ptr[dest] = static_cast<IndexT>(it.col());
        data_ptr[dest] = *it;
    }

    return scipy_sparse.attr("csr_matrix")(
        py::make_tuple(data, indices, indptr),
        py::make_tuple(n_rows, n_cols)
    );
}

// Directly export Armadillo sp_mat as scipy CSC by reading its internal CSC storage.
// Armadillo stores sp_mat in CSC format internally, so this avoids any conversion.
template <typename IndexT>
py::object arma_sparse_to_scipy_csc_impl(const arma::sp_mat& sp_mat) {
    py::module_ scipy_sparse = py::module_::import("scipy.sparse");

    const py::ssize_t n_rows = checked_py_size(sp_mat.n_rows, "Sparse matrix row count");
    const py::ssize_t n_cols = checked_py_size(sp_mat.n_cols, "Sparse matrix column count");
    const py::ssize_t nnz    = checked_py_size(sp_mat.n_nonzero, "Sparse matrix nnz");

    py::array_t<double>  data(nnz);
    py::array_t<IndexT>  indices(nnz);            // row indices
    py::array_t<IndexT>  indptr(n_cols + 1);      // column pointers

    double* data_ptr    = data.mutable_data();
    IndexT* indices_ptr = indices.mutable_data();
    IndexT* indptr_ptr  = indptr.mutable_data();

    // Armadillo sp_mat internals are accessible via col_ptrs and row_indices.
    const arma::uword* arma_col_ptrs   = sp_mat.col_ptrs;
    const arma::uword* arma_row_ind    = sp_mat.row_indices;
    const double*      arma_vals       = sp_mat.values;

    for (py::ssize_t j = 0; j <= n_cols; ++j) {
        indptr_ptr[j] = static_cast<IndexT>(arma_col_ptrs[j]);
    }
    for (py::ssize_t i = 0; i < nnz; ++i) {
        indices_ptr[i] = static_cast<IndexT>(arma_row_ind[i]);
        data_ptr[i]    = arma_vals[i];
    }

    return scipy_sparse.attr("csc_matrix")(
        py::make_tuple(data, indices, indptr),
        py::make_tuple(n_rows, n_cols)
    );
}

template <typename IndexT>
py::object csr_graph_to_scipy_impl(const actionet::CSRGraph& graph) {
    py::module_ scipy_sparse = py::module_::import("scipy.sparse");

    const py::ssize_t n = checked_py_size(graph.n, "Sparse matrix row count");
    const py::ssize_t nnz = checked_py_size(graph.nnz(), "Sparse matrix nnz");

    py::array_t<double> data(nnz);
    py::array_t<IndexT> indices(nnz);
    py::array_t<IndexT> indptr(n + 1);

    double* data_ptr = data.mutable_data();
    IndexT* indices_ptr = indices.mutable_data();
    IndexT* indptr_ptr = indptr.mutable_data();

    for (py::ssize_t i = 0; i < n; ++i) {
        indptr_ptr[i] = static_cast<IndexT>(graph.indptr[static_cast<size_t>(i)]);
    }
    indptr_ptr[n] = static_cast<IndexT>(graph.indptr.back());

    for (py::ssize_t i = 0; i < nnz; ++i) {
        const size_t idx = static_cast<size_t>(i);
        indices_ptr[i] = static_cast<IndexT>(graph.indices[idx]);
        data_ptr[i] = static_cast<double>(graph.data[idx]);
    }

    return scipy_sparse.attr("csr_matrix")(
        py::make_tuple(data, indices, indptr),
        py::make_tuple(n, n)
    );
}

} // namespace

// Convert NumPy array to Armadillo dense matrix
arma::mat numpy_to_arma_mat(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Expected 2D array");
    }

    auto ptr = static_cast<const double*>(buf.ptr);
    arma::uword n_rows = static_cast<arma::uword>(buf.shape[0]);
    arma::uword n_cols = static_cast<arma::uword>(buf.shape[1]);

    // C-contiguous row-major (n_rows, n_cols) is equivalent to Fortran-contiguous
    // column-major (n_cols, n_rows).  Construct the transposed view without copying,
    // then transpose in Armadillo (optimised for non-square matrices).
    arma::mat tmp(const_cast<double*>(ptr), n_cols, n_rows, /*copy_aux_mem=*/false, /*strict=*/true);
    return tmp.t();
}

// Convert SciPy sparse matrix to Armadillo sparse matrix
arma::sp_mat scipy_to_arma_sparse(py::object scipy_sparse) {
    // Ensure CSC format — AnnData typically stores X as CSC, so this is usually a no-op.
    py::object csc = scipy_sparse.attr("tocsc")();

    auto data = csc.attr("data").cast<py::array_t<double, py::array::forcecast>>();
    auto indices = csc.attr("indices").cast<py::array_t<py::ssize_t, py::array::forcecast>>();
    auto indptr = csc.attr("indptr").cast<py::array_t<py::ssize_t, py::array::forcecast>>();
    auto shape = csc.attr("shape").cast<std::pair<py::ssize_t, py::ssize_t>>();

    auto data_ptr = data.data();
    auto indices_ptr = indices.data();
    auto indptr_ptr = indptr.data();

    if (shape.first < 0 || shape.second < 0) {
        throw std::runtime_error("Sparse matrix shape must be non-negative");
    }

    const arma::uword n_rows = static_cast<arma::uword>(shape.first);
    const arma::uword n_cols = static_cast<arma::uword>(shape.second);
    const arma::uword nnz    = static_cast<arma::uword>(data.size());

    if (indptr.size() < 1 || static_cast<arma::uword>(indptr.size()) != (n_cols + 1)) {
        throw std::runtime_error("Invalid CSC indptr length");
    }
    if (indptr_ptr[indptr.size() - 1] != static_cast<py::ssize_t>(nnz)) {
        throw std::runtime_error("CSC indptr does not match data length");
    }

    auto to_uword = [](py::ssize_t v, const char* name) -> arma::uword {
        if (v < 0) {
            throw std::runtime_error(std::string("Negative index in sparse matrix: ") + name);
        }
        if (static_cast<unsigned long long>(v) > std::numeric_limits<arma::uword>::max()) {
            throw std::runtime_error(std::string("Index too large for Armadillo uword: ") + name);
        }
        return static_cast<arma::uword>(v);
    };

    // Build Armadillo CSC arrays directly — no COO intermediary needed.
    // Armadillo sp_mat internal layout is CSC: row_indices + col_ptrs + values.
    arma::uvec row_indices(nnz);
    arma::uvec col_ptrs(n_cols + 1);
    arma::vec  values(nnz);

    for (arma::uword i = 0; i < nnz; ++i) {
        row_indices(i) = to_uword(indices_ptr[i], "row");
        values(i)      = data_ptr[i];
    }
    for (arma::uword i = 0; i <= n_cols; ++i) {
        col_ptrs(i) = to_uword(indptr_ptr[i], "colptr");
    }

    // Armadillo batch CSC constructor: sp_mat(row_indices, col_ptrs, values, n_rows, n_cols)
    // The CSC arrays from scipy are already sorted (per-column row indices are sorted),
    // so sort_locations=false avoids an unnecessary sort pass.
    return arma::sp_mat(row_indices, col_ptrs, values, n_rows, n_cols, /*sort_locations=*/false);
}

// Convert Armadillo dense matrix to NumPy array (Fortran-order, single memcpy)
py::array_t<double> arma_mat_to_numpy(const arma::mat& mat) {
    // Armadillo is column-major (Fortran order).  Return a Fortran-order NumPy
    // array with matching strides so the data can be copied with a single memcpy
    // rather than an element-by-element loop.
    const py::ssize_t n_rows = static_cast<py::ssize_t>(mat.n_rows);
    const py::ssize_t n_cols = static_cast<py::ssize_t>(mat.n_cols);
    const py::ssize_t elem   = static_cast<py::ssize_t>(sizeof(double));

    // Fortran-order strides: stride[0]=1 element, stride[1]=n_rows elements
    std::vector<py::ssize_t> shape   = {n_rows, n_cols};
    std::vector<py::ssize_t> strides = {elem, n_rows * elem};

    py::array_t<double> arr(shape, strides);
    std::memcpy(arr.mutable_data(), mat.memptr(), mat.n_elem * sizeof(double));
    return arr;
}

// Convert Armadillo dense matrix to C-contiguous (row-major) NumPy array.
// Performs a column-major → row-major transpose during the copy.  Slightly
// costlier than arma_mat_to_numpy, but the returned array is optimal for
// h5py/HDF5 writes which store data in row-major order.
py::array_t<double> arma_mat_to_numpy_c(const arma::mat& mat) {
    const py::ssize_t n_rows = static_cast<py::ssize_t>(mat.n_rows);
    const py::ssize_t n_cols = static_cast<py::ssize_t>(mat.n_cols);
    const py::ssize_t elem   = static_cast<py::ssize_t>(sizeof(double));

    std::vector<py::ssize_t> shape   = {n_rows, n_cols};
    std::vector<py::ssize_t> strides = {n_cols * elem, elem};  // C-order

    py::array_t<double> arr(shape, strides);
    double* dst       = arr.mutable_data();
    const double* src = mat.memptr();
    for (py::ssize_t r = 0; r < n_rows; ++r) {
        for (py::ssize_t c = 0; c < n_cols; ++c) {
            dst[r * n_cols + c] = src[c * n_rows + r];
        }
    }
    return arr;
}

// Convert Armadillo sparse matrix to SciPy sparse matrix (CSC — matches Armadillo's internal layout)
py::object arma_sparse_to_scipy(const arma::sp_mat& sp_mat) {
    try {
        if (fits_sparse_index_type<std::int32_t>(sp_mat)) {
            return arma_sparse_to_scipy_csc_impl<std::int32_t>(sp_mat);
        }
        return arma_sparse_to_scipy_csc_impl<std::int64_t>(sp_mat);
    } catch (const py::error_already_set&) {
        throw;
    } catch (const std::exception&) {
        // Fall back to the iterator-based CSR path if direct CSC extraction fails
        // (e.g. on an unexpected Armadillo build where internal arrays differ).
        if (fits_sparse_index_type<std::int32_t>(sp_mat)) {
            return arma_sparse_to_scipy_csr_impl<std::int32_t>(sp_mat);
        }
        return arma_sparse_to_scipy_csr_impl<std::int64_t>(sp_mat);
    }
}

py::object csr_graph_to_scipy(const actionet::CSRGraph& graph) {
    if (graph.indptr.size() != static_cast<size_t>(graph.n) + 1) {
        throw std::runtime_error("CSRGraph indptr length does not match row count");
    }
    if (graph.indices.size() != graph.data.size()) {
        throw std::runtime_error("CSRGraph indices/data lengths do not match");
    }

    if (fits_sparse_index_type<std::int32_t>(graph)) {
        return csr_graph_to_scipy_impl<std::int32_t>(graph);
    }
    return csr_graph_to_scipy_impl<std::int64_t>(graph);
}

// Convert NumPy vector to Armadillo vector
arma::vec numpy_to_arma_vec(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Expected 1D array");
    }

    auto ptr = static_cast<double*>(buf.ptr);
    arma::vec vec(static_cast<arma::uword>(buf.shape[0]));
    std::memcpy(vec.memptr(), ptr, static_cast<size_t>(buf.shape[0]) * sizeof(double));
    return vec;
}

// Convert Armadillo vector to NumPy array
py::array_t<double> arma_vec_to_numpy(const arma::vec& vec) {
    py::array_t<double> arr(vec.n_elem);
    auto buf = arr.request();
    double* ptr = static_cast<double*>(buf.ptr);
    std::memcpy(ptr, vec.memptr(), vec.n_elem * sizeof(double));
    return arr;
}

PythonMatrixOperator::PythonMatrixOperator(py::object op) : op_(std::move(op)), rows_(0), cols_(0) {
    py::gil_scoped_acquire gil;
    py::object shape_obj = op_.attr("shape");
    auto shape = shape_obj.cast<std::pair<py::ssize_t, py::ssize_t>>();
    if (shape.first < 0 || shape.second < 0) {
        throw std::runtime_error("PythonMatrixOperator: negative shape is invalid");
    }
    rows_ = static_cast<arma::uword>(shape.first);
    cols_ = static_cast<arma::uword>(shape.second);
}

void PythonMatrixOperator::matvec(const arma::vec& x, arma::vec& y) const {
    py::gil_scoped_acquire gil;

    // Create a zero-copy NumPy view over the Armadillo input vector.
    // The Armadillo vector is const and its data is contiguous, so this is safe
    // as long as the Python callback does not store the array beyond the call.
    py::array_t<double> x_arr({static_cast<py::ssize_t>(x.n_elem)},
                               {static_cast<py::ssize_t>(sizeof(double))},
                               const_cast<double*>(x.memptr()),
                               py::none());

    py::object out_obj = op_.attr("matvec")(x_arr);
    py::array_t<double, py::array::forcecast> out_arr = out_obj.cast<py::array_t<double, py::array::forcecast>>();
    py::buffer_info buf = out_arr.request();

    if (buf.ndim != 1 || static_cast<arma::uword>(buf.shape[0]) != rows_) {
        throw std::runtime_error("PythonMatrixOperator.matvec returned vector with incorrect size");
    }

    y.set_size(rows_);
    std::memcpy(y.memptr(), static_cast<double*>(buf.ptr), rows_ * sizeof(double));
}

void PythonMatrixOperator::rmatvec(const arma::vec& x, arma::vec& y) const {
    py::gil_scoped_acquire gil;

    // Zero-copy NumPy view (see matvec above).
    py::array_t<double> x_arr({static_cast<py::ssize_t>(x.n_elem)},
                               {static_cast<py::ssize_t>(sizeof(double))},
                               const_cast<double*>(x.memptr()),
                               py::none());

    py::object out_obj = op_.attr("rmatvec")(x_arr);
    py::array_t<double, py::array::forcecast> out_arr = out_obj.cast<py::array_t<double, py::array::forcecast>>();
    py::buffer_info buf = out_arr.request();

    if (buf.ndim != 1 || static_cast<arma::uword>(buf.shape[0]) != cols_) {
        throw std::runtime_error("PythonMatrixOperator.rmatvec returned vector with incorrect size");
    }

    y.set_size(cols_);
    std::memcpy(y.memptr(), static_cast<double*>(buf.ptr), cols_ * sizeof(double));
}
