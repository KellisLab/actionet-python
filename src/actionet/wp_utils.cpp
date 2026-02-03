// Pybind11 interface utilities for ACTIONet
// Conversion functions between Python and C++ data structures

#include "wp_utils.h"
#include <limits>

// Convert NumPy array to Armadillo dense matrix
arma::mat numpy_to_arma_mat(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Expected 2D array");
    }

    auto ptr = static_cast<double*>(buf.ptr);
    // Copy data (Armadillo uses column-major, NumPy default is row-major)
    arma::mat mat(buf.shape[0], buf.shape[1]);
    for (size_t i = 0; i < buf.shape[0]; ++i) {
        for (size_t j = 0; j < buf.shape[1]; ++j) {
            mat(i, j) = ptr[i * buf.shape[1] + j];
        }
    }
    return mat;
}

// Convert SciPy sparse matrix to Armadillo sparse matrix
arma::sp_mat scipy_to_arma_sparse(py::object scipy_sparse) {
    // Convert to CSC format
    py::object csc = scipy_sparse.attr("tocsc")();

    auto data = csc.attr("data").cast<py::array_t<double>>();
    auto indices = csc.attr("indices").cast<py::array_t<py::ssize_t, py::array::forcecast>>();
    auto indptr = csc.attr("indptr").cast<py::array_t<py::ssize_t, py::array::forcecast>>();
    auto shape = csc.attr("shape").cast<std::pair<py::ssize_t, py::ssize_t>>();

    auto data_ptr = data.data();
    auto indices_ptr = indices.data();
    auto indptr_ptr = indptr.data();

    if (shape.first < 0 || shape.second < 0) {
        throw std::runtime_error("Sparse matrix shape must be non-negative");
    }

    const size_t n_rows = static_cast<size_t>(shape.first);
    const size_t n_cols = static_cast<size_t>(shape.second);
    const size_t nnz = static_cast<size_t>(data.size());

    if (indptr.size() < 1 || static_cast<size_t>(indptr.size()) != (n_cols + 1)) {
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

    arma::umat locations(2, nnz);
    arma::vec values(nnz);

    for (py::ssize_t col = 0; col < shape.second; ++col) {
        const py::ssize_t start = indptr_ptr[col];
        const py::ssize_t end = indptr_ptr[col + 1];
        if (start < 0 || end < start) {
            throw std::runtime_error("Invalid CSC indptr range");
        }
        for (py::ssize_t j = start; j < end; ++j) {
            const size_t idx = static_cast<size_t>(j);
            const py::ssize_t row = indices_ptr[j];
            if (static_cast<size_t>(row) >= n_rows) {
                throw std::runtime_error("Row index out of bounds in sparse matrix");
            }
            locations(0, idx) = to_uword(row, "row");
            locations(1, idx) = to_uword(col, "col");
            values(idx) = data_ptr[j];
        }
    }

    return arma::sp_mat(locations, values, n_rows, n_cols);
}

// Convert Armadillo dense matrix to NumPy array
py::array_t<double> arma_mat_to_numpy(const arma::mat& mat) {
    py::array_t<double> arr({mat.n_rows, mat.n_cols});
    auto buf = arr.request();
    double* ptr = static_cast<double*>(buf.ptr);

    for (size_t i = 0; i < mat.n_rows; ++i) {
        for (size_t j = 0; j < mat.n_cols; ++j) {
            ptr[i * mat.n_cols + j] = mat(i, j);
        }
    }
    return arr;
}

// Convert Armadillo sparse matrix to SciPy sparse matrix
py::object arma_sparse_to_scipy(const arma::sp_mat& sp_mat) {
    py::module scipy_sparse = py::module::import("scipy.sparse");

    std::vector<double> data;
    std::vector<py::ssize_t> rows;
    std::vector<py::ssize_t> cols;

    for (arma::sp_mat::const_iterator it = sp_mat.begin(); it != sp_mat.end(); ++it) {
        data.push_back(*it);
        rows.push_back(static_cast<py::ssize_t>(it.row()));
        cols.push_back(static_cast<py::ssize_t>(it.col()));
    }

    return scipy_sparse.attr("coo_matrix")(
        py::make_tuple(
            py::array_t<double>(data.size(), data.data()),
            py::make_tuple(
                py::array_t<py::ssize_t>(rows.size(), rows.data()),
                py::array_t<py::ssize_t>(cols.size(), cols.data())
            )
        ),
        py::make_tuple(sp_mat.n_rows, sp_mat.n_cols)
    ).attr("tocsr")();
}

// Convert NumPy vector to Armadillo vector
arma::vec numpy_to_arma_vec(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Expected 1D array");
    }

    auto ptr = static_cast<double*>(buf.ptr);
    arma::vec vec(buf.shape[0]);
    for (size_t i = 0; i < buf.shape[0]; ++i) {
        vec(i) = ptr[i];
    }
    return vec;
}

// Convert Armadillo vector to NumPy array
py::array_t<double> arma_vec_to_numpy(const arma::vec& vec) {
    py::array_t<double> arr(vec.n_elem);
    auto buf = arr.request();
    double* ptr = static_cast<double*>(buf.ptr);

    for (size_t i = 0; i < vec.n_elem; ++i) {
        ptr[i] = vec(i);
    }
    return arr;
}
