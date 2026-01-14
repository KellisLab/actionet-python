// Pybind11 interface utilities for ACTIONet
// Conversion functions between Python and C++ data structures

#include "wp_utils.h"

// Convert NumPy array to Armadillo dense matrix
arma::mat numpy_to_arma_mat(py::array_t<double> arr) {
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
    auto indices = csc.attr("indices").cast<py::array_t<int>>();
    auto indptr = csc.attr("indptr").cast<py::array_t<int>>();
    auto shape = csc.attr("shape").cast<std::pair<int, int>>();

    auto data_ptr = data.data();
    auto indices_ptr = indices.data();
    auto indptr_ptr = indptr.data();

    arma::umat locations(2, data.size());
    arma::vec values(data.size());

    size_t idx = 0;
    for (int col = 0; col < shape.second; ++col) {
        for (int j = indptr_ptr[col]; j < indptr_ptr[col + 1]; ++j) {
            locations(0, idx) = indices_ptr[j];  // row
            locations(1, idx) = col;              // col
            values(idx) = data_ptr[j];
            ++idx;
        }
    }

    return arma::sp_mat(locations, values, shape.first, shape.second);
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
    std::vector<int> rows;
    std::vector<int> cols;

    for (arma::sp_mat::const_iterator it = sp_mat.begin(); it != sp_mat.end(); ++it) {
        data.push_back(*it);
        rows.push_back(it.row());
        cols.push_back(it.col());
    }

    return scipy_sparse.attr("coo_matrix")(
        py::make_tuple(
            py::array_t<double>(data.size(), data.data()),
            py::make_tuple(
                py::array_t<int>(rows.size(), rows.data()),
                py::array_t<int>(cols.size(), cols.data())
            )
        ),
        py::make_tuple(sp_mat.n_rows, sp_mat.n_cols)
    ).attr("tocsr")();
}

// Convert NumPy vector to Armadillo vector
arma::vec numpy_to_arma_vec(py::array_t<double> arr) {
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
