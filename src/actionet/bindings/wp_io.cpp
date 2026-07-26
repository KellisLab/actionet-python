// Pybind11 interface for backed HDF5 matrix operators.

#include "wp_utils.h"
#include "libactionet.hpp"
#ifndef LIBACTIONET_NO_HDF5
#include "io/backed_h5ad/backed_sparse_matrix_operator.hpp"
#include "io/backed_h5ad/backed_dense_matrix_operator.hpp"
#include "io/backed_h5ad/create_backed_operator.hpp"
#include "io/backed_h5ad/h5ad_matrix_io.hpp"
#endif

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

#ifndef LIBACTIONET_NO_HDF5

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

    const char* matrix_encoding_name(actionet::h5ad::MatrixEncoding encoding) {
        switch (encoding) {
            case actionet::h5ad::MatrixEncoding::Dense:
                return "dense";
            case actionet::h5ad::MatrixEncoding::CSR:
                return "csr";
            case actionet::h5ad::MatrixEncoding::CSC:
                return "csc";
        }
        throw std::runtime_error("unknown native H5AD matrix encoding");
    }

    py::dict dataset_info_to_dict(const actionet::h5ad::DatasetInfo& info) {
        py::dict out;
        out["name"] = info.name;
        out["dtype"] = info.dtype;
        out["shape"] = info.shape;
        out["logical_bytes"] = info.logical_bytes;
        out["stored_bytes"] = info.stored_bytes;
        out["layout"] = info.layout;
        out["chunks"] = info.chunks;

        py::list filters;
        for (const auto& filter : info.filters) {
            py::dict item;
            item["id"] = filter.id;
            item["flags"] = filter.flags;
            item["name"] = filter.name;
            item["client_data"] = filter.client_data;
            item["decode_available"] = filter.decode_available;
            item["encode_available"] = filter.encode_available;
            filters.append(std::move(item));
        }
        out["filters"] = std::move(filters);
        return out;
    }

    py::dict matrix_info_to_dict(const actionet::h5ad::MatrixInfo& info) {
        py::dict out;
        out["encoding"] = matrix_encoding_name(info.encoding);
        out["encoding_type"] = info.encoding_type;
        out["encoding_version"] = info.encoding_version;
        out["shape"] = py::make_tuple(info.rows, info.cols);
        out["nnz"] = info.nnz;
        out["data_item_size"] = info.data_item_size;
        out["indices_item_size"] = info.indices_item_size;
        out["indptr_item_size"] = info.indptr_item_size;
        out["chunked"] = info.chunked;
        out["filtered"] = info.filtered;
        out["logical_bytes"] = info.logical_bytes;
        out["stored_bytes"] = info.stored_bytes;
        py::list datasets;
        for (const auto& dataset : info.datasets) {
            datasets.append(dataset_info_to_dict(dataset));
        }
        out["datasets"] = std::move(datasets);
        return out;
    }

    py::dict transfer_stats_to_dict(const actionet::h5ad::TransferStats& stats) {
        py::dict out;
        out["source"] = matrix_info_to_dict(stats.source);
        out["destination"] = matrix_info_to_dict(stats.destination);
        out["selected_source_bytes"] = stats.selected_source_bytes;
        out["source_bytes_read"] = stats.source_bytes_read;
        out["gap_bytes_read"] = stats.gap_bytes_read;
        out["destination_bytes_written"] = stats.destination_bytes_written;
        out["hdf5_read_calls"] = stats.hdf5_read_calls;
        out["hdf5_write_calls"] = stats.hdf5_write_calls;
        out["span_count"] = stats.span_count;
        out["peak_buffer_bytes"] = stats.peak_buffer_bytes;
        out["planning_seconds"] = stats.planning_seconds;
        out["source_read_seconds"] = stats.source_read_seconds;
        out["packing_seconds"] = stats.packing_seconds;
        out["destination_write_seconds"] = stats.destination_write_seconds;
        out["flush_seconds"] = stats.flush_seconds;
        out["destination_fsync_seconds"] = stats.destination_fsync_seconds;

        py::list spans;
        for (const auto& span : stats.spans) {
            py::dict item;
            item["major_start"] = span.major_start;
            item["major_end"] = span.major_end;
            item["selected_major_entries"] = span.selected_major_entries;
            item["source_elements"] = span.source_elements;
            item["source_bytes"] = span.source_bytes;
            item["source_read_seconds"] = span.source_read_seconds;
            item["packing_seconds"] = span.packing_seconds;
            spans.append(std::move(item));
        }
        out["spans"] = std::move(spans);
        return out;
    }

    actionet::h5ad::AxisSelection axis_selection_from_python(
        const py::object& value,
        const char* argument_name) {
        if (value.is_none()) {
            return actionet::h5ad::AxisSelection::all();
        }

        py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> arr(value);
        const py::buffer_info buffer = arr.request();
        if (buffer.ndim != 1) {
            throw std::runtime_error(std::string(argument_name) + " must be a 1D integer array or None");
        }

        const auto* values = static_cast<const std::int64_t*>(buffer.ptr);
        std::vector<std::uint64_t> converted;
        converted.reserve(static_cast<std::size_t>(buffer.shape[0]));
        for (py::ssize_t i = 0; i < buffer.shape[0]; ++i) {
            if (values[i] < 0) {
                throw std::runtime_error(std::string(argument_name) + " cannot contain negative indices");
            }
            converted.push_back(static_cast<std::uint64_t>(values[i]));
        }
        return actionet::h5ad::AxisSelection::from_indices(std::move(converted));
    }

    actionet::h5ad::TransferOptions transfer_options_from_python(
        std::size_t max_buffer_bytes,
        std::size_t gap_merge_bytes,
        std::size_t max_rows_per_batch,
        bool preserve_layout,
        bool collect_span_stats) {
        actionet::h5ad::TransferOptions options;
        options.max_buffer_bytes = max_buffer_bytes;
        options.gap_merge_bytes = gap_merge_bytes;
        options.max_rows_per_batch = max_rows_per_batch;
        options.layout_policy = preserve_layout
            ? actionet::h5ad::LayoutPolicy::Preserve
            : actionet::h5ad::LayoutPolicy::Uncompressed;
        options.collect_span_stats = collect_span_stats;
        return options;
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
    // ``optional_row_scale`` touches Python objects and must run under the
    // GIL; the actual HDF5 open+inspect below can execute without it.
    auto row_scale = optional_row_scale(std::move(row_scale_factors));
    py::gil_scoped_release release;
    return actionet::createBackedOperator(
        file_path,
        group_path,
        static_cast<arma::uword>(std::max(1, chunk_size)),
        std::move(row_scale),
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

              arma::uvec col_indices = int_array_to_uvec(col_indices_arr, "backed index array");

              arma::uvec row_indices;
              if (!row_indices_obj.is_none()) {
                  py::array_t<int64_t> row_arr = row_indices_obj.cast<py::array_t<int64_t>>();
                  row_indices = int_array_to_uvec(row_arr, "backed index array");
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

    m.def("h5ad_inspect_matrix",
          [](const std::string& file_path, const std::string& group_path) {
              actionet::h5ad::MatrixInfo info;
              {
                  py::gil_scoped_release release;
                  info = actionet::h5ad::inspect_matrix(file_path, group_path);
              }
              return matrix_info_to_dict(info);
          },
          "Inspect a version-checked H5AD dense/CSR/CSC matrix",
          py::arg("file_path"), py::arg("group_path"));

    m.def("h5ad_validate_matrix",
          [](const std::string& file_path,
             const std::string& group_path,
             bool full) {
              actionet::h5ad::ValidationReport report;
              {
                  py::gil_scoped_release release;
                  report = actionet::h5ad::validate_matrix(
                      file_path,
                      group_path,
                      full ? actionet::h5ad::ValidationLevel::Full
                           : actionet::h5ad::ValidationLevel::Structural);
              }
              py::dict out;
              out["valid"] = report.valid;
              out["info"] = matrix_info_to_dict(report.info);
              out["indices_scanned"] = report.indices_scanned;
              out["error"] = report.error;
              return out;
          },
          "Validate a version-checked H5AD dense/CSR/CSC matrix",
          py::arg("file_path"), py::arg("group_path"), py::arg("full") = false);

    m.def("h5ad_subset_matrix",
          [](const std::string& source_file_path,
             const std::string& source_group_path,
             const std::string& destination_file_path,
             const std::string& destination_group_path,
             py::object row_indices,
             py::object column_indices,
             std::size_t max_buffer_bytes,
             std::size_t gap_merge_bytes,
             std::size_t max_rows_per_batch,
             bool preserve_layout,
             bool collect_span_stats) {
              const auto rows = axis_selection_from_python(row_indices, "row_indices");
              const auto columns = axis_selection_from_python(column_indices, "column_indices");
              const auto options = transfer_options_from_python(
                  max_buffer_bytes,
                  gap_merge_bytes,
                  max_rows_per_batch,
                  preserve_layout,
                  collect_span_stats);
              actionet::h5ad::TransferStats stats;
              {
                  py::gil_scoped_release release;
                  stats = actionet::h5ad::subset_matrix(
                      source_file_path,
                      source_group_path,
                      destination_file_path,
                      destination_group_path,
                      rows,
                      columns,
                      options);
              }
              return transfer_stats_to_dict(stats);
          },
          "Subset a supported H5AD matrix with the native transfer engine",
          py::arg("source_file_path"),
          py::arg("source_group_path"),
          py::arg("destination_file_path"),
          py::arg("destination_group_path"),
          py::arg("row_indices") = py::none(),
          py::arg("column_indices") = py::none(),
          py::arg("max_buffer_bytes") = 128ULL * 1024ULL * 1024ULL,
          py::arg("gap_merge_bytes") = 64ULL * 1024ULL,
          py::arg("max_rows_per_batch") = 16384,
          py::arg("preserve_layout") = true,
          py::arg("collect_span_stats") = false);

    m.def("h5ad_copy_matrix",
          [](const std::string& source_file_path,
             const std::string& source_group_path,
             const std::string& destination_file_path,
             const std::string& destination_group_path,
             std::size_t max_buffer_bytes,
             std::size_t max_rows_per_batch,
             bool preserve_layout,
             bool collect_span_stats) {
              const auto options = transfer_options_from_python(
                  max_buffer_bytes,
                  64ULL * 1024ULL,
                  max_rows_per_batch,
                  preserve_layout,
                  collect_span_stats);
              actionet::h5ad::TransferStats stats;
              {
                  py::gil_scoped_release release;
                  stats = actionet::h5ad::copy_matrix(
                      source_file_path,
                      source_group_path,
                      destination_file_path,
                      destination_group_path,
                      options);
              }
              return transfer_stats_to_dict(stats);
          },
          "Copy a supported H5AD matrix with the native transfer engine",
          py::arg("source_file_path"),
          py::arg("source_group_path"),
          py::arg("destination_file_path"),
          py::arg("destination_group_path"),
          py::arg("max_buffer_bytes") = 128ULL * 1024ULL * 1024ULL,
          py::arg("max_rows_per_batch") = 16384,
          py::arg("preserve_layout") = true,
          py::arg("collect_span_stats") = false);

    m.def("h5ad_transform_matrix",
          [](const std::string& source_file_path,
             const std::string& source_group_path,
             const std::string& destination_file_path,
             const std::string& destination_group_path,
             py::object row_scale,
             bool apply_log,
             double pseudocount,
             double log_scale,
             const std::string& output_dtype,
             std::size_t max_buffer_bytes,
             std::size_t max_rows_per_batch,
             bool preserve_layout,
             py::object destination_structure_path,
             bool destination_structure_is_exact_copy) {
              actionet::h5ad::TransformOptions options;
              options.row_scale = optional_row_scale(std::move(row_scale));
              options.apply_log = apply_log;
              options.pseudocount = pseudocount;
              options.log_scale = log_scale;
              if (output_dtype == "float32") {
                  options.output_dtype = actionet::h5ad::TransformDType::Float32;
              } else if (output_dtype == "float64") {
                  options.output_dtype = actionet::h5ad::TransformDType::Float64;
              } else {
                  throw std::invalid_argument(
                      "output_dtype must be 'float32' or 'float64'");
              }
              options.transfer = transfer_options_from_python(
                  max_buffer_bytes,
                  64ULL * 1024ULL,
                  max_rows_per_batch,
                  preserve_layout,
                  false);
              if (!destination_structure_path.is_none()) {
                  options.destination_structure_path =
                      destination_structure_path.cast<std::string>();
              }
              options.destination_structure_is_exact_copy =
                  destination_structure_is_exact_copy;
              actionet::h5ad::TransferStats stats;
              {
                  py::gil_scoped_release release;
                  stats = actionet::h5ad::transform_matrix(
                      source_file_path,
                      source_group_path,
                      destination_file_path,
                      destination_group_path,
                      options);
              }
              return transfer_stats_to_dict(stats);
          },
          "Persist a row-scaled/log-transformed H5AD matrix natively",
          py::arg("source_file_path"),
          py::arg("source_group_path"),
          py::arg("destination_file_path"),
          py::arg("destination_group_path"),
          py::arg("row_scale"),
          py::arg("apply_log") = false,
          py::arg("pseudocount") = 1.0,
          py::arg("log_scale") = 1.0,
          py::arg("output_dtype") = "float32",
          py::arg("max_buffer_bytes") = 128ULL * 1024ULL * 1024ULL,
          py::arg("max_rows_per_batch") = 16384,
          py::arg("preserve_layout") = true,
          py::arg("destination_structure_path") = py::none(),
          py::arg("destination_structure_is_exact_copy") = false);
}

#else

void init_io(py::module_&) {}

#endif
