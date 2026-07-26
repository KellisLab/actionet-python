"""
AnnData I/O utilities for annotation results.

Functions for collecting annotation results from an in-memory AnnData object
and writing them back to backed H5AD files on disk.
"""

import numpy as np
import pandas as pd
import h5py
from scipy import sparse
import warnings

class ValidationError(Exception):
    """Raised when data validation fails."""
    pass


def _validate_results(h5_path, results, verbose=False):
    """
    Validate results before writing to H5AD file.

    Checks:
    - Data shapes match file dimensions
    - Data types are supported
    - Required structure is present
    - No corrupted data

    Parameters
    ----------
    h5_path : str
        Path to H5AD file
    results : dict
        Results dictionary to validate
    verbose : bool
        Print validation messages

    Raises
    ------
    ValidationError
        If validation fails
    """
    if verbose:
        print("[INFO] Validating data before writing...")

    # Open file to check dimensions
    try:
        with h5py.File(h5_path, 'r') as f:
            # Get dimensions
            if 'X' in f:
                if isinstance(f['X'], h5py.Group):
                    # Sparse matrix
                    n_obs, n_vars = f['X'].attrs['shape']
                else:
                    # Dense matrix
                    n_obs, n_vars = f['X'].shape
            elif 'obs' in f and 'var' in f:
                n_obs = len(f['obs']['_index'])
                n_vars = len(f['var']['_index'])
            else:
                raise ValidationError("Cannot determine AnnData dimensions from file")

            if verbose:
                print(f"[INFO]   File dimensions: n_obs={n_obs}, n_vars={n_vars}")

            # Validate obs columns
            for col, values in results.get('obs_columns', {}).items():
                _validate_obs_var_column(col, values, n_obs, 'obs', verbose)

            # Validate var columns
            for col, values in results.get('var_columns', {}).items():
                _validate_obs_var_column(col, values, n_vars, 'var', verbose)

            # Validate obsm
            for key, array in results.get('obsm_keys', {}).items():
                _validate_matrix(key, array, n_obs, 'obsm', 'obs', verbose)

            # Validate varm
            for key, array in results.get('varm_keys', {}).items():
                _validate_matrix(key, array, n_vars, 'varm', 'var', verbose)

            # Validate obsp
            for key, matrix in results.get('obsp_keys', {}).items():
                _validate_pairwise_matrix(key, matrix, n_obs, 'obsp', verbose)

            # Validate varp
            for key, matrix in results.get('varp_keys', {}).items():
                _validate_pairwise_matrix(key, matrix, n_vars, 'varp', verbose)

            # Validate layers
            for key, matrix in results.get('layers_keys', {}).items():
                _validate_layer(key, matrix, n_obs, n_vars, verbose)

            # Validate uns (basic checks)
            for key, value in results.get('uns_keys', {}).items():
                _validate_uns_value(key, value, verbose)

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        else:
            raise ValidationError(f"Validation failed: {e}")

    if verbose:
        print("[INFO] ✓ Validation passed")


def _validate_obs_var_column(col, values, expected_len, df_name, verbose):
    """Validate an obs or var column.

    Accepts any 1-D array-like that can serve as a DataFrame column:
    ``pd.Series``, ``np.ndarray``, ``list``, or similar.
    """
    # Coerce to Series so downstream checks (.isna(), .dtype) work uniformly.
    if not isinstance(values, pd.Series):
        try:
            values = pd.Series(values)
        except Exception:
            raise ValidationError(
                f"{df_name}['{col}']: Cannot coerce to pandas Series "
                f"(type={type(values).__name__})"
            )

    if len(values) != expected_len:
        raise ValidationError(
            f"{df_name}['{col}']: Length mismatch. Expected {expected_len}, got {len(values)}"
        )

    # Check for None values (can't store in HDF5)
    if values.isna().any() and values.dtype == object:
        n_na = values.isna().sum()
        raise ValidationError(
            f"{df_name}['{col}']: Contains {n_na} None/NaN values in object column. "
            "Convert to appropriate dtype or fill missing values."
        )

    if verbose:
        print(f"[INFO]   ✓ {df_name}['{col}']: {len(values)} values, dtype={values.dtype}")


def _validate_matrix(key, array, expected_first_dim, container_name, axis_name, verbose):
    """Validate obsm, varm, or similar matrix.

    Accepts dense ``np.ndarray`` and ``scipy.sparse`` matrices. For
    ``obsm``/``varm`` payloads, ``pd.DataFrame`` is also accepted and will be
    coerced via ``np.asarray`` for shape/value validation.
    """
    is_sparse = sparse.issparse(array)

    if not is_sparse and not isinstance(array, np.ndarray):
        if isinstance(array, pd.DataFrame):
            # Validate shape directly on the DataFrame, then run per-column
            # NaN/inf checks so error messages include the offending column name.
            if array.shape[0] != expected_first_dim:
                raise ValidationError(
                    f"{container_name}['{key}']: First dimension must match "
                    f"n_{axis_name}={expected_first_dim}, got {array.shape[0]}"
                )
            for col in array.columns:
                col_vals = array[col]
                if pd.api.types.is_float_dtype(col_vals):
                    if col_vals.isna().any():
                        raise ValidationError(
                            f"{container_name}['{key}']: Column '{col}' contains NaN values"
                        )
                    if np.isinf(col_vals).any():
                        raise ValidationError(
                            f"{container_name}['{key}']: Column '{col}' contains infinite values"
                        )
            if verbose:
                print(f"[INFO]   ✓ {container_name}['{key}']: DataFrame, "
                      f"shape={array.shape}, columns={list(array.columns)}")
            return

        # Attempt coercion; AnnData occasionally stores pd.DataFrame in obsm.
        try:
            array = np.asarray(array)
        except Exception:
            raise ValidationError(
                f"{container_name}['{key}']: Must be numpy array or sparse matrix, "
                f"got {type(array)}"
            )

    if array.ndim < 1 or array.ndim > 3:
        raise ValidationError(
            f"{container_name}['{key}']: Array must be 1D, 2D, or 3D, got {array.ndim}D"
        )

    if array.shape[0] != expected_first_dim:
        raise ValidationError(
            f"{container_name}['{key}']: First dimension must match n_{axis_name}={expected_first_dim}, "
            f"got {array.shape[0]}"
        )

    # Check for invalid values (dense only; sparse data checked via .data attribute)
    if is_sparse:
        if np.issubdtype(array.data.dtype, np.floating):
            if np.any(np.isnan(array.data)):
                raise ValidationError(
                    f"{container_name}['{key}']: Sparse data contains NaN values"
                )
            if np.any(np.isinf(array.data)):
                raise ValidationError(
                    f"{container_name}['{key}']: Sparse data contains infinite values"
                )
    else:
        if np.issubdtype(array.dtype, np.floating):
            if np.any(np.isnan(array)):
                raise ValidationError(
                    f"{container_name}['{key}']: Contains NaN values"
                )
            if np.any(np.isinf(array)):
                raise ValidationError(
                    f"{container_name}['{key}']: Contains infinite values"
                )

    if verbose:
        if is_sparse:
            print(f"[INFO]   ✓ {container_name}['{key}']: sparse {array.format}, "
                  f"shape={array.shape}, nnz={array.nnz}")
        else:
            print(f"[INFO]   ✓ {container_name}['{key}']: shape={array.shape}, dtype={array.dtype}")


def _validate_pairwise_matrix(key, matrix, expected_dim, container_name, verbose):
    """Validate obsp or varp pairwise matrix."""
    is_sparse = sparse.issparse(matrix)

    if not is_sparse and not isinstance(matrix, np.ndarray):
        raise ValidationError(
            f"{container_name}['{key}']: Must be numpy array or sparse matrix, got {type(matrix)}"
        )

    if matrix.shape != (expected_dim, expected_dim):
        raise ValidationError(
            f"{container_name}['{key}']: Must be square matrix ({expected_dim}, {expected_dim}), "
            f"got {matrix.shape}"
        )

    # Check for invalid values in dense matrices
    if not is_sparse and np.issubdtype(matrix.dtype, np.floating):
        if np.any(np.isnan(matrix)):
            raise ValidationError(
                f"{container_name}['{key}']: Contains NaN values"
            )
        if np.any(np.isinf(matrix)):
            raise ValidationError(
                f"{container_name}['{key}']: Contains infinite values"
            )

    # Check sparse matrix
    if is_sparse:
        if matrix.format not in ['csr', 'csc', 'coo']:
            warnings.warn(
                f"{container_name}['{key}']: Sparse format '{matrix.format}' will be converted to CSR"
            )

    if verbose:
        if is_sparse:
            print(f"[INFO]   ✓ {container_name}['{key}']: sparse {matrix.format}, "
                  f"shape={matrix.shape}, nnz={matrix.nnz}")
        else:
            print(f"[INFO]   ✓ {container_name}['{key}']: dense, shape={matrix.shape}, dtype={matrix.dtype}")


def _validate_layer(key, matrix, n_obs, n_vars, verbose):
    """Validate layer matrix."""
    is_sparse = sparse.issparse(matrix)

    if not is_sparse and not isinstance(matrix, np.ndarray):
        raise ValidationError(
            f"layers['{key}']: Must be numpy array or sparse matrix, got {type(matrix)}"
        )

    if matrix.shape != (n_obs, n_vars):
        raise ValidationError(
            f"layers['{key}']: Shape must be ({n_obs}, {n_vars}), got {matrix.shape}"
        )

    # Check for invalid values in dense matrices
    if not is_sparse and np.issubdtype(matrix.dtype, np.floating):
        if np.any(np.isnan(matrix)):
            raise ValidationError(
                f"layers['{key}']: Contains NaN values"
            )
        if np.any(np.isinf(matrix)):
            raise ValidationError(
                f"layers['{key}']: Contains infinite values"
            )

    if verbose:
        if is_sparse:
            print(f"[INFO]   ✓ layers['{key}']: sparse {matrix.format}, "
                  f"shape={matrix.shape}, nnz={matrix.nnz}")
        else:
            print(f"[INFO]   ✓ layers['{key}']: dense, shape={matrix.shape}, dtype={matrix.dtype}")


def _validate_uns_value(key, value, verbose):
    """Validate uns value."""
    # Check supported types
    supported_types = (
        str, int, float, bool,
        np.integer, np.floating, np.bool_,
        np.ndarray, list, tuple, dict,
        pd.DataFrame, pd.Series
    )

    if not isinstance(value, supported_types):
        raise ValidationError(
            f"uns['{key}']: Unsupported type {type(value)}. "
            f"Must be one of: scalar, array, list, dict, DataFrame, Series"
        )

    # Check for None (can't store)
    if value is None:
        warnings.warn(f"uns['{key}']: None values cannot be stored in HDF5 and will be skipped")

    # Validate arrays
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            warnings.warn(
                f"uns['{key}']: Object dtype arrays may not store correctly. "
                "Consider converting to specific dtype."
            )
        if np.issubdtype(value.dtype, np.floating):
            if np.any(np.isnan(value)):
                raise ValidationError(f"uns['{key}']: Array contains NaN values")
            if np.any(np.isinf(value)):
                raise ValidationError(f"uns['{key}']: Array contains infinite values")

    # Validate DataFrame
    if isinstance(value, pd.DataFrame):
        if value.empty:
            warnings.warn(f"uns['{key}']: Empty DataFrame")
        # Check for object columns with None
        for col in value.columns:
            if value[col].dtype == object and value[col].isna().any():
                warnings.warn(
                    f"uns['{key}']: DataFrame column '{col}' contains None/NaN in object column"
                )

    # Recursively validate dict values
    if isinstance(value, dict):
        for k, v in value.items():
            if v is None:
                warnings.warn(f"uns['{key}']['{k}']: None values will be skipped")
            elif isinstance(v, dict):
                _validate_uns_value(f"{key}.{k}", v, verbose=False)

    if verbose:
        if isinstance(value, dict):
            print(f"[INFO]   ✓ uns['{key}']: dict with {len(value)} keys")
        elif isinstance(value, (np.ndarray, list)):
            if isinstance(value, list):
                value = np.array(value)
            print(f"[INFO]   ✓ uns['{key}']: array, shape={value.shape if hasattr(value, 'shape') else len(value)}")
        elif isinstance(value, pd.DataFrame):
            print(f"[INFO]   ✓ uns['{key}']: DataFrame, shape={value.shape}")
        else:
            print(f"[INFO]   ✓ uns['{key}']: {type(value).__name__}")


def collect_annotation_results(
    adata,
    obs_columns=None,
    var_columns=None,
    obsm_keys=None,
    varm_keys=None,
    obsp_keys=None,
    varp_keys=None,
    layers_keys=None,
    uns_keys=None,
    verbose=False
):
    """
    Collect annotation results from an AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object containing results
    obs_columns : list of str, optional
        Column names to collect from adata.obs
    var_columns : list of str, optional
        Column names to collect from adata.var
    obsm_keys : list of str, optional
        Keys to collect from adata.obsm
    varm_keys : list of str, optional
        Keys to collect from adata.varm
    obsp_keys : list of str, optional
        Keys to collect from adata.obsp
    varp_keys : list of str, optional
        Keys to collect from adata.varp
    layers_keys : list of str, optional
        Keys to collect from adata.layers
    uns_keys : list of str, optional
        Keys to collect from adata.uns
    verbose : bool, optional
        Print progress messages
    
    Returns
    -------
    dict
        Dictionary with keys 'obs_columns', 'var_columns', 'obsm_keys', 'varm_keys',
        'obsp_keys', 'varp_keys', 'layers_keys', 'uns_keys' containing the collected data
    """
    results = {
        'obs_columns': {},
        'var_columns': {},
        'obsm_keys': {},
        'varm_keys': {},
        'obsp_keys': {},
        'varp_keys': {},
        'layers_keys': {},
        'uns_keys': {}
    }
    
    # Collect obs columns
    if obs_columns:
        for col in obs_columns:
            if col in adata.obs.columns:
                results['obs_columns'][col] = adata.obs[col]
                if verbose:
                    print(f"[INFO]   Collected obs['{col}']")
    
    # Collect var columns
    if var_columns:
        for col in var_columns:
            if col in adata.var.columns:
                results['var_columns'][col] = adata.var[col]
                if verbose:
                    print(f"[INFO]   Collected var['{col}']")

    # Collect obsm keys
    if obsm_keys:
        for key in obsm_keys:
            if key in adata.obsm.keys():
                results['obsm_keys'][key] = adata.obsm[key]
                if verbose:
                    print(f"[INFO]   Collected obsm['{key}']")
    
    # Collect varm keys
    if varm_keys:
        for key in varm_keys:
            if key in adata.varm.keys():
                results['varm_keys'][key] = adata.varm[key]
                if verbose:
                    print(f"[INFO]   Collected varm['{key}']")

    # Collect obsp keys
    if obsp_keys:
        for key in obsp_keys:
            if key in adata.obsp.keys():
                results['obsp_keys'][key] = adata.obsp[key]
                if verbose:
                    print(f"[INFO]   Collected obsp['{key}']")
    
    # Collect varp keys
    if varp_keys:
        for key in varp_keys:
            if key in adata.varp.keys():
                results['varp_keys'][key] = adata.varp[key]
                if verbose:
                    print(f"[INFO]   Collected varp['{key}']")

    # Collect layers keys. anndata >= 0.13 exposes an alias ``layers[None]``
    # that points back to ``.X`` so any caller iterating over ``layers.keys()``
    # sees a spurious ``None`` entry. Skip it here (the caller should read
    # ``.X`` explicitly if they want the primary matrix).
    if layers_keys:
        for key in layers_keys:
            if key is None:
                continue
            if key in adata.layers.keys():
                results['layers_keys'][key] = adata.layers[key]
                if verbose:
                    print(f"[INFO]   Collected layers['{key}']")

    # Collect uns keys
    if uns_keys:
        for key in uns_keys:
            if key in adata.uns.keys():
                results['uns_keys'][key] = adata.uns[key]
                if verbose:
                    print(f"[INFO]   Collected uns['{key}']")
    
    return results


def append_to_anndata(
    h5_path,
    results,
    verbose=False,
    validate=True,
    chunk_size=None,
):
    """
    Append annotation results through the shared atomic H5AD rewrite.

    Numeric matrices are copied by the native data plane. Updated values use
    AnnData's public codec, then the completed temporary file is validated,
    synced, and atomically replaces the original.

    Parameters
    ----------
    h5_path : str
        Path to H5AD file to update
    results : dict
        Dictionary of results from collect_annotation_results()
    verbose : bool, optional
        Print progress messages
    validate : bool, optional
        Validate data before writing (default: True)
    chunk_size : int or None, optional
        Row/element chunk size for the full-file payload copy. When ``None``
        the shared default (:data:`DEFAULT_BACKED_WRITE_CHUNK_SIZE`) is used.

    Returns
    -------
    None

    Raises
    ------
    ValidationError
        If validation fails and validate=True
    FileNotFoundError
        If h5_path does not exist
    IOError
        If file cannot be opened for writing
    """
    import os
    import anndata as ad

    from .checkpoint import rewrite_h5ad_payload
    from .chunking import DEFAULT_BACKED_WRITE_CHUNK_SIZE
    from .rewrite import RewriteTransaction

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5AD file not found: {h5_path}")

    if chunk_size is None:
        chunk_size = DEFAULT_BACKED_WRITE_CHUNK_SIZE

    if validate:
        _validate_results(h5_path, results, verbose)

    if verbose:
        print(f"[INFO] Writing results to {h5_path}")

    obs_columns = results.get('obs_columns', {})
    var_columns = results.get('var_columns', {})
    obsm_keys = results.get('obsm_keys', {})
    varm_keys = results.get('varm_keys', {})
    obsp_keys = results.get('obsp_keys', {})
    varp_keys = results.get('varp_keys', {})
    layers_keys = results.get('layers_keys', {})
    uns_keys = results.get('uns_keys', {})

    matrix_mappings = {
        'obsm': obsm_keys,
        'varm': varm_keys,
        'obsp': obsp_keys,
        'varp': varp_keys,
        'layers': layers_keys,
    }

    source_path = os.path.realpath(os.fspath(h5_path))
    with RewriteTransaction(source_path, source_path) as transaction:
        rewrite_h5ad_payload(
            source_path,
            transaction.temp_path,
            chunk_size=chunk_size,
        )
        with h5py.File(transaction.temp_path, 'r+') as destination:
            for frame_name, updated_columns in (
                ('obs', obs_columns),
                ('var', var_columns),
            ):
                if not updated_columns:
                    continue
                frame = ad.io.read_elem(destination[frame_name])
                for column, values in updated_columns.items():
                    frame[column] = values
                del destination[frame_name]
                ad.io.write_elem(destination, frame_name, frame)

            for container_name, updated_mapping in matrix_mappings.items():
                if not updated_mapping:
                    continue
                if container_name not in destination:
                    ad.io.write_elem(destination, container_name, {})
                group = destination[container_name]
                for key, value in updated_mapping.items():
                    if key in group:
                        del group[key]
                    ad.io.write_elem(group, key, value)
                    if verbose:
                        print(f"[INFO]   Added {container_name}['{key}']")

            if uns_keys:
                if 'uns' not in destination:
                    ad.io.write_elem(destination, 'uns', {})
                uns_group = destination['uns']
                for key, value in uns_keys.items():
                    if key in uns_group:
                        del uns_group[key]
                    ad.io.write_elem(uns_group, key, value)
                    if verbose:
                        print(f"[INFO]   Added uns['{key}']")

            destination.flush()

        validated = ad.read_h5ad(transaction.temp_path, backed='r')
        file_handle = getattr(validated, "file", None)
        if file_handle is not None:
            try:
                file_handle.close()
            except Exception:
                pass
        transaction.commit()
