"""
AnnData I/O utilities for annotation results.

Functions for collecting annotation results from an in-memory AnnData object
and writing them back to backed H5AD files on disk.
"""

import numpy as np
import pandas as pd
import h5py
from pandas.api.types import is_string_dtype
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

    Accepts dense ``np.ndarray`` and ``scipy.sparse`` matrices -- any format
    that AnnData itself allows in ``obsm`` / ``varm``.
    """
    is_sparse = sparse.issparse(array)

    if not is_sparse and not isinstance(array, np.ndarray):
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

    # Collect layers keys
    if layers_keys:
        for key in layers_keys:
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


def coerce_series_to_legacy_strings(s: pd.Series) -> pd.Series:
    """Convert pandas StringDtype series to object for legacy AnnData/anndataR writes."""
    if is_string_dtype(s.dtype):
        return s.astype(object)
    return s


def coerce_legacy_string_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas StringDtype columns/index to object to keep HDF5 writes backward compatible."""
    if is_string_dtype(df.index.dtype):
        df.index = df.index.astype(object)
    str_cols = [c for c in df.columns if is_string_dtype(df[c].dtype)]
    if str_cols:
        df[str_cols] = df[str_cols].astype(object)
    return df


def sanitize_for_legacy_anndata(adata_obj):
    """Normalize obs/var string-like data to be writable by older AnnData/anndataR.

    Converts pandas StringDtype columns and indices to object dtype for compatibility
    with AnnData < 0.11 and anndataR. Using astype(object) instead of astype(str)
    ensures correct conversion even when pandas future.infer_string=True (pandas 3.0+).
    """
    adata_obj.obs = coerce_legacy_string_dtypes(adata_obj.obs)
    adata_obj.var = coerce_legacy_string_dtypes(adata_obj.var)
    adata_obj.obs_names = adata_obj.obs_names.astype(object)
    adata_obj.var_names = adata_obj.var_names.astype(object)


def _coerce_results_strings(results):
    """Coerce pandas nullable string columns in results to object for legacy AnnData/HDF5 writes."""
    for key in ("obs_columns", "var_columns"):
        cols = results.get(key, {})
        for col_name, series in list(cols.items()):
            cols[col_name] = coerce_series_to_legacy_strings(series)
    return results


def _coerce_categorical_categories(categories):
    """Convert categorical categories to an HDF5-safe NumPy array and encoding label."""
    categories_arr = np.asarray(categories)

    # Native numeric/bool categories can be written directly.
    if categories_arr.dtype.kind in ("i", "u", "f", "b"):
        return categories_arr, "array"

    # String-like categories are stored as byte strings for broad HDF5 compatibility.
    if categories_arr.dtype.kind == "U":
        return categories_arr.astype("S"), "string-array"
    if categories_arr.dtype.kind == "S":
        return categories_arr, "string-array"

    # Object/extension/other dtypes (mixed, datetime-like, pandas extension types).
    # Fall back to byte strings to avoid h5py object-dtype failures.
    try:
        return categories_arr.astype("S"), "string-array"
    except (TypeError, ValueError):
        return np.array([str(x) for x in categories_arr], dtype="S"), "string-array"


def append_to_anndata(
    h5_path,
    results,
    verbose=False,
    validate=True
):
    """
    Append annotation results to an H5AD file using direct HDF5 operations.

    This function bypasses AnnData's backed mode limitations by directly
    writing to the HDF5 file structure. Existing keys with the same name
    will be overwritten.

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
    # Check file exists
    import os
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5AD file not found: {h5_path}")

    # Normalize string dtypes for compatibility
    results = _coerce_results_strings(results)

    # Validate data before writing
    if validate:
        _validate_results(h5_path, results, verbose)

    if verbose:
        print(f"[INFO] Writing results to {h5_path}")
    
    with h5py.File(h5_path, 'r+') as f:
        obs_columns = results.get('obs_columns', {})
        var_columns = results.get('var_columns', {})
        obsm_keys = results.get('obsm_keys', {})
        varm_keys = results.get('varm_keys', {})
        obsp_keys = results.get('obsp_keys', {})
        varp_keys = results.get('varp_keys', {})
        layers_keys = results.get('layers_keys', {})
        uns_keys = results.get('uns_keys', {})

        # Write obs columns
        for col, values in obs_columns.items():
            _write_dataframe_column(f, 'obs', col, values, verbose)

        # Update obs column-order attribute
        if obs_columns:
            _update_column_order(f, 'obs')

        # Write var columns
        for col, values in var_columns.items():
            _write_dataframe_column(f, 'var', col, values, verbose)

        # Update var column-order attribute
        if var_columns:
            _update_column_order(f, 'var')

        # Write obsm keys
        for key, array in obsm_keys.items():
            _write_matrix(f, 'obsm', key, array, verbose)

        # Write varm keys
        for key, array in varm_keys.items():
            _write_matrix(f, 'varm', key, array, verbose)

        # Write obsp keys
        for key, matrix in obsp_keys.items():
            _write_matrix(f, 'obsp', key, matrix, verbose)

        # Write varp keys
        for key, matrix in varp_keys.items():
            _write_matrix(f, 'varp', key, matrix, verbose)

        # Write layers keys
        for key, matrix in layers_keys.items():
            _write_matrix(f, 'layers', key, matrix, verbose)

        # Write uns keys
        for key, value in uns_keys.items():
            h5_key = f'uns/{key}'

            # Delete existing key if present
            if h5_key in f:
                del f[h5_key]

            # Handle different data types
            if isinstance(value, pd.DataFrame):
                # Store as structured dataset or group
                _write_dataframe_to_h5(f, h5_key, value)

            elif isinstance(value, pd.Series):
                # Convert to array
                f.create_dataset(h5_key, data=value.values)

            elif isinstance(value, dict):
                # Create group and store each item
                grp = f.create_group(h5_key)
                grp.attrs['encoding-type'] = 'dict'
                grp.attrs['encoding-version'] = '0.1.0'

                for k, v in value.items():
                    _write_dict_value(grp, k, v)

            elif isinstance(value, (np.ndarray, list)):
                f.create_dataset(h5_key, data=value)

            elif isinstance(value, str):
                # String scalars stored as datasets (no compression for scalars)
                ds = f.create_dataset(h5_key, data=value.encode('utf-8'))
                ds.attrs['encoding-type'] = 'string'
                ds.attrs['encoding-version'] = '0.2.0'

            elif isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
                # Numeric/boolean scalars stored as datasets (no compression for scalars)
                ds = f.create_dataset(h5_key, data=value)
                ds.attrs['encoding-type'] = 'numeric-scalar'
                ds.attrs['encoding-version'] = '0.2.0'

            else:
                # Try to convert to array
                try:
                    f.create_dataset(h5_key, data=np.array(value))
                except Exception as e:
                    if verbose:
                        print(f"[WARNING] Could not store uns['{key}']: {e}")
                    continue

            if verbose:
                print(f"[INFO]   Added uns['{key}']")


def _write_dict_value(grp, key, value):
    """
    Write a single value from a dictionary to an HDF5 group.

    Parameters
    ----------
    grp : h5py.Group
        HDF5 group to write to
    key : str
        Key name
    value : any
        Value to write
    """
    if isinstance(value, str):
        # String values
        ds = grp.create_dataset(key, data=value.encode('utf-8'))
        ds.attrs['encoding-type'] = 'string'
        ds.attrs['encoding-version'] = '0.2.0'
    elif isinstance(value, dict):
        # Nested dict - recursively create subgroup
        subgrp = grp.create_group(key)
        subgrp.attrs['encoding-type'] = 'dict'
        subgrp.attrs['encoding-version'] = '0.1.0'
        for k, v in value.items():
            _write_dict_value(subgrp, k, v)
    elif isinstance(value, (list, tuple)):
        # Convert list/tuple to array
        try:
            arr = np.array(value)
            # Check if it's a string array
            if arr.dtype == object or arr.dtype.kind == 'U':
                ds = grp.create_dataset(key, data=arr.astype('S'))
                ds.attrs['encoding-type'] = 'string-array'
            else:
                ds = grp.create_dataset(key, data=arr)
                ds.attrs['encoding-type'] = 'array'
            ds.attrs['encoding-version'] = '0.2.0'
        except:
            # If array conversion fails, skip
            pass
    elif isinstance(value, np.ndarray):
        # NumPy array
        if value.dtype == object or value.dtype.kind == 'U':
            ds = grp.create_dataset(key, data=value.astype('S'))
            ds.attrs['encoding-type'] = 'string-array'
        else:
            ds = grp.create_dataset(key, data=value)
            ds.attrs['encoding-type'] = 'array'
        ds.attrs['encoding-version'] = '0.2.0'
    elif isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
        # Numeric/boolean scalar
        ds = grp.create_dataset(key, data=value)
        ds.attrs['encoding-type'] = 'numeric-scalar'
        ds.attrs['encoding-version'] = '0.2.0'
    elif value is None:
        # None values - skip (can't store in HDF5)
        pass
    else:
        # Try to convert to array as last resort
        try:
            ds = grp.create_dataset(key, data=value)
            ds.attrs['encoding-type'] = 'array'
            ds.attrs['encoding-version'] = '0.2.0'
        except:
            # If all else fails, skip
            pass


def _update_column_order(f, df_name):
    """
    Update the column-order attribute for obs or var DataFrame.

    AnnData stores a 'column-order' attribute in obs/var groups that lists
    which columns should be loaded. This function updates it to include all
    columns present in the group (excluding '_index').

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle
    df_name : str
        Name of the dataframe ('obs' or 'var')
    """
    if df_name not in f:
        return

    grp = f[df_name]

    # Get all keys except _index
    columns = [k for k in grp.keys() if k != '_index']

    # Update column-order attribute
    grp.attrs['column-order'] = np.array(columns, dtype=object)


def _write_dataframe_column(f, df_name, col, values, verbose):
    """
    Write a single column to an obs or var dataframe in HDF5.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle
    df_name : str
        Name of the dataframe ('obs' or 'var')
    col : str
        Column name
    values : array-like
        Column values (pd.Series, np.ndarray, list, etc.)
    verbose : bool
        Print progress messages
    """
    key = f'{df_name}/{col}'

    # Coerce to pd.Series so downstream .cat / .dtype / .values work uniformly.
    if not isinstance(values, pd.Series):
        values = pd.Series(values)

    # Delete existing key if present
    if key in f:
        del f[key]

    if hasattr(values, 'cat'):
        # Categorical data - store as group with categories and codes
        categories = values.cat.categories.values
        codes = values.cat.codes.values

        # Create group for categorical data
        cat_grp = f.create_group(key)
        cat_grp.attrs['encoding-type'] = 'categorical'
        cat_grp.attrs['encoding-version'] = '0.2.0'
        cat_grp.attrs['ordered'] = values.cat.ordered

        # Store codes
        codes_ds = cat_grp.create_dataset('codes', data=codes)
        codes_ds.attrs['encoding-type'] = 'array'
        codes_ds.attrs['encoding-version'] = '0.2.0'

        # Store categories
        categories_h5, encoding_type = _coerce_categorical_categories(categories)
        cat_ds = cat_grp.create_dataset('categories', data=categories_h5)
        cat_ds.attrs['encoding-type'] = encoding_type
        cat_ds.attrs['encoding-version'] = '0.2.0'
        if verbose:
            print(f"[INFO]   Added {df_name}['{col}']")
        return

    values = coerce_series_to_legacy_strings(values)

    if values.dtype == object or values.dtype.kind == 'U':
        # String data - convert to fixed-length bytes
        str_values = values.astype(str).values
        ds = f.create_dataset(key, data=str_values.astype('S'))
        ds.attrs['encoding-type'] = 'string-array'
        ds.attrs['encoding-version'] = '0.2.0'

    else:
        # Numeric data
        ds = f.create_dataset(key, data=values.values)
        ds.attrs['encoding-type'] = 'array'
        ds.attrs['encoding-version'] = '0.2.0'

    if verbose:
        print(f"[INFO]   Added {df_name}['{col}']")


def _write_matrix(f, container_name, key, matrix, verbose, compression_kwargs=None):
    """
    Write a matrix to obsm, varm, obsp, varp, or layers in HDF5.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle
    container_name : str
        Name of the container ('obsm', 'varm', 'obsp', 'varp', 'layers')
    key : str
        Key name for the matrix
    matrix : np.ndarray or sparse matrix
        Matrix to store
    verbose : bool
        Print progress messages
    compression_kwargs : dict or None
        Optional h5py create_dataset compression kwargs (e.g.
        ``{"compression": "gzip", "compression_opts": 4}``).  When ``None``
        (the default) no compression is applied, matching AnnData's own
        default behaviour and avoiding the read-time decompression overhead
        that caused the backed-reopen slowdown.
    """
    if compression_kwargs is None:
        compression_kwargs = {}

    h5_key = f'{container_name}/{key}'

    # Preserve existing compression from the on-disk dataset when no explicit
    # policy was provided by the caller.
    if not compression_kwargs and h5_key in f:
        existing = f[h5_key]
        # Sparse group: look at the 'data' sub-dataset.
        if hasattr(existing, 'keys') and 'data' in existing:
            ds_ref = existing['data']
        else:
            ds_ref = existing if hasattr(existing, 'compression') else None
        if ds_ref is not None and getattr(ds_ref, 'compression', None) is not None:
            compression_kwargs = {'compression': ds_ref.compression}
            if ds_ref.compression_opts is not None:
                compression_kwargs['compression_opts'] = ds_ref.compression_opts

    # Delete existing key if present
    if h5_key in f:
        del f[h5_key]

    # Handle sparse matrices
    if sparse.issparse(matrix):
        # Convert to CSR format for storage
        matrix = matrix.tocsr()

        # Create group for sparse matrix
        grp = f.create_group(h5_key)
        grp.create_dataset('data', data=matrix.data, **compression_kwargs)
        grp.create_dataset('indices', data=matrix.indices, **compression_kwargs)
        grp.create_dataset('indptr', data=matrix.indptr, **compression_kwargs)
        grp.attrs['shape'] = matrix.shape
        grp.attrs['encoding-type'] = 'csr_matrix'
        grp.attrs['encoding-version'] = '0.1.0'

    else:
        # Dense matrix — ensure C-contiguous layout before writing.
        # Armadillo returns Fortran-order arrays; h5py's gzip path is
        # dramatically slower when it must internally relayout F→C order.
        matrix = np.ascontiguousarray(matrix)

        _CHUNK_WRITE_BYTES = 50 * 1024 * 1024  # 50 MB
        if matrix.ndim == 2 and matrix.nbytes > _CHUNK_WRITE_BYTES:
            chunk_rows = max(
                1, _CHUNK_WRITE_BYTES // (matrix.shape[1] * matrix.dtype.itemsize)
            )
            ds = f.create_dataset(
                h5_key, shape=matrix.shape, dtype=matrix.dtype,
                **compression_kwargs,
            )
            for start in range(0, matrix.shape[0], chunk_rows):
                ds[start:start + chunk_rows] = matrix[start:start + chunk_rows]
        else:
            ds = f.create_dataset(h5_key, data=matrix, **compression_kwargs)

        ds.attrs['encoding-type'] = 'array'
        ds.attrs['encoding-version'] = '0.2.0'

    if verbose:
        print(f"[INFO]   Added {container_name}['{key}']")


def _write_dataframe_to_h5(f, key, df):
    """
    Write a pandas DataFrame to an HDF5 file in AnnData-compatible format.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle
    key : str
        Key path where DataFrame should be stored
    df : pd.DataFrame
        DataFrame to store
    """
    grp = f.create_group(key)

    # Set encoding attributes
    grp.attrs['encoding-type'] = 'dataframe'
    grp.attrs['encoding-version'] = '0.2.0'
    grp.attrs['_index'] = '_index'
    grp.attrs['column-order'] = np.array(df.columns.tolist(), dtype='S') if len(df.columns) else np.array([], dtype='S')

    # Store index - convert to string array for HDF5
    index_values = np.array([str(x) for x in df.index], dtype='S')
    ds = grp.create_dataset('_index', data=index_values)
    ds.attrs['encoding-type'] = 'string-array'
    ds.attrs['encoding-version'] = '0.2.0'

    # Store each column
    for col in df.columns:
        values = df[col]
        col_key = f'{key}/{col}'

        if hasattr(values, 'cat'):
            # Categorical column
            categories = values.cat.categories.values
            codes = values.cat.codes.values

            col_grp = grp.create_group(col)
            col_grp.attrs['encoding-type'] = 'categorical'
            col_grp.attrs['encoding-version'] = '0.2.0'
            col_grp.attrs['ordered'] = values.cat.ordered

            codes_ds = col_grp.create_dataset('codes', data=codes)
            codes_ds.attrs['encoding-type'] = 'array'
            codes_ds.attrs['encoding-version'] = '0.2.0'

            categories_h5, encoding_type = _coerce_categorical_categories(categories)
            cat_ds = col_grp.create_dataset('categories', data=categories_h5)
            cat_ds.attrs['encoding-type'] = encoding_type
            cat_ds.attrs['encoding-version'] = '0.2.0'

        else:
            values = coerce_series_to_legacy_strings(values)

            if values.dtype == object or values.dtype.kind == 'U':
                # String column
                ds = f.create_dataset(col_key, data=values.astype('S'))
                ds.attrs['encoding-type'] = 'string-array'
                ds.attrs['encoding-version'] = '0.2.0'
            else:
                # Numeric column
                ds = f.create_dataset(col_key, data=values.values)
                ds.attrs['encoding-type'] = 'array'
                ds.attrs['encoding-version'] = '0.2.0'
