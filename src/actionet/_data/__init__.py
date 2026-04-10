"""Internal package data loader for ACTIONet."""

from __future__ import annotations

import importlib.resources
from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=1)
def load_mito_features() -> pd.DataFrame:
    """Load the bundled mitochondrial feature table.

    Returns a DataFrame with columns:
    ``ensembl_id``, ``gene_name``, ``chromosome``, ``feature_type``, ``species``.

    The table covers human (``hsapiens``) and mouse (``mmusculus``) and is
    sourced from the ``mito_human_mouse_biomart_240924`` dataset in the R
    ``actionet`` package (``R/sysdata.rda``).
    """
    pkg = importlib.resources.files(__package__)
    csv_path = pkg / "mito_features.csv"
    with importlib.resources.as_file(csv_path) as path:
        return pd.read_csv(path, dtype=str)
