import io

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

lets_plot = pytest.importorskip("lets_plot")
matplotlib = pytest.importorskip("matplotlib")

from matplotlib.figure import Figure

from actionet.plotting import plot_feature_expression_raster, plot_umap, plot_umap_raster


def _shared_keys(plot) -> set[str]:
    if hasattr(plot, "get_plot_shared_data"):
        shared = plot.get_plot_shared_data()
        if shared is None:
            return set()
        if isinstance(shared, pd.DataFrame):
            return set(shared.columns)
        return set(shared.keys())
    return set(plot.as_dict().get("data", {}).keys())


def _make_adata() -> AnnData:
    x = np.array(
        [
            [1.0, 0.5, 0.0],
            [0.2, 1.1, 0.3],
            [0.7, 0.0, 1.4],
            [0.9, 0.4, 0.1],
            [1.3, 0.8, 0.2],
            [0.4, 1.0, 1.1],
        ],
        dtype=float,
    )
    adata = AnnData(x)
    adata.var_names = pd.Index(["GeneA", "GeneB", "GeneC"])
    adata.obs["cluster"] = pd.Categorical(["a", "a", "b", "b", None, "c"])
    adata.obs["score"] = np.array([0.1, 0.8, np.nan, 0.3, 0.9, 0.5], dtype=float)
    adata.obs["annot_conf"] = np.array([0.2, 0.5, 0.7, 0.1, 0.8, 0.4], dtype=float)
    adata.obsm["umap_2d_actionet"] = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [1.5, 1.0],
            [0.5, 1.5],
            [1.8, 0.2],
            [0.2, 1.8],
        ],
        dtype=float,
    )
    adata.obsm["colors_actionet"] = np.array(
        [
            [0.9, 0.1, 0.1],
            [0.1, 0.7, 0.3],
            [0.2, 0.4, 0.9],
            [0.8, 0.8, 0.1],
            [0.6, 0.3, 0.8],
            [0.2, 0.8, 0.8],
        ],
        dtype=float,
    )
    adata.layers["logcounts"] = x.copy()
    return adata


def test_plot_umap_default_spec_omits_redundant_columns() -> None:
    plot = plot_umap(_make_adata(), color=None, color_source=None, color_slot=None, legend=False)
    spec = plot.as_dict()

    assert _shared_keys(plot) == set()
    assert set(spec["layers"][0]["data"].keys()) == {"x", "y"}
    assert spec["layers"][0]["sampling"] == "none"
    assert spec["layers"][0]["tooltips"] == "none"
    assert "alpha" not in spec["mapping"]


def test_plot_umap_vector_alpha_keeps_alpha_column() -> None:
    alpha = np.linspace(0.2, 1.0, _make_adata().n_obs)
    plot = plot_umap(_make_adata(), color="cluster", alpha=alpha)

    assert "alpha" in _shared_keys(plot)


def test_plot_umap_random_sampling_validation() -> None:
    adata = _make_adata()
    with pytest.raises(ValueError, match="sample_n must be provided"):
        plot_umap(adata, color="cluster", sampling="random")
    with pytest.raises(ValueError, match="positive integer"):
        plot_umap(adata, color="cluster", sampling="random", sample_n=0)


def test_plot_umap_random_sampling_serializes_sampling_spec() -> None:
    plot = plot_umap(
        _make_adata(),
        color="cluster",
        sampling="random",
        sample_n=3,
        tooltips="default",
    )
    layer = plot.as_dict()["layers"][0]

    assert layer["sampling"] != "none"
    assert "tooltips" not in layer or layer["tooltips"] != "none"


def test_plot_umap_continuous_na_layer_has_minimal_data() -> None:
    plot = plot_umap(_make_adata(), color="score", color_source="obs", legend=False)
    spec = plot.as_dict()

    assert len(spec["layers"]) == 2
    assert set(spec["layers"][1]["data"].keys()) == {"x", "y"}


def test_plot_umap_to_png_smoke() -> None:
    lets_plot.LetsPlot.setup_html(no_js=True)
    plot = plot_umap(_make_adata(), color="cluster")
    buf = io.BytesIO()
    plot.to_png(buf)

    assert buf.getbuffer().nbytes > 0


def test_plot_umap_raster_savefig_smoke() -> None:
    fig = plot_umap_raster(_make_adata(), color="score", color_source="obs")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")

    assert isinstance(fig, Figure)
    assert buf.getbuffer().nbytes > 0


def test_plot_feature_expression_raster_returns_figure_for_single_feature() -> None:
    fig = plot_feature_expression_raster(
        _make_adata(),
        features="GeneA",
        alpha=0,
        layer="logcounts",
    )

    assert isinstance(fig, Figure)


def test_plot_feature_expression_raster_single_plot_grid() -> None:
    fig = plot_feature_expression_raster(
        _make_adata(),
        features=["GeneA", "GeneB"],
        alpha=0,
        layer="logcounts",
        single_plot=True,
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 2
