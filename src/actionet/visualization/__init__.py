"""Visualization subpackage: network layout and plotting."""

from .feature_expression import (
    plot_feature_expression,
    plot_feature_expression_raster,
)
from .layout import compute_node_colors, layout_network
from .qc import (
    get_feature_abundance,
    get_mito_feats,
    plot_mito_violin,
    plot_mito_violin_raster,
    plot_qc_violin,
    plot_qc_violin_raster,
)
from .umap import plot_umap, plot_umap_interactive, plot_umap_raster
from .utils import register_figure_display_formatter

register_figure_display_formatter()

__all__ = [
    "compute_node_colors",
    "get_feature_abundance",
    "get_mito_feats",
    "layout_network",
    "plot_feature_expression",
    "plot_feature_expression_raster",
    "plot_mito_violin",
    "plot_mito_violin_raster",
    "plot_qc_violin",
    "plot_qc_violin_raster",
    "plot_umap",
    "plot_umap_interactive",
    "plot_umap_raster",
    "register_figure_display_formatter",
]
