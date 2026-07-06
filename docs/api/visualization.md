# Visualization

Layout / color-assignment helpers plus all plotting variants (interactive
lets-plot / plotly, and rasterized matplotlib).

## Layout and node colors

::: actionet.visualization.layout
    options:
      members:
        - layout_network
        - compute_node_colors

## UMAP

::: actionet.visualization.umap
    options:
      members:
        - plot_umap
        - plot_umap_interactive
        - plot_umap_raster

## Feature expression

::: actionet.visualization.feature_expression
    options:
      members:
        - plot_feature_expression
        - plot_feature_expression_raster

## Quality control

::: actionet.visualization.qc
    options:
      members:
        - get_feature_abundance
        - get_mito_feats
        - plot_mito_violin
        - plot_mito_violin_raster
        - plot_qc_violin
        - plot_qc_violin_raster
