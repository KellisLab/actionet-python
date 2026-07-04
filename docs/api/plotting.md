# Plotting

UMAP scatter plots, feature-expression plots, and QC violins. Both interactive
(lets-plot / plotly) and rasterized (matplotlib) variants are provided.

## UMAP

::: actionet.plotting.umap
    options:
      members:
        - plot_umap
        - plot_umap_interactive
        - plot_umap_raster

## Feature expression

::: actionet.plotting.feature_expression
    options:
      members:
        - plot_feature_expression
        - plot_feature_expression_raster

## Quality control

::: actionet.plotting.qc
    options:
      members:
        - get_feature_abundance
        - get_mito_feats
        - plot_mito_violin
        - plot_mito_violin_raster
        - plot_qc_violin
        - plot_qc_violin_raster
