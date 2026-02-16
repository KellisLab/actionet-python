# %%
import actionet as an
import anndata
import scanpy as sc
import numpy as np
import scipy.sparse as sp

import lets_plot as lp
lp.LetsPlot.setup_html()

# %%
adata_ref = anndata.read_h5ad("../data/test_adata.h5ad")
adata_ref.write_h5ad("../data/test_adata_backed.h5ad")
adata = anndata.read_h5ad("../data/test_adata_backed.h5ad", backed="r+")
adata
# %%
an.filter_anndata(adata, min_cells_per_feat=0.01)
adata
# %%
an.normalize_ace(adata, target_sum=1e4, inplace=True)
# %%
an.log1p_ace(adata, base=2, inplace=True)
# %%
an.reduce_kernel(adata, n_components=30, layer=None, key_added="action")
an.run_actionet(adata, k_max=30, inplace=True)
# %%
# an.plot_umap(adata, color="CellLabel")
# %%

markers = an.find_markers(adata, adata.obs["CellLabel"], features_use="Gene", top_genes=10, return_type="dataframe")
markers
# %%
annota_out = an.annotate_cells(adata, markers, method="actionet", features_use="Gene")
# %%
imputed_expression = an.impute_features(adata, features=markers["CT_11"], features_use="Gene")
# %%

# an.plot_umap(adata, color=imputed_expression.iloc[:,0])
# %%
