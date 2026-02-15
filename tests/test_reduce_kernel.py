# %%
import actionet as an
import anndata
import scanpy as sc
import numpy as np
import scipy.sparse as sp

import lets_plot as lp
lp.LetsPlot.setup_html()

# %%
adata = anndata.read_h5ad("../data/test_adata.h5ad")
adata
# %%
an.filter_anndata(adata, min_cells_per_feat=0.01)
sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
sc.pp.log1p(adata, base=2, copy=False)

# %%
# svd_result = an.run_svd(adata.X, n_components=30)
# an.reduce_kernel_from_svd(adata, svd_result=svd_result, layer=None, key_added="action")
# %%
an.reduce_kernel(adata, n_components=30, layer=None, key_added="action")

# %%
an.run_actionet(adata, k_max=30, inplace=True)
# %%
an.plot_umap(adata, color="CellLabel")
# %%
