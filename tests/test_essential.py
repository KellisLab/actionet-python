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
# %%
an.filter_anndata(adata, min_cells_per_feat=0.01)
an.normalize_anndata(adata, target_sum=1e4, log_transform=True, log_base=2, inplace=True)
sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
sc.pp.log1p(adata, base=2, copy=False)

# %%
an.reduce_kernel(adata, n_components=30, layer=None, key_added="action")
# an.correct_batch_effect(adata, batch_key=batch_key, reduction_key="action", corrected_suffix="corrected")
an.run_actionet(adata, reduction_key="action", k_max=30, inplace=True)
# %%

markers = an.find_markers(adata, adata.obs["CellLabel"], features_use="Gene", top_genes=10, return_type="dataframe")
markers
# %%
annota_out = an.annotate_cells(adata, markers, method="actionet", features_use="Gene")
# %%
imputed_expression = an.impute_features(adata, features=markers["CT_11"], features_use="Gene")
# %%
an.plot_umap(adata, color=annota_out["labels"])
# %%
for i in range(1, 5):
    p =an.plot_feature_expression(adata, features=markers["CT_11"][i], features_use="Gene", 
    layer=None)
    display(p)
# %%
