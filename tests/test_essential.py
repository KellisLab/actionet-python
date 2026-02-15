# %%
import actionet as an
import anndata
import scanpy as sc
import numpy as np
import scipy.sparse as sp

# %%
adata = anndata.read_h5ad("../data/test_adata.h5ad")
# %%
# an.filter_anndata(adata, min_cells_per_feat=0.01)
sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
sc.pp.log1p(adata, base=2, copy=False)

# %%
an.reduce_kernel(adata, n_components=30, layer=None, key_added="action")
if "sample" in adata.obs:
    batch_key = "sample"
elif "Sample" in adata.obs:
    batch_key = "Sample"
elif "batch" in adata.obs:
    batch_key = "batch"
else:
    adata.obs["__batch__"] = (np.arange(adata.n_obs) % 2).astype(str)
    batch_key = "__batch__"

an.correct_batch_effect(adata, batch_key=batch_key, reduction_key="action", corrected_suffix="corrected")
an.run_actionet(adata, reduction_key="action_corrected", k_max=30, inplace=True)
# %%

markers = an.find_markers(adata, adata.obs["CellLabel"], features_use="Gene", top_genes=10, return_type="dataframe")
markers
# %%
annota_out = an.annotate_cells(adata, markers, method="actionet", features_use="Gene")
# %%
imputed_expression = an.impute_features(adata, features=markers["CT_11"], features_use="Gene")
# %%
