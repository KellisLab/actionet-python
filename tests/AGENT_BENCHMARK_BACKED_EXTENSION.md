# Benchmark actionet-python backed extension

Background: A major refactoring and feature extension of this package was just completed. Please review the context files.

Task: Please generate and execute a series of tests comparing the performance of running a typicaly workflow in-memory vs backed. Test any other functionality you deem appropriate. Run multiple trials.

## Datasets

Use two datasets for testing:

1. `data/test_adata.h5ad` contains aa small singular sample.
2. `data/adata_agg_Hm_STR_MSN_1000plus_only_processed.h5ad` contains a larger dataset with multiple samples. This one requires using batch correction. Use .obs['UID] as the batch labels.

## Implementation

Use the following workflow for testing:

1. Filter the data to remove genes that are not expressed in at least 1% of cells. Do not filter cells.
2. Normalize and log transform the data to remove library size effects. Scale to 10,000 counts and use log base 2.
3. Reduce the data to a lower dimensional space using the kernel reduction. Use PRIMME for all tests. Keep additional parameters as default.
4. Correct the batch effects using the batch correction (when applicable).
5. Run the core pipeline components separately (ACTION decomposition, network construction, archetype diffusion, UMAP generation, color computation, feature specificity) for profiling purposes. Remember to use the approriate reduction key if preceded by batch correction.
6. Find the top 30 markers for each cell type. Use .obs['CellLabel'] for dataset (1) and .obs['CellType'] for dataset (2) as the cell type labels. Use .var['Gene'] as features.
7. Annotate the cells using the cell type annotation. Use the vision method with the default parameters. Use .var['Gene'] as features.
8. Impute 10 arbitrary features for each dataset. Use .var['Gene'] as features.

## Evaluation

Use the following metrics for testing:

1. The runtime of each major component and the total runtime of the workflow.
2. The peak memory usage of each major component and the peak memory usage of the workflow.
3. Result parity between in-memory and backed modes.

With these results and knowledge of the computational complexity of the workflow, estimate the runtime and memory usage for datasets with:

1. 100k, 1M, and 10M cells expressing an average of 10,000 genes.
2. 1, 5, 25, 50, and 100 batches.

## Output

Generate a comprehensive report of the results. Produce tables and figures to visualize the results as appropriate.
