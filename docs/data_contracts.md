# Data Contracts

This repository stores benchmark scripts rather than a single packaged API. The contracts below describe the common file formats expected by the evaluation scripts and utility wrappers.

## H5 Matrix Files

Most Python evaluation scripts expect an HDF5 file with a dataset named `data`.

Expected shape:

- Rows are cells or spots.
- Columns are features, latent dimensions, or neighbors, depending on the task.
- Some legacy scripts transpose matrices when rows < columns.

Example paths:

- `data/classification/demo_data/data1.h5`
- `data/dr&bc/embedding/embedding.h5`
- `data/imputation/real_data.h5`

## Label CSV Files

Classification scripts expect a column named `x`.

Embedding and clustering metric scripts are more permissive in practice, but the recommended format is:

```csv
x
B cell
T cell
NK cell
```

For multi-batch runs, pass one label CSV per batch in the same order as the corresponding embedding files or graph rows.

## Clustering H5 Files

scIB-style metrics expect clustering files with this dataset:

```text
/obs/cluster_leiden
```

Values are stored as byte strings in the existing examples and are decoded by the metric scripts.

Example paths:

- `data/clustering/embedding/sinfonia_clustering.h5`
- `data/clustering/embedding/sinfonia_clustering_batch.h5`

## Graph Files

Graph-based scIB-style metrics expect two H5 files:

- `knn_indices.h5`: nearest-neighbor indices under `data`
- `knn_dists.h5`: nearest-neighbor distances under `data`

Example paths:

- `data/dr&bc/graph/knn_indices.h5`
- `data/dr&bc/graph/knn_dists.h5`

## Imputation Files

Imputation metrics compare a real matrix and imputed matrix. Both should use the same feature order and cell order.

Example paths:

- `data/imputation/real_data.h5`
- `data/imputation/imputed_data.h5`
- `data/imputation/cty.csv`

## Output Conventions

Recommended output locations:

- `results/scib_metrics/<run_name>/metric.csv`
- `results/classification/<run_name>/predict.csv`
- `results/classification/<run_name>/query.csv`
- `results/imputation/<run_name>/`
- `results/spatial_registration/<run_name>/`

The `results/` directory is ignored by Git.

## Validation

Use `scripts/validate_inputs.py` before running a heavier workflow:

```bash
python scripts/validate_inputs.py classification \
  --reference data/classification/demo_data/data1.h5 \
  --query data/classification/demo_data/data2.h5 \
  --reference-labels data/classification/demo_data/cty1.csv \
  --query-labels data/classification/demo_data/cty2.csv
```
