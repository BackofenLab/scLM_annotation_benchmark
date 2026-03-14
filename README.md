## Single-cell and Language foundation models

### Workflow description

#### nextflow.config

Config file with the following parameters:

| Parameter | Description |
|---|---|
| `datasets` | List of datasets to use. Each needs a YAML config in `datasets/<name>.yml` |
| `models` | List of models to use. Each needs `models/<name>/src/run.sh` |
| `cache_dir` | Directory to cache downloaded data, preprocessed files, subsets, and model weights |
| `subset_categories` | Map of model name patterns to subset sizes (e.g., `['openrouter': 1000, 'default': 10000]`). Models whose name contains the key get that subset size; `'default'` is the fallback |
| `min_cells_per_type` | Minimum cells per cell type in subsets (default: 20). Types below this threshold are dropped |
| `apply_ontology_mapping` | Fallback for datasets without a per-dataset `apply_ontology_mapping` flag in their YAML |

There are profiles `local` for running locally, `helix` for the helix cluster, and `imbi_gpu` for the IMBI GPU cluster.

#### Pipeline processes

The pipeline runs the following processes in order:

1. **download** - Fetches raw data from URL/Google Drive/CellxGene Census. Caches the full file under `cache/`.
2. **preprocess** - QC filtering, gene/cell filtering, normalization, log transform, Ensembl ID mapping. QC plots saved to `cache/plots/`.
3. **ontology_mapping** - Maps cell type labels to Cell Ontology terms (optional, controlled per-dataset via `apply_ontology_mapping` in the dataset YAML).
4. **subset** - Stratified sampling preserving cell type proportions. One subset per unique size across model categories. QC outputs (CSV + plots) saved to `cache/subset_qc/<dataset>/sub<size>/`.
5. **prepare_model** - Optional model setup (git cloning, etc.) via `prepare.sh`.
6. **run_model** - Runs model predictions. Results cached under `cache/models/predictions/`.
7. **evaluate_annotation** - Computes metrics (accuracy, F1, ontology similarity) and confusion matrices.
8. **combine_results** - Aggregates results into CSV and visualization plots.

#### Running the pipeline

```bash
# Basic run
nextflow run main.nf -profile local

# With specific datasets and models
nextflow run main.nf -profile local \
  --datasets '["HFRA", "Zeisel"]' \
  --models '["celltypist_blind", "openrouter_LLM_blind"]'

# Override subset sizes
nextflow run main.nf -profile local \
  --subset_categories '["openrouter": 500, "default": 5000]'
```

#### Cache invalidation

Older runs may have cached random subset files at `cache/*_sub*.h5ad`. Those files are incompatible with the new stratified subsetting pipeline and should be removed before rerunning:

```bash
rm -f cache/*_sub*.h5ad
```

#### Dataset configuration

Each dataset has a YAML config in `datasets/` with fields:

- `name` - Dataset identifier
- `source` - Download source (`cellxgene_census` or omit for URL)
- `url` or `census_dataset_id` - Data location
- `batch_key` - Batch column in `adata.obs`
- `annotation_key` - Cell type annotation column
- `apply_ontology_mapping` - Whether to map labels to Cell Ontology terms
- `celltypist_model` - CellTypist model file (for celltypist models)

#### Adding a new model

1. Create `models/<model_name>/src/run.sh` taking inputs: (1) h5ad file, (2) output predictions.csv path, (3) dataset config YAML, (4) cache_dir (optional)
2. Add a Conda environment file `envs/models/<model_name>.yml`
3. If setup is needed (git clone, etc.), add `models/<model_name>/src/prepare.sh`
4. Output: `predictions.csv` with columns `true_labels`, `predicted_labels`

#### LLM models (OpenRouter)

You must provide your own [OpenRouter](https://openrouter.ai/) API key:

```bash
# Recommended: environment variable
export OPENROUTER_API_KEY=sk-or-v1-...
nextflow run main.nf -profile local

# Alternative: command-line flag
nextflow run main.nf --openrouter_api_key sk-or-v1-... -profile local
```

**Never commit or push your API key to the repository.**

### Planning excel sheet

<[internal link removed]>

### Meeting minutes

<[internal link removed]>
