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
| `subset_strategy` | `'stratified'` preserves per-type proportions; `'random'` is uniform across cells |
| `min_cells_per_type` | Minimum cells per cell type in subsets (default: 20). Types below this threshold are dropped in stratified mode (ignored in random mode) |
| `apply_ontology_mapping` | Fallback for datasets without a per-dataset `apply_ontology_mapping` flag in their YAML |
| `download_from_zenodo` | If `true`, skip `download`→`preprocess`→`subset` and fetch preprocessed+subsetted files from the Zenodo record listed in `datasets/zenodo_manifest.yml`. All requested `(dataset, subset_size, subset_strategy)` combinations must be in the manifest or the pipeline aborts. |

There are profiles `local` for running locally, `helix` for the helix cluster, and `imbi_gpu` for the IMBI GPU cluster.

#### Pipeline processes

The pipeline runs the following processes in order:

1. **download** - Fetches raw data from URL/Google Drive/CellxGene Census. Caches the full file under `cache/`.
2. **preprocess** - QC filtering, gene/cell filtering, normalization, log transform, Ensembl ID mapping, and (optionally) cell-type ontology mapping. Ontology mapping is controlled per-dataset via `apply_ontology_mapping` in the dataset YAML (falls back to the global `params.apply_ontology_mapping`). QC plots saved to `cache/plots/`.
3. **subset** - Subsampling to one of the configured sizes. Strategy is either `stratified` (preserves per-type proportions, drops types below `min_cells_per_type`) or `random` (uniform; preserves natural class imbalance). One subset per unique `(size, strategy)` pair. QC outputs (CSV + plots) saved to `cache/subset_qc/<dataset>/sub<size>_<strategy>/`.
4. **prepare_model** - Optional model setup (git cloning, etc.) via `prepare.sh`.
5. **run_model** - Runs model predictions. Results cached under `cache/models/predictions/<model>/<dataset>/<size>_<strategy>/`.
6. **evaluate_annotation** - Computes metrics (accuracy, F1, ontology similarity) and confusion matrices.
7. **combine_results** - Aggregates results into CSV and visualization plots.

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

After upgrading to the `subset_strategy` / merged-preprocess pipeline, the following cached artifacts are incompatible with the new layout and can be deleted to reclaim disk:

```bash
# Old pre-merge ontology step (now folded into preprocess):
rm -rf cache/datasets/mapped/

# Old untagged subset files (new files carry a `_stratified` / `_random` suffix before `.h5ad`;
# only remove files that end with `_sub<digits>.h5ad` and nothing else):
find cache/datasets/subset -maxdepth 1 -type f -regex '.*_sub[0-9]+\.h5ad' -delete

# Old untagged prediction caches (new caches carry a strategy suffix in the path).
# Inspect cache/models/predictions/ for `<size>/` directories without a `_<strategy>` suffix and remove them.
```

The pipeline itself does not remove these automatically.

#### Fetching preprocessed data from Zenodo

To skip the full preprocessing and subsetting chain, set `download_from_zenodo=true`
(via `nextflow.config` or `--download_from_zenodo true`). The pipeline will then
fetch files directly from the public Zenodo record pinned in
`datasets/zenodo_manifest.yml` and place them under `cache/datasets/subset/`.

Currently mirrored on Zenodo (16 files): datasets `HFRA`, `HIHA`, `human_pancreas`,
`Zeisel`; subset sizes `1000` and `10000`; strategies `stratified` and `random`.
Requested combinations outside this set cause the pipeline to abort upfront with
a list of the missing files. MD5 checksums are verified after each download.

Zenodo files are ontology-mapped (produced with `apply_ontology_mapping=true`);
the value of `params.apply_ontology_mapping` is ignored for the fetch step but
still forwarded to the evaluation step.

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
