# Parameters

Every parameter has one place:

| What | Where |
|---|---|
| The run: datasets, methods, test sets, evaluation, folders | `nextflow.config`, or a parameter file given with `-params-file` |
| A dataset: source, labels, quality control and preprocessing, inputs of single methods | `datasets/<name>.yml` |
| A method: what it needs from the pipeline | `models/<method>/method.yml` |
| A method's own settings | `models/<method>/src/run.sh`, and for the LLMs `src/openrouter.yml` and `prompt.txt` |
| The mapping of answers to the Cell Ontology | `resources/cell_type_ontology_mappings.csv` |

## Pipeline (`nextflow.config`)

| Parameter | Default | Meaning |
|---|---|---|
| `datasets` | the four datasets of the paper | names of files in `datasets/` |
| `models` | the eight established methods | names of folders in `models/` |
| `subset_sizes` | `[1000, 10000]` | cells per test set; a method can set a maximum (`max_subset_size`) |
| `subset_strategy` | `stratified` | `stratified` keeps the proportions of the cell types, `random` samples cells |
| `min_cells_per_type` | `20` | a stratified test set holds at least this many cells of every type; rarer types are left out |
| `download_from_zenodo` | `true` | fetch the test sets from Zenodo (`datasets/zenodo_manifest.yml`) instead of building them |
| `fetch_context_from_zenodo` | `true` | the same for the clustering samples of scType and SCSA |
| `context_max_size` | `100000` | cells in a clustering sample; the whole dataset if it is smaller |
| `context_min_cells_per_type` | `150` | cells per type that a clustering sample aims for |
| `candidate_types` | `resources/all_cell_types.txt` | cell type names offered to the embedding methods in the unconstrained setting |
| `singler_reference_cap_per_label` | `500` | cells per label kept of a CELLxGENE reference for SingleR; `0` keeps all |
| `ortholog_tables` | mouse to human | one-to-one ortholog tables for methods that read human genes only |
| `mappings` | `resources/cell_type_ontology_mappings.csv` | the mapping table |
| `append_new_answers` | `true` | `main.nf` adds new answers to the mapping table for review (`annotate.nf` never does) |
| `min_scoring_coverage` | `0.9` | share of answers that must be Cell Ontology terms or reviewed for a run to get metrics |
| `use_text_encoder` | `false` | also score the similarity of the label texts (needs torch) |
| `text_encoder_model` | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | the text encoder |
| `cache_dir` | `cache` | downloads, test sets and intermediate files |
| `outdir` | `results` | annotations, scores, tables and plots |
| `predictions_dir` | `<outdir>/predictions` | stored predictions; runs found there are not run again |
| `container_image`, `r_container_image` | see `nextflow.config` | images of the container profiles; the second one for the R methods |

Profiles: `local` (conda), `docker` and `apptainer` (containers, one GPU), `docker_local` (Docker
without GPU), `helix` and `slurm_gpu` (SLURM clusters).

## Dataset (`datasets/<name>.yml`)

Source, one of:

| Key | Meaning |
|---|---|
| `path` | a local h5ad file, relative to the repository |
| `url` | an h5ad file to download (direct links, Google Drive and figshare links) |
| `source: cellxgene_census`, `census_dataset_id`, `census_version` | a dataset of CELLxGENE Census |

Cells and labels:

| Key | Meaning |
|---|---|
| `name` | the dataset's name, the same as the file name |
| `annotation_key` | column of `obs` with the cell type labels |
| `batch_key` | column of `obs` with the batch |
| `apply_ontology_mapping` | `true` if the labels are not Cell Ontology terms and are mapped through the mapping table |

Biological context, for methods that use it: `organism` (default `Homo sapiens`; a mouse dataset
sets `Mus musculus`), `species` (`human` or `mouse`), `tissue`, and for InstructCell
`sequencing_method`, or the per-cell columns `species_key`, `tissue_key`,
`sequencing_method_key`.

Quality control and preprocessing (`preprocessing:`, read by `bin/preprocess.py`; the values in
the files of the paper are the defaults):

| Key | Default | Meaning |
|---|---|---|
| `nmads_main` | `5` | cells further than this many median absolute deviations from the median in counts, genes or the share of the top genes are removed |
| `nmads_mt` | `3` | the same for the mitochondrial share |
| `pct_counts_mt_threshold` | `8.0` | cells with a higher mitochondrial percentage are removed |
| `percent_top` | `[20]` | the number of top genes whose share of the counts is checked |
| `min_cells` | `10` | genes detected in fewer cells are removed |
| `min_genes` | `200` | cells with fewer detected genes are removed |
| `count_layer_key` | `counts` | layer with the raw counts, if they are not in `X` |
| `ensembl_col`, `gene_names_col` | found automatically | columns of `var` with Ensembl ids and gene symbols |
| `species` | `hsapiens` | Ensembl BioMart species, used only to map gene symbols to Ensembl ids |

After quality control, every cell is scaled to 10,000 counts and log-transformed, and the genes
get Ensembl ids. The same steps run in both entry points. Points to know when you build test sets
from your own data:

- Mitochondrial genes are recognized by gene symbols starting with `MT-` in the index of `var`. If
  the index holds Ensembl ids, the mitochondrial criteria have no effect.
- The outlier rule uses log-scaled counts only if `obs` already has `log1p_*` columns.
- Normalization and log transform are skipped if the data look done already: a matrix without
  negative values and with a maximum below 100 counts as log-transformed, which raw counts of very
  shallow data can also be.
- Preprocessed data and test sets are cached under the dataset's name. After changing a value,
  delete the dataset's files in `cache/datasets/preprocessed/` and `cache/datasets/subset/`.

Inputs of single methods: `celltypist_model` (CellTypist), `sctype_tissue` (scType),
`scsa_tissue` (SCSA), and for SingleR either a celldex reference (`singler_reference_source:
celldex`, `singler_reference`, `singler_reference_tier`) or a CELLxGENE dataset
(`singler_reference_source: cellxgene`, `singler_reference_id`, `singler_reference_version`,
`singler_reference_label_key`).

## Method (`models/<method>/method.yml`)

| Key | Meaning |
|---|---|
| `name`, `description` | for people |
| `needs.candidate_types` | the method chooses from a list of cell type names; its path is in `CANDIDATE_TYPES_FILE` |
| `needs.clustering_context` | the method clusters a larger sample of the dataset; its path is in `CONTEXT_H5AD` |
| `needs.label_constraint` | the method keeps only labels that count for the expected cell types; the table is in `LABEL_CONSTRAINT` |
| `needs.singler_reference` | a reference from CELLxGENE, for datasets that name one; its path is in `SINGLER_REF_H5AD` |
| `needs.human_genes` | the method reads human genes only; mouse data are given as human orthologs |
| `needs.api_key` | the method calls the OpenRouter API; the key is in `API_KEY` |
| `container: r` | the method runs in the R image under the container profiles |
| `max_subset_size` | the largest test set the method runs on |
| `served_models` | for a wrapper of several models: the file whose `models` list names them, one prediction file per model |

The method's own settings (model checkpoints, batch sizes, random seeds) are arguments in its
`src/run.sh` or defaults in its `src/predict.py`.

## LLMs (`models/openrouter_LLM_*`)

Both settings share `src/` (`openrouter_LLM_unblind/src` links to `openrouter_LLM_blind/src`).
`src/openrouter.yml` lists the served models and the request parameters (seed, number of genes per
cell, parallel requests, provider routing, generation and reasoning settings). Each setting has its
own prompt template in `prompt.txt`. In the template, `{cell_sentence}` is the list of genes
ordered by expression, `{num_genes}` the value of `top_k_genes`, and `{unique_cell_types}` the list
of expected cell types, which only the constrained prompt uses. Besides the predictions, the
wrapper writes for every request the provider, the tokens and the cost (`usage_<model>.csv`).

## Evaluation

Metrics and the coverage threshold are set in `nextflow.config` (see above). The mapping table is
described in the [README](../README.md#mapping-answers-to-the-cell-ontology). The Cell Ontology
release is fixed in `bin/util/cell_ontology_similarity.py` (`CL_RELEASE`).
