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
| `annotation_subset_size` | `null` | `annotate.nf` only: cells to annotate per dataset; `null` annotates all cells |
| `fetch_context_from_zenodo` | `true` | the same for the clustering samples of scType and SCSA |
| `context_max_size` | `100000` | cells in a clustering sample; the whole dataset if it is smaller |
| `context_min_cells_per_type` | `150` | cells per type that a clustering sample aims for |
| `candidate_types` | `resources/all_cell_types.txt` | cell type names offered to the embedding methods in the unconstrained setting |
| `cluster_resolution` | `1.0` | Leiden resolution of the clustering step; higher values give more clusters |
| `cluster_n_pcs`, `cluster_n_neighbors` | `50`, `15` | principal components and nearest neighbours of the clustering |
| `cluster_n_top_genes` | `2000` | highly variable genes for the PCA; `0` uses all genes |
| `cluster_scale` | `true` | scale the genes before the PCA |
| `cluster_use_batch_key` | `true` | select the highly variable genes within each batch (`batch_key` of the dataset) |
| `cluster_integration` | `none` | `harmony` corrects the principal components for the batches (`batch_key` of the dataset) with Harmony; implies `cluster_use_batch_key` |
| `cluster_marker_genes` | `25` | marker genes per cluster, ranked with `cluster_rank_method` (`wilcoxon`) |
| `cluster_seed` | `0` | random seed of PCA, neighbours and Leiden |
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
| `path` | a local h5ad file, as an absolute path or relative to the repository |
| `url` | an h5ad file to download (direct links, Google Drive and figshare links) |
| `source: cellxgene_census`, `census_dataset_id`, `census_version` | a dataset of CELLxGENE Census |

Cells and labels:

| Key | Meaning |
|---|---|
| `name` | the dataset's name, the same as the file name |
| `annotation_key` | column of `obs` with the cell type labels (default `cell_type`). The benchmark needs it. In annotation mode it is optional, and data without the column give every cell the placeholder label `unlabeled` |
| `batch_key` | column of `obs` with the batch, for data from several batches; optional |
| `cluster_key` | column of `obs` with clusters of your own, which the per-cluster methods then use instead of the clustering step |
| `expected_cell_types` | `annotate.nf` only: a list of cell type names that the constrained methods take as the expected cell types instead of those in the labels. Cell Ontology names let the established methods relate their labels to the list. Without a list and without `annotation_key`, the constrained methods are left out for the dataset. |
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

Inputs of single methods: `celltypist_model` (CellTypist, default `Immune_All_High.pkl`),
`sctype_tissue` (scType), `scsa_tissue` (SCSA), and for SingleR either a celldex reference
(`singler_reference_source: celldex`, `singler_reference`, `singler_reference_tier`) or a
CELLxGENE dataset (`singler_reference_source: cellxgene`, `singler_reference_id`,
`singler_reference_version`, `singler_reference_label_key`).

## Method (`models/<method>/method.yml`)

| Key | Meaning |
|---|---|
| `name`, `description` | for people |
| `needs.candidate_types` | the method chooses from a list of cell type names; its path is in `CANDIDATE_TYPES_FILE` |
| `needs.clustering_context` | the method clusters a larger sample of the dataset; its path is in `CONTEXT_H5AD` |
| `needs.clusters` | the method annotates clusters, from the clustering step or the dataset's `cluster_key`; the cluster of every cell is in `CLUSTERS_CSV`, the marker genes of every cluster in `MARKERS_CSV` |
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

Both settings share `src/` (`openrouter_LLM_constrained/src` links to
`openrouter_LLM_unconstrained/src`). `src/openrouter.yml` lists the served models and the request
parameters (seed, number of genes per cell, parallel requests, provider routing, generation and
reasoning settings). A cell sentence holds the `top_k_genes` most highly expressed genes of a
cell, taken from all genes or, with `gene_pool: hvg`, from the `n_hvgs` most variable genes of the
test set. These are selected with scanpy (flavor `seurat`), within each batch if the dataset has a
`batch_key`. `exclude_rp_mt_hb: true` leaves out ribosomal, mitochondrial and hemoglobin genes
first. As in the clustering step, these are the genes whose symbols start with RPS, RPL, MT- or
HB, which also catches a few other genes such as HBEGF. Each setting has its own prompt template
in `prompt.txt`. In the template,
`{cell_sentence}` is the list of genes ordered by expression, `{num_genes}` the value of
`top_k_genes`, and `{unique_cell_types}` the list of expected cell types, which only the
constrained prompt uses. Besides the predictions, the wrapper writes for every request the
provider, the tokens and the cost (`usage_<model>.csv`).

The per-cluster methods (`openrouter_LLM_percluster_*`) use the same `src/` and served models
with their own `prompt.txt`: `{marker_genes}` are the cluster's marker genes, `{num_genes}`
their number, and `{unique_cell_types}` the list of expected cell types, written as in the
per-cell prompt and drawn in a new order for each cluster. Their usage file has one row per
cluster.

## Clustering step

`bin/cluster_cells.py` clusters with Leiden on the PCA of the log-normalized expression and ranks
the marker genes of every cluster, without ribosomal, mitochondrial and hemoglobin genes. A drawn
test set is clustered within the clustering sample that contains it, all cells of a dataset as
they are. The defaults follow the usual scanpy workflow: the PCA uses the 2,000 most variable
genes, selected within each batch, and scaled. The per-cluster analysis of the paper used all genes
without scaling; `cluster_n_top_genes: 0`, `cluster_scale: false` and `cluster_use_batch_key:
false` reproduce its clusters exactly. On the clustering samples of the paper, 100,000 cells each,
the step needed 10 to 13 GB of memory.

Datasets that combine technologies or donors can form clusters along these.
`cluster_integration: harmony` corrects the principal components for the dataset's `batch_key`
with Harmony before the neighbour graph is built (it needs at least 45 cells). On the 999 cells of
the pancreas test set, which come from seven technologies, Harmony raised the agreement of the
clusters with the cell types (adjusted Rand index) from 0.38 to 0.62 with the defaults and from
0.32 to 0.58 with the paper's settings, and lowered their agreement with the technology from
about 0.25 to below 0.04.

A dataset with `cluster_key` in its yml brings its own clusters, for example from an earlier
analysis. The cells keep these clusters, the clustering parameters are not used, and only the
marker genes are ranked.

## Evaluation

Metrics and the coverage threshold are set in `nextflow.config` (see above). The mapping table is
described in the [README](../README.md#mapping-answers-to-the-cell-ontology). The Cell Ontology
release is fixed in `bin/util/cell_ontology_similarity.py` (`CL_RELEASE`).
