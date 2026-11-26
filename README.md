# Cell type annotation benchmark

A [Nextflow](https://www.nextflow.io/) pipeline that annotates single-cell RNA-seq data with
general-purpose large language models (LLMs), single-cell language models (scLMs) and established
annotation methods, maps every answer to the [Cell Ontology](https://obofoundry.org/ontology/cl.html)
and scores it against the labels of the dataset. It also holds the data and code behind the figures
and tables of our paper (see [paper/README.md](paper/README.md)).

## What the pipeline does

1. **Test sets.** For every dataset it draws test sets of a given size (1,000 and 10,000 cells in
   the paper), keeping the proportions of the cell types and at least 20 cells per type. The test
   sets of the paper are on [Zenodo](https://doi.org/10.5281/zenodo.21443586) and are fetched from
   there by default. They can also be built from the original data: download, quality control,
   normalization, subsampling.
2. **Annotation.** Every selected method annotates every cell of a test set. A method runs without
   the list of the dataset's cell types (folder name ending in `_blind`) or with it (`_unblind`).
3. **Mapping.** Every answer is matched to a Cell Ontology term, directly or through a curated table
   of answers that are not ontology terms (`resources/cell_type_ontology_mappings.csv`).
4. **Evaluation.** The answers are scored against the dataset's labels: accuracy, balanced
   accuracy, F1, Lin and Rada similarity on the ontology, an agreement score that gives half a point
   for a related coarser or finer cell type, and optionally a text similarity of the labels.

There are two entry points:

| Entry point | Runs | Ends with |
|---|---|---|
| `main.nf` | steps 1 to 4 on test sets | annotations, scores, combined tables and plots |
| `annotate.nf` | steps 1 to 3 on all cells of a dataset, without subsampling | annotations mapped to the Cell Ontology |

## Requirements

- Nextflow 25.04 or later (tested with 25.04.6) and Java 17 or later.
- Conda or Mamba (profile `local`), or Docker or Apptainer (profiles `docker`, `apptainer`).
  The pipeline builds one environment per step from the files in `envs/`.
- For the scLMs: a CUDA GPU.
- For the LLMs: an [OpenRouter](https://openrouter.ai/) API key. Annotating the four 1,000-cell
  test sets of the paper with all eleven LLMs in both settings cost about 1,200 USD in May 2026.

## Quick start

The toy dataset in `tests/data/` holds 40 cells of four cell types. This run builds a test set
from it, annotates it with the random baseline and scores the result:

```bash
nextflow run main.nf -profile local -params-file tests/toy.params.yml
```

The results are in `results_toy/` (see [Outputs](#outputs)). The first run builds the conda
environments, which takes a while.

## Annotating a dataset

`annotate.nf` annotates every cell of the selected datasets after quality control and
preprocessing (see [docs/parameters.md](docs/parameters.md#dataset-datasetsnameyml)) and maps the
answers to the Cell Ontology, without scoring:

```yaml
# annotate.yml
datasets: [toy]
models: [celltypist_blind, openrouter_LLM_percluster_blind]
```

```bash
nextflow run annotate.nf -profile local -params-file annotate.yml
```

With `annotation_subset_size` it annotates a subsample of that many cells instead. Methods with a
maximum number of cells (`max_subset_size` in their `method.yml`), such as the LLMs that send one
request per cell, are left out when all cells are annotated; the per-cluster LLMs send one
request per cluster instead. The input still needs a column of labels
(`annotation_key`); for unlabeled data a column with one placeholder value serves, with methods
that run without the list of expected cell types.

## Running the benchmark

Choose datasets, methods and test set sizes in a parameter file:

```yaml
# my_run.yml
datasets: [HIHA, human_pancreas]
models: [celltypist_blind, celltypist_unblind, scsa_blind, scsa_unblind]
subset_sizes: [1000]
```

```bash
nextflow run main.nf -profile local -params-file my_run.yml
```

Single values can also be given on the command line (`--subset_strategy random`), lists only in
a parameter file. All parameters and their defaults are listed in
[docs/parameters.md](docs/parameters.md). The methods are the folders in `models/`, the datasets
the files in `datasets/`. Without a parameter file the pipeline runs the eight established
methods on the four datasets of the paper.

**LLMs.** The OpenRouter key is read from a Nextflow secret, so it is never written to a log or a
task script:

```bash
nextflow secrets set OPENROUTER_API_KEY 'sk-or-v1-...'
```

Put a space before the command if your shell keeps commands that start with one out of its history
(`setopt HIST_IGNORE_SPACE` in zsh). Only the LLM wrappers need the key. Which LLMs are queried and
how is set in `models/openrouter_LLM_blind/src/openrouter.yml`, the prompt of each setting in
`models/openrouter_LLM_blind/prompt.txt` and `models/openrouter_LLM_unblind/prompt.txt`.

**Stored predictions.** The predictions of every run are kept in `<outdir>/predictions/` (or in
`--predictions_dir`). A run whose predictions are there is not run again; only mapping and scoring
are repeated. Delete the files of a run to run it again.

**Apple Silicon.** The R methods (SingleR, scType) have no conda packages for `osx-arm64`. Start
them with `CONDA_SUBDIR=osx-64 CONDA_ALWAYS_COPY=true nextflow run ...`, which uses the Intel
packages through Rosetta 2, or use a container profile.

## Mapping answers to the Cell Ontology

An answer that names a Cell Ontology term (by name or id, ignoring case) needs no mapping. Every
other answer is looked up in `resources/cell_type_ontology_mappings.csv`, a semicolon-separated
table with one row per answer:

| Column | Meaning |
|---|---|
| `reviewed` | `True` once a person has checked the row |
| `origin` | the run that first produced the answer |
| `predicted_label` | the answer, in lower case |
| `chosen_match` | the Cell Ontology term the answer stands for (`cell` for answers that name no cell type) |
| `best_match` | the automatic suggestion |

`main.nf` adds answers that are not in the table yet as unreviewed rows with a suggestion and says
so at the end of the run. Unreviewed answers count as wrong, and a run in which fewer than 90% of the
answers are ontology terms or reviewed gets no metrics. After reviewing the new rows (set
`chosen_match`, then `reviewed` to `True`), run the pipeline again with `-resume`; mapping and
scoring are repeated, the methods are not. `annotate.nf` never changes the table: answers it does
not know are marked `unmapped` in its output.

## Outputs

| Path in `outdir` | Content |
|---|---|
| `predictions/<method>/<dataset>/<size>_<strategy>/` | the answers of each run, as the method wrote them; for the LLMs also the provider, tokens and cost of every request (`usage_<model>.csv`) |
| `annotations/<dataset>/<method>/<size>_<strategy>/` (or `all/`) | per cell the answer, its Cell Ontology term and id, and how it was mapped; with a summary in JSON |
| `clusters/<dataset>/` | for methods that annotate clusters: the cluster of every cell, the marker genes of every cluster and the settings |
| `evaluation/<dataset>/` | per run the metrics (JSON), the scores per true cell type and per answer pair, and a confusion matrix |
| `benchmark_metrics.csv` | one row per dataset, model and test set size |
| `per_class_scores.csv` | recall and ontology scores per true cell type and run |
| `confusion_pairs.csv` | number of cells per pair of true and predicted cell type and run |
| `plots/` | heatmaps of the models by dataset and rankings across datasets, per metric and test set size |

Downloads, test sets and other intermediate files are kept in `cache/`.

## Methods

| Family | Methods (folders in `models/`) |
|---|---|
| General-purpose LLMs, via OpenRouter | `openrouter_LLM_blind`, `openrouter_LLM_unblind`: Claude Opus 4.7, Claude Sonnet 4.6, DeepSeek V3.2, Gemini 3.1 Pro, GPT-5.5, GPT-OSS 20B and 120B, Kimi K2.6, Llama 3.1 70B, Llama 3.3 70B, Llama 4 Maverick |
| The same LLMs, per cluster | `openrouter_LLM_percluster_blind`, `openrouter_LLM_percluster_unblind` |
| Single-cell language models | `c2s_scale_2B_*`, `c2s_scale_27B_*`, `cellwhisperer_*`, `instructcell_*`, `langcell_*`, `mmcontext_*` |
| Established methods | `celltypist_*`, `scsa_*`, `sctype_*`, `singler_*` |
| Chance reference | `random_stratified` |

The LLMs per cell get the 200 most expressed genes of each cell. Per cluster, the cells are first
clustered (Leiden), and each cluster's 25 top marker genes go into one request, whose answer all
cells of the cluster get. This is the per-cluster analysis of the paper, and far cheaper for large
datasets. The clustering settings are parameters (see [docs/parameters.md](docs/parameters.md)).

The established methods answer from a fixed vocabulary. In their constrained setting
(`*_unblind`) they keep only the labels that lie at or below one of the expected cell types in the
Cell Ontology. Each method folder holds a `method.yml` with its name and the inputs it needs.

## Datasets

| Name | Tissue | Organism | Source |
|---|---|---|---|
| `HFRA` | fetal retina | human | CELLxGENE Census |
| `HIHA` | peripheral blood | human | CELLxGENE Census |
| `Zeisel` | nervous system | mouse | authors' loom file, converted to h5ad |
| `human_pancreas` | pancreas | human | figshare (Luecken et al. 2022) |
| `toy` | peripheral blood, 40 cells | human | a sample of `HIHA`, for testing |

## Adding datasets and methods

- [docs/adding_datasets.md](docs/adding_datasets.md) shows how to add a dataset, with the toy
  dataset as the example.
- [docs/adding_methods.md](docs/adding_methods.md) shows how to add a method, with the random
  baseline as the example.

## Repository layout

```
main.nf, annotate.nf   entry points
nextflow.config        parameters and profiles
workflows/             the annotation and evaluation workflows
modules/               the processes and helper functions
bin/                   the scripts the processes call
datasets/              one configuration per dataset, and the Zenodo manifest
models/                one folder per method
envs/                  conda environments and container recipes
resources/             the mapping table and Ensembl gene tables
tests/                 unit tests and the toy dataset
docs/                  parameters and how-to guides
paper/                 data, results and code of the paper
```

## Tests

The unit tests run with pytest or directly, in the evaluation environment (`envs/evaluation.yml`):

```bash
for t in tests/test_*.py paper/tests/test_*.py; do python "$t"; done
```

## License

The code is released under the MIT license (see `LICENSE`). Third-party material keeps its own
license: the ScType marker database (`models/sctype_blind/src/ScTypeDB_full.xlsx`, GPL-3.0), the
Cell Ontology (CC BY 4.0), the Ensembl gene tables in `resources/ensembl/` and the toy dataset
(CC BY 4.0, see `tests/data/README.md`). The methods fetch their own code and weights under their
own licenses.

## Citation

Please cite the paper when you use this pipeline (reference to follow).
