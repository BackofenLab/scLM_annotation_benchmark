# Adding a dataset

A dataset is an h5ad file with raw counts and a column of cell type labels, plus a configuration
file in `datasets/`. The toy dataset is a complete example. Annotation mode also takes data
without labels (see the [README](../README.md#annotation-mode)).

## 1. The data

The file needs

- raw counts, in `X` or in a layer (name it with `preprocessing.count_layer_key`),
- genes with Ensembl ids or gene symbols in `var` (symbols are mapped to Ensembl ids of the
  dataset's `species`, which needs the `pybiomart` package and an internet connection; offline,
  a table of human genes in `resources/ensembl/` serves human data only),
- a column in `obs` with the cell type of every cell, and one with the batch if the cells come
  from several batches.

The toy dataset `tests/data/toy_blood.h5ad` has raw counts in `X`, Ensembl ids as `var` index,
symbols in `var["feature_name"]`, and `cell_type` and `batch_id` in `obs`.

## 2. The configuration

Create `datasets/<name>.yml`. The file name is the dataset's name. The toy dataset's
`datasets/toy.yml` in short:

```yaml
name: toy
path: tests/data/toy_blood.h5ad   # or url: ..., or source: cellxgene_census with census_dataset_id

annotation_key: "cell_type"
batch_key: "batch_id"
apply_ontology_mapping: false     # the labels are Cell Ontology terms already

species: "human"
tissue: "blood (peripheral blood immune cells)"

preprocessing:
  min_cells: 3                    # the defaults suit large datasets; 40 cells need a lower value
```

Every key is explained in [parameters.md](parameters.md#dataset-datasetsnameyml). Keys that only
one method reads (`celltypist_model`, `sctype_tissue`, `scsa_tissue`, `singler_reference*`) are
needed only when that method runs on the dataset.

## 3. Labels that are not Cell Ontology terms

The scores compare Cell Ontology terms, so every label must resolve to one. Labels from CELLxGENE
are terms already. For other labels set `apply_ontology_mapping: true`. On the first run, labels
that are not in the mapping table are added to `resources/cell_type_ontology_mappings.csv` with a
suggested term and `reviewed` set to `False`. Check the suggestions, correct `chosen_match` where
needed and set `reviewed` to `True`. The first run has already written the suggested terms
into the preprocessed data, so delete the dataset's files in `cache/datasets/preprocessed/` and
`cache/datasets/subset/` before you run again.

## 4. Run it

Write a parameter file, for example `my_dataset.yml`:

```yaml
datasets: [toy]
models: [random_stratified]
subset_sizes: [40]          # cells per test set
min_cells_per_type: 5       # cell types with fewer cells are left out of the test set
download_from_zenodo: false # build the test set from the data
fetch_context_from_zenodo: false
```

```bash
nextflow run main.nf -profile local -params-file my_dataset.yml
```

This is `tests/toy.params.yml`. The run preprocesses the data (QC plots in `cache/plots/`), draws
the test set (cell type counts in `cache/subset_qc/`), runs the methods and scores them.

## 5. Methods with extra inputs

- scType and SCSA annotate clusters. They get a clustering sample of the full dataset that contains
  the test set; with `fetch_context_from_zenodo: false` the pipeline builds it.
- The scLMs that read human genes only get mouse data as one-to-one human orthologs. Other
  organisms need an ortholog table in `ortholog_tables`.
- SingleR needs a reference: name a celldex reference or a CELLxGENE dataset in the configuration.
- The constrained settings (`*_constrained`) use the dataset's cell types as the list of expected
  types.
