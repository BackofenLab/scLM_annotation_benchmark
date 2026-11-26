# Adding a method

A method is a folder in `models/` with a script that reads a test set and writes one answer per
cell, plus a conda environment in `envs/models/`. The random baseline is the smallest complete
example:

```
models/random_stratified/
    method.yml          name and what the method needs from the pipeline
    src/run.sh          called by the pipeline
    src/predict.py      the method itself
envs/models/random_stratified.yml
```

## 1. The script

The pipeline calls

```bash
bash models/<method>/src/run.sh <test_set.h5ad> <out_dir>/predictions.csv <datasets/NAME.yml> <cache_dir>/models
```

- `test_set.h5ad` holds the test set: log-normalized expression in `X`, raw counts in
  `layers["counts"]`, Ensembl ids as `var` index, gene symbols in `var["gene_names"]` and the true
  labels in `obs[annotation_key]`.
- `predictions.csv` must hold one row per cell with the columns `cell_id`, `true_labels` and
  `predicted_labels`. The true labels are passed through for scoring; the method must not use them
  to answer.
- The dataset configuration gives `annotation_key` and anything else the method needs from the
  dataset, for example a tissue name.
- `<cache_dir>/models` is a folder for downloaded weights that is kept between runs.

The random baseline reads the label column from the dataset configuration and starts its script:

```bash
ANNOTATION_KEY=$(python - <<'PY' "$CFG_YAML"
import yaml, sys
config = yaml.safe_load(open(sys.argv[1]))
print(config.get('annotation_key', 'cell_type'))
PY
)

python "$(dirname "$0")/predict.py" \
  --input "$INPUT_H5AD" \
  --output "$OUT_CSV" \
  --annotation_key "$ANNOTATION_KEY" \
  --strategy stratified \
  --seed 42
```

`predict.py` draws a label for every cell from the label frequencies of the test set and writes
the three columns. It is a chance reference, which is why it may look at the true labels.

Settings of the method itself (a checkpoint, a batch size, a seed) belong in `run.sh` or in the
defaults of the method's script.

## 2. The environment

`envs/models/<method>.yml` is a conda environment with everything the scripts import. Pin the
versions. The random baseline needs Python, NumPy, pandas, anndata and PyYAML. Under the container
profiles the methods run in the image named in `container_image` instead.

## 3. The description

`models/<method>/method.yml`:

```yaml
name: Random labels
description: "Chance reference: draws a label for every cell from the label frequencies of the test set."
```

Add under `needs:` what the method gets from the pipeline (each is explained in
[parameters.md](parameters.md#method-modelsmethodmethodyml)):

```yaml
needs:
  candidate_types: true      # CANDIDATE_TYPES_FILE: cell type names to choose from
  clustering_context: true   # CONTEXT_H5AD: a larger sample of the dataset to cluster
  clusters: true             # CLUSTERS_CSV, MARKERS_CSV: clusters and their marker genes
  label_constraint: true     # LABEL_CONSTRAINT: which labels count for the expected cell types
  singler_reference: true    # SINGLER_REF_H5AD: a reference from CELLxGENE
  human_genes: true          # mouse data are given as human orthologs
  api_key: true              # API_KEY: the OpenRouter key
max_subset_size: 1000        # skip larger test sets
```

A method that needs setup once (cloning a repository, downloading weights) puts it in
`src/prepare.sh`, which the pipeline runs in `src/` before the first run.

## 4. The two settings

The constrained setting is a second folder whose name ends in `_unblind`. If the code is the same,
link its `src` to the first folder (`ln -s ../<method>_blind/src models/<method>_unblind/src`) and
give it its own `method.yml`. A method that answers from a fixed vocabulary can declare
`needs.label_constraint` and keep only the labels in the table at `LABEL_CONSTRAINT`
(`bin/util/label_constraint.py`, `bin/util/label_constraint.R`). A generative method gets the list
of expected cell types in its prompt instead; the true labels of the test set are that list. In
annotation mode, a dataset can give its own list (`expected_cell_types`). The pipeline then
writes it to the file in `EXPECTED_TYPES_FILE`, one name per line, which a constrained method
reads in place of the labels; the label constraint is built from the same list.

## 5. Test it on the toy dataset

```yaml
# my_method.yml
datasets: [toy]
models: [random_stratified]   # your method
subset_sizes: [40]
min_cells_per_type: 5
download_from_zenodo: false
fetch_context_from_zenodo: false
context_max_size: 40
context_min_cells_per_type: 5
```

```bash
nextflow run main.nf -profile local -params-file my_method.yml
```

Then look at `results/annotations/toy/<method>/sub40/predictions_annotations.csv`. Answers marked
`unreviewed mapping` or `unmapped` are not Cell Ontology terms. `main.nf` has added them to
`resources/cell_type_ontology_mappings.csv`; review them as described in the
[README](../README.md#mapping-answers-to-the-cell-ontology) and run again with `-resume`.

A method whose labels are all coarser than the toy dataset's types (CellTypist with
`Immune_All_High.pkl`, for example) gets no allowed label in its constrained setting and stops with
a message. That is the expected behaviour of the constraint.
