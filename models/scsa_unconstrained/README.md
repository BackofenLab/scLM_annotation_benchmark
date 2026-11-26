# SCSA

Wrapper for SCSA (Cao, Wang and Peng 2020, Front. Genet. 11:490,
https://doi.org/10.3389/fgene.2020.00490), which annotates cell clusters by their marker genes
with the CellMarker database. `src/prepare.sh` clones https://github.com/bioinfo-ibms-pumc/SCSA
into `src/SCSA` (no commit is pinned). `src/run.sh` reads `annotation_key`, `species` (or
`organism`) and `scsa_tissue` from the dataset YAML and starts `src/predict.py` in that checkout.

## What the wrapper does

1. Clusters the cells with scanpy: 2,000 highly variable genes (flavor `seurat`; `seurat_v3`
   would need raw counts, which the clustering contexts lack), scaling, 50 principal components,
   a neighbour graph with k = 15 and Louvain at resolution 1.0. With `CONTEXT_H5AD`, the
   clustering context (`bin/build_context.py`) is clustered instead of the test set, so that
   rare cell types can form their own cluster.
2. Ranks the marker genes of every cluster with scanpy's t-test on the unscaled log-normalized
   data, since SCSA filters them by log fold change, and writes them in SCSA's scanpy format.
3. Runs `SCSA.py` and gives every cell the cell type with the highest Z-score of its cluster, or
   "Unknown". Only the test cells are written (`cell_id`, `true_labels`, `predicted_labels`);
   the marker table and the SCSA result are kept next to the output.

With `LABEL_CONSTRAINT`, only the candidates that count for the expected cell types compete
(`bin/util/label_constraint.py`). `src/map_scsa_to_cells.py` is a standalone tool that the
pipeline does not use.

## SCSA settings

| Option | Value | Reason |
|---|---|---|
| `-d` | `whole_v2.db` | SCSA's database with CellMarker version 2 |
| `-s` | `scanpy` | format of the marker table |
| `-f`, `-p` | 1.5, 0.01 | fold change and p-value thresholds of SCSA's scanpy example |
| `-E` | set | the markers are gene symbols (from `var['gene_names']`) |
| `-g` | `Human` or `Mouse` | from `species`; SCSA compares this value case-sensitively |
| `-k` | `scsa_tissue` | CellMarker tissue names, comma-separated, chosen per dataset; default `All` |
| `-T`, `-t`, `-m` | `normal`, `cellmarker`, `txt` | normal cells, CellMarker, text output |
