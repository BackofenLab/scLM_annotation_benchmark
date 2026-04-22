# Subset Strategy, Preprocess Merge, and Memory Redesign

**Status:** Design approved, awaiting implementation plan (writing-plans).

## Overview

Three coordinated changes to the data pipeline:

1. **Add `params.subset_strategy`** (`stratified` | `random`) so subsetting can preserve the natural class imbalance instead of forcing proportional representation per cell type.
2. **Merge `ontology_mapping` into `preprocess`** so the pipeline collapses from 4 data steps (download → preprocess → ontology_mapping → subset) to 3 (download → preprocess → subset). Ontology mapping stays optional, resolved per-dataset from the YAML with a global fallback.
3. **Reduce memory footprint** across all three data steps: apply the existing preprocessing memory plan, switch `download.py` cache validation and column checks to backed mode, and add backed-mode slicing to `subset.py`.

## Scope

### In scope

- `nextflow.config`: add `params.subset_strategy`.
- `main.nf`: delete the `ontology_mapping` process; thread `annotation_key` + `apply_ontology_mapping` into `preprocess`; thread `strategy` into `subset` and into the `run_model` prediction cache path.
- `bin/preprocess.py`: new `--annotation_key`, `--apply_ontology_mapping`, `--dataset` args; optional mapping call at the end.
- `bin/subset.py`: new `--strategy` arg; new `random_subset()` function; backed-mode load; strategy-tagged QC filenames.
- `bin/download.py`: backed-mode cache validation and column verification.
- `bin/util/pp_utils.py`: apply the 6 fixes from `2026-03-14-preprocessing-memory-optimization.md`.
- `bin/ontology_map.py`: delete.
- `README.md`: update pipeline-process list and params table; add cache-cleanup note.

### Out of scope

- Dropping unneeded `obs`/`var` columns in preprocess (deferred file-size optimization).
- Switching storage from h5ad to zarr.
- Evaluation / `combine_results` / model code.

## Design decisions (recap)

| Decision | Choice | Rationale |
|---|---|---|
| Random-mode semantics | Pure uniform, no `min_cells_per_type` filtering | Preserve original proportions strictly (in expectation) |
| Strategy flag location | Global `params.subset_strategy` in `nextflow.config` | Simple; not dataset-dependent |
| Ontology mapping placement | Final optional step inside `preprocess.py` | Cheaper than a separate Nextflow process; YAML flag is stable per dataset, so cache staleness is not a real concern |
| Ontology flag resolution | Unchanged — dataset YAML > `params.apply_ontology_mapping` | Matches existing convention |
| Subset cache disambiguation | Strategy in filename | `<dataset>_sub<size>_<strategy>.h5ad` is consistent with `<dataset>_sub<size>`; legible in logs |
| Prediction cache disambiguation | Strategy in path | `cache/models/predictions/<model>/<dataset>/<subset_size>_<strategy>/` — without this, toggling strategy reuses stale predictions |

## Pipeline DAG after changes

```
download → preprocess (QC + filter + norm/log + ensembl + optional ontology) → subset (stratified|random) → prepare_model / run_model → evaluate → combine
```

Artifacts:

| Path | Content |
|---|---|
| `cache/datasets/raw/<dataset>_raw.h5ad` | Raw download (unchanged) |
| `cache/datasets/preprocessed/<dataset>_pp.h5ad` | Preprocessed, optionally ontology-mapped |
| `cache/datasets/subset/<dataset>_sub<size>_<strategy>.h5ad` | Sampled subset, strategy-tagged |
| `cache/subset_qc/<dataset>/sub<size>_<strategy>/` | QC CSV + PDF per (size, strategy); filenames also carry `_<strategy>` |
| `cache/models/predictions/<model>/<dataset>/<size>_<strategy>/` | Model predictions, strategy-tagged |
| `cache/datasets/mapped/` | **Gone.** Stale contents from prior runs are harmless; README advises deletion. |

## Per-file changes

### `nextflow.config`

- Add `params.subset_strategy = 'stratified'`.

### `main.nf`

- Delete the `ontology_mapping` process block.
- `preprocess` process input tuple becomes `(raw_file, dataset, cfg_yaml, cache_dir, annotation_key, apply_mapping)`; script passes `--annotation_key`, `--apply_ontology_mapping`, `--dataset`.
- Workflow body: move the mapping-resolution block (dataset-YAML > global-fallback) from the old `ontology_input` builder up onto `preprocess_input`. Remove `ontology_out`.
- `subset` input tuple gains `strategy`. Output filename becomes `${dataset}_sub${subset_size}_${strategy}.h5ad`; `tag` and `qc_dir` include strategy.
- `run_model` prediction cache path becomes `${cache_dir}/models/predictions/${model}/${dataset}/${subset_size}_${strategy}/`. Pass `strategy` through `model_input`.

### `bin/preprocess.py`

- Add CLI args: `--annotation_key <str>`, `--apply_ontology_mapping true|false`, `--dataset <name>`.
- After existing preprocessing (`ensure_raw_counts_layer` → QC → filter → `ensure_norm_log` → `remove_infinite_values` → `ensure_ensembl_ids`) and before writing the output, apply ontology mapping:
  - If `apply_ontology_mapping` is true: call `apply_ontology_mapping_to_dataset(adata, annotation_key=..., cache_dir=..., verbose=True, origin=dataset)` from `util.download_helpers`.
  - Else: pass-through. If `original_<annotation_key>` exists, restore from it; otherwise no-op.
- Import `apply_ontology_mapping_to_dataset` only when needed (optional import inside the branch).

### `bin/subset.py`

- Add CLI arg: `--strategy stratified|random` (required, no default — main.nf always passes it).
- New function `random_subset(adata, subset_size)`:
  - `idx = np.random.default_rng(RANDOM_SEED).choice(adata.n_obs, size=min(subset_size, adata.n_obs), replace=False)`
  - Return `adata[idx].to_memory() if adata.isbacked else adata[idx].copy()` plus a QC DataFrame (same schema as stratified, with `dropped=False` everywhere).
- Dispatch in `main()`:
  - Validate strategy against `{'stratified', 'random'}`; exit 1 with clear message on invalid input.
  - If `strategy == 'random'` and `--min_cells_per_type` differs from the default, log `"random mode: ignoring --min_cells_per_type"`.
- **Backed-mode load**: replace `anndata.read_h5ad(args.input)` with `anndata.read_h5ad(args.input, backed='r')`. Only materialize selected rows (`adata[indices].to_memory()`). For stratified, read `adata.obs[annotation_key]` directly — `.obs` is materialized in backed mode, so no matrix touch until the slice.
- QC output filenames become `cell_type_distribution_<strategy>.csv` and `subset_qc_plots_<strategy>.pdf`. QC dir is driven by `--qc_dir` from `main.nf` and already includes `sub<size>_<strategy>/`.

### `bin/download.py`

- `_cache_valid()` (line 95): replace `read_adata(dest)` with either `anndata.read_h5ad(path, backed='r')` (close immediately) or a structural check via `h5py.File(path, 'r')` looking for the `X` group. O(1) RAM instead of O(full matrix).
- `verify_columns()` (line 339): use `anndata.read_h5ad(path, backed='r')` and only read `.obs.columns` (columns list is cheap; no need to materialize values). O(metadata) RAM.

### `bin/util/pp_utils.py`

Apply all 6 tasks from `docs/plans/2026-03-14-preprocessing-memory-optimization.md`:

1. `is_raw_counts`: sparse `.data` path, no `.toarray()`.
2. `is_normalized`: `X.sum(axis=1)` works on sparse natively.
3. `is_log_transformed`: sparse-compatible via `.data.min()` / `.data.max()`.
4. `ensure_norm_log`: pass `adata.X` directly to the check functions — no dense copy.
5. `ensure_raw_counts_layer`: reduce one redundant `.copy()` in Case 1 (user-specified layer).
6. `nextflow.config` local profile: already at `32 GB` (task 6 superseded by existing value).

That plan is the canonical source for these fixes; the implementation plan for this design links to it rather than duplicating content.

### `bin/ontology_map.py`

Delete. Its sole caller (the `ontology_mapping` process in `main.nf`) is gone; the function it wraps (`apply_ontology_mapping_to_dataset`) is used directly from `preprocess.py`.

### `README.md`

- Pipeline-process list: remove step 3 (ontology_mapping); update step 2 to mention optional ontology mapping; renumber.
- Config params table: add `subset_strategy` row.
- New cache-invalidation note: advise deleting `cache/datasets/mapped/` and any pre-existing `cache/datasets/subset/<dataset>_sub<size>.h5ad` files (without strategy suffix) after upgrading.

## Error handling

- **Invalid `--strategy`**: `subset.py` validates early and exits with a clear message.
- **Random mode + custom `min_cells_per_type`**: log INFO note; continue.
- **`subset_size >= n_obs`**: return full dataset in both strategies (existing behavior extended to random).
- **Ontology mapping failure** inside `preprocess`: exceptions propagate, failing the `preprocess` process. `ONTOLOGY_MAPPING_WARNING.txt` side-channel (`main.nf:326–341`) continues to work because `apply_ontology_mapping_to_dataset` still writes it.
- **Stale caches**: no auto-cleanup. Old `cache/datasets/mapped/` and un-tagged `<dataset>_sub<size>.h5ad` files are ignored by the new layout.

## Testing / verification

No test suite today. Verification is manual:

1. **Schema parity**: run preprocess on one dataset (e.g., Zeisel) with `apply_ontology_mapping=true`. Confirm `<dataset>_pp.h5ad` has the same `obs` schema (including `original_<annotation_key>` backup) as the two-step pipeline produced.
2. **Mapping pass-through**: toggle `apply_ontology_mapping: false` in one YAML; confirm labels match the `original_<annotation_key>` backup when present.
3. **Stratified vs random on the same dataset/size**: run both; confirm `<dataset>_sub<size>_stratified.h5ad` and `<dataset>_sub<size>_random.h5ad` coexist, QC CSVs show the expected distributions (proportional vs natural), and prediction caches live under distinct `.../<size>_<strategy>/` paths.
4. **Cache hits**: re-run both strategies with caches in place; confirm `storeDir` short-circuits both.
5. **End-to-end**: `celltypist_blind` on one dataset through both strategies; confirm `evaluate_annotation` + `combine_results` produce distinct outputs per strategy.
6. **Memory**: run HIHA (1.8M × 32K) preprocess locally; confirm peak RAM stays under ~20 GB (validates the `pp_utils.py` fixes). Run download step twice; confirm second run doesn't spike RAM during cache validation (validates backed-mode fix).
7. **Subset backed-mode**: on a large preprocessed file, confirm `subset.py` peak RAM is proportional to the subset size, not the full dataset size.

## Implementation ordering

Suggested commit sequence (each step green before the next):

1. **`pp_utils.py` memory fixes** — follow the existing plan end-to-end. Standalone, unblocks nothing but lowest-risk to land first.
2. **`download.py` backed-mode cache validation + column check** — isolated to one file.
3. **Ontology merge**: `preprocess.py` gains the flag + mapping call; delete `ontology_mapping` process in `main.nf`; delete `bin/ontology_map.py`; update docs for the reduced pipeline.
4. **Subset strategy**: `subset.py` gains `--strategy` + `random_subset` + backed-mode load; `main.nf` threads `strategy` into `subset` and `run_model` cache path; `nextflow.config` gains `params.subset_strategy`; update params table in README.
5. **README cache-cleanup note.**

Each step is independently runnable end-to-end — no split commits leave the pipeline broken.
