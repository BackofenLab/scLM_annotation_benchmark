# Merge report — `origin/blind` → `optimize-llm-module`

Date: 2026-04-23
Integration branch: `integrate-blind`
Merge commit: `bfe3383`
Follow-up commits: `ec6f9e2` (docker_local profile)

## Branches

| Ref | SHA | Role |
|---|---|---|
| merge base | `7d656f1` | Common ancestor (old `blind` tip) |
| optimize-llm-module | `f59fc96` | "ours" — 54 commits ahead of base |
| origin/blind | `c0ceca0` | "theirs" — 38 commits ahead of base |

Local `blind` (stale, identical to merge base) and `origin/blind` are **not** modified. Integration lives on `integrate-blind`.

## Priorities applied

**Kept from optimize-llm-module:**
- OpenRouter LLM modules (unified `run_model` lane; `run_model_throttled` removed)
- download → preprocess → subset pipeline (new `bin/subset.py`, `subset_strategy` param, strategy-suffixed cache paths)
- `storeDir` caching layer
- Backed-mode h5ad cache validation in `bin/download.py`
- QC plots as PNG (lightweight) + macOS-safe `ax.violinplot` workaround
- `scipy.sparse` handling in `is_raw_counts`
- `ONTOLOGY_REVIEW_PENDING` marker system (`bin/evaluate.py` + `main.nf` workflow.onComplete)
- `errorStrategy 'ignore'` on `evaluate_annotation`

**Kept from origin/blind:**
- Docker/container env handling: `envs/Dockerfile`, `envs/environment.yml`, `params.container_image`, per-process `container` directives, `docker`/`apptainer` profiles
- BioMart mirror fallback in `bin/util/pp_utils.py` (Ensembl mirrors: www, useast, asia)
- Figshare API URL fix (`api.figshare.com/v2/file/download/`)
- Gene-name extraction via `ensure_gene_names_column` (candidates: `gene_names`, `feature_names`, `feature_name`, `var_names` with heuristic)
- `--organism` CLI arg + mismatch logging in `bin/preprocess.py`
- Celltypist model fixes: Organ Atlas URL fallback, `Adult_Human_Bone_marrow.pkl` (lowercase) name correction
- `helix` and `imbi_gpu` profiles in `nextflow.config`
- slurm ignore patterns in `.gitignore` (`slurm-*`, `*.sbatch`)
- Pip deps in `envs/cell_types.yml` and `envs/download.yml` (`boltons`, `cytoolz`, `frozendict`)

## Conflict resolutions (7 files)

| File | Bucket | Resolution |
|---|---|---|
| `datasets/Zeisel.yml` | datasets | **ours** — kept quoted organism, dropped reintroduced `subset_size` |
| `datasets/tabula_sapiens_bonemarrow.yml` | datasets | **ours (deletion)** — superseded by `_CxG` variant |
| `bin/download.py` | preprocess core | **ours** — backed-mode validation, no inline subsetting, no inline ontology mapping (these live in dedicated processes now) |
| `bin/preprocess.py` | preprocess core | **theirs** — `ensure_gene_names_column` helper call and richer organism mismatch logging integrate with auto-merged `--organism` CLI + `args.organism` default |
| `.gitignore` | pipeline | **theirs** (union) — slurm ignore patterns added |
| `main.nf` | pipeline | **ours** — removed `run_model_throttled`; container directives (theirs) auto-merged separately on every process; `extract_cell_types` process kept as dead code (commented-out call on theirs too) |
| `nextflow.config` | pipeline | **mixed** — ours 16 GB local memory + conda comment, theirs `docker.enabled = false`. Stale `run_model_throttled` references in docker/apptainer profiles cleaned up. `docker_local` profile added as follow-up |

## Auto-merge spot checks (BOTH sides touched, no conflict)

| File | Why safe |
|---|---|
| `bin/util/pp_utils.py` | Ours' macOS fixes (Agg/violinplot, PNG QC, scipy.sparse) and theirs' additions (BioMart mirror iteration, `feature_name` in gene-name candidates) touch disjoint sections. Both preserved |
| `datasets/tabula_sapiens_bonemarrow_CxG.yml` | Ours dropped `subset_size`; theirs fixed celltypist model-name casing. Both preserved |
| `models/celltypist_blind/src/predict.py` | Ours fixed `preprocess_data` input scale (restore raw counts); theirs added `_download_model` Organ Atlas fallback and changed one call site. Different functions |
| `models/langcell-fewshot/src/predict.py` | Ours adds raw-count restoration before Geneformer-style tokenization; theirs reorders two items in a `features_to_keep` list (functional no-op) |
| `models/openrouter_LLM_unblind/src/predict.py` | Theirs deletes a single `logger.info("Loading data from ...")` line; ours is a 487-line refactor that preserved `load_data` around that line. Merged file == ours minus that line |

All Python files touched by the merge pass `py_compile`.

## theirs-only files (untouched on our side since merge base)

Taken verbatim from `origin/blind`:
- `models/c2s_scale_{2B,27B}_{blind,unblind}/src/predict.py`
- `models/cellwhisperer_blind/src/{prepare,run}.sh`, `models/cellwhisperer_unblind/src/run.sh`
- `models/instructcell_{blind,unblind}/src/predict.py`
- `envs/Dockerfile`, `envs/environment.yml` (new)
- `envs/cell_types.yml`, `envs/download.yml` (dep additions)
- `envs/models/cellwhisperer_{blind,unblind}.yml` (duplicate cleanup)

## ours-only files

Taken verbatim from `optimize-llm-module`:
- `models/openrouter_LLM_blind/src/predict.py`, `models/scelmo/src/predict.py`
- `bin/aggregate.py`, `bin/subset.py` (new)
- `docs/plans/*` (design + implementation plan + several earlier pre-merge plans)

## Follow-ups / open items

- **Smoke test not yet green.** First run attempted with `-profile docker_local` but Docker Desktop daemon was not active (socket missing). Pipeline logic itself untested end-to-end at time of writing. Plan: restart Docker Desktop and re-run, or re-run with `-profile local` (conda) as the conda-native alternative.
- **`extract_cell_types`** process in `main.nf` + `bin/extract_cell_types.py` + `envs/cell_types.yml` are dead code (no workflow call). Kept to match theirs; candidate for follow-up cleanup.
- **`docker_local` profile** added specifically for laptop runs (no GPU, 4 cpu / 16 GB). The original `docker` profile (GPU + 12 cpu / 48 GB) is preserved for cluster/server use.
- **Local `nextflow.config` tweaks** (pre-merge run-specific edits — `datasets`, `subset_size`, `subset_strategy`) were stashed as `stash@{0}` before the merge. Discard or reapply as needed.
- **Promotion decision pending:** fast-forward `optimize-llm-module` to `integrate-blind`, or PR review. Not part of this integration.

## Main changes on `optimize-llm-module` (since merge base `7d656f1`)

For context — what the 54 pre-merge commits on our side actually shipped:

- **OpenRouter LLM rework.** Multi-LLM support in a single module, cleaner parallel execution, default model list, prompt-template tightening for smaller models. `bin/aggregate.py` fixes for multi-LLM JSONs (per-model keys, model-name parsing with underscores). Model-name fixes in aggregated results.
- **Pipeline split: `download → preprocess → subset`.** `bin/download.py` now only downloads + verifies (storeDir cached); subsetting moved to a new `bin/subset.py` with `stratified` and `random` strategies; ontology mapping merged as an optional final step inside `preprocess`. `subset_size` moved out of per-dataset YAMLs into `nextflow.config` (with `subset_categories` for per-model-category sizing). `subset_strategy` threaded through the workflow; prediction cache paths include the strategy.
- **storeDir caching.** Dropped the private flat cache; rely on Nextflow `storeDir` for cross-run reuse. Backed-mode h5ad validation keeps cache-hit checks O(1) RAM.
- **Preprocess invariants.** Guarantee `adata.var['gene_names']` and `adata.obs['organism']` post-preprocess for downstream models. Raw-counts detection avoids dense conversion on sparse matrices. Output h5ad size reduced; QC plots switched to high-res PNG.
- **Model input-scale fix.** `celltypist`, `scelmo`, and `langcell` now restore raw counts from `adata.layers['counts']` before their own normalize/tokenize step, avoiding the double-log1p that silently degraded predictions on already-log-normalized inputs.
- **Evaluation + aggregation.** New `ONTOLOGY_REVIEW_PENDING` marker surfaced by `evaluate.py` and reported/archived in `workflow.onComplete`; `errorStrategy 'ignore'` on eval so one failure does not abort the run. Per-model confusion matrices exported as PDF. Heatmap in `aggregate.py`. Unmapped-ratio thresholds tuned.
- **macOS stability.** `KMP_DUPLICATE_LIB_OK=TRUE` in `nextflow.config` beforeScript avoids libomp double-init aborts. `ax.violinplot` replaces `sc.pl.violin` to avoid the seaborn/macOS segfault in parallel runs.

## Design + plan references

- Design: `docs/plans/2026-04-23-merge-blind-into-optimize-llm-module-design.md`
- Implementation plan: `docs/plans/2026-04-23-merge-blind-into-optimize-llm-module.md`

---

## Post-merge feature — `download_from_zenodo` flag

Date: 2026-04-23
Branch: `integrate-blind` (commits `990d009` … `8891a66`)

Opt-in shortcut that replaces the `download → preprocess → subset` chain with a direct fetch of pre-processed, pre-subsetted `.h5ad` files from public Zenodo record `19707183`. Default is `false`, so no behaviour change for existing users.

### Commits

| SHA | Scope |
|---|---|
| `990d009` | `feat(config)`: `params.download_from_zenodo = false` |
| `149de05` | `feat(datasets)`: `datasets/zenodo_manifest.yml` pins 16 files (MD5 + size) |
| `81794ad` | `refactor(bin)`: lift `new_session` / `stream_to_file` into `bin/util/http.py` |
| `fdbf19f` | `feat(bin)`: `bin/fetch_zenodo_subset.py` — download, MD5 verify, annotation-key check |
| `0d2609a` | `feat(nextflow)`: `download_from_zenodo` process (storeDir into `cache/datasets/subset/`) |
| `f1683bf` | `feat(nextflow)`: workflow `if/else` on the flag + upfront manifest validator |
| `8891a66` | `docs(readme)`: parameter row + "Fetching preprocessed data from Zenodo" section |

### Manifest coverage

Datasets `HFRA`, `HIHA`, `human_pancreas`, `Zeisel` × sizes `1000`, `10000` × strategies `stratified`, `random` = 16 files. Requests outside this set abort upfront with the list of missing filenames.

### Verification

- `nextflow inspect main.nf` parses; `download_from_zenodo` registered.
- Negative path (`sub500`) aborts with expected error before any process launches.
- Positive path (`HIHA`, `sub1000`, `stratified`, `celltypist_blind`): full pipeline completes; `cache/datasets/subset/HIHA_sub1000_stratified.h5ad` (12 MB) MD5-verified; re-run hits `storeDir` without re-downloading.
- Integrity regression (MD5 tamper) not exercised.

### Notes

- `main.nf:265` uses `"${...}".toString()` because `HashSet<String>.contains(GString)` returns false on equal content.
- CLI overrides for list/map params (`--datasets '[...]'`) don't round-trip through Nextflow; use `-params-file` for ad-hoc tests.
- Zenodo files were produced with `apply_ontology_mapping=true`; `params.apply_ontology_mapping` is ignored on the fetch path but still forwarded to eval.

### References

- Design: `docs/plans/2026-04-23-download-from-zenodo-design.md`
- Implementation plan: `docs/plans/2026-04-23-download-from-zenodo.md`
