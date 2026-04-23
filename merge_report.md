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

## Design + plan references

- Design: `docs/plans/2026-04-23-merge-blind-into-optimize-llm-module-design.md`
- Implementation plan: `docs/plans/2026-04-23-merge-blind-into-optimize-llm-module.md`
