# Merge `origin/blind` into `optimize-llm-module` — design

Date: 2026-04-23
Status: design approved, implementation pending

## Goal

Integrate the work on `origin/blind` (tip `c0ceca0`) into `optimize-llm-module`
without losing either branch's distinct contributions. `origin/blind` is the
live remote branch that will soon be merged into `main`; `optimize-llm-module`
contains parallel work on LLM modules and the download/preprocess/subset
pipeline. Both must end up in one branch.

Non-goals: deleting or force-pushing `blind` / `origin/blind`. Promoting the
integration result into `optimize-llm-module` or `main` is a separate decision
after this merge is validated.

## Inputs to the merge

- Merge base: `7d656f1` (old tip of the original `blind`, now also an ancestor
  of `optimize-llm-module`).
- `optimize-llm-module` is 54 commits ahead of the base (subset redesign with
  `bin/subset.py`, preprocess+ontology merge, storeDir caching, openrouter LLM
  rework, KMP_DUPLICATE_LIB_OK fix, PNG QC plots, etc.).
- `origin/blind` is 38 commits ahead of the base (Docker/containers via
  `add-containers` merge, BioMart mirror fallback, gene-name extraction from
  `feature_name` / `var_names`, backed-mode reads for memory, celltypist
  model-name corrections + organ-atlas fallback, Figshare API fix, `organism`
  added to obs, c2s_scaled HIHA fix, reset of `nextflow.config` and conda
  envs).

Local `blind` (`7d656f1`) is stale (== merge base). It is not involved in the
merge.

## Priorities for conflict resolution

From user, broad rather than per-file:

- **Prefer `optimize-llm-module`** for: LLM modules (openrouter), and the
  download → preprocess → subset mechanism plus every file materially
  affected by it (`bin/download.py`, `bin/preprocess.py`, `bin/subset.py`,
  `bin/util/pp_utils.py`, related parts of `main.nf` / `nextflow.config`).
- **Prefer `origin/blind`** for: Docker/container environment handling, and
  model-side bug fixes (gene-name extraction, BioMart fallback, celltypist
  fixes, Figshare API, etc.).

Policy is not exhaustive; per-file calls happen in the review loop (below).

## Approach

Considered:

- A. Blind merge, resolve file-by-file on the fly — inconsistent calls under
  pressure.
- B. Pre-committed per-area policy, then merge — user can only provide broad
  guidance, so a detailed policy can't be locked in up front.
- A′. Merge, bucket, review together — the chosen approach.

## Plan — Section 1: branch and merge mechanics

1. `git fetch origin` to confirm `origin/blind` tip.
2. From `optimize-llm-module` HEAD, cut a new branch `integrate-blind`.
3. On `integrate-blind`: `git merge --no-ff origin/blind`. Expected: many
   conflicts.
4. `blind` (local) and `origin/blind` are not touched: no delete, no force-push.
5. Rollback is `git merge --abort` or `git branch -D integrate-blind`.

Promotion of `integrate-blind` back into `optimize-llm-module` (fast-forward
vs. PR review) is deferred until after smoke test passes.

## Plan — Section 2: conflict bucketing and review loop

After the merge produces conflicts, group unresolved files by area:

1. **Pipeline orchestration** — `main.nf`, `nextflow.config`, `.gitignore`
2. **Download / preprocess / subset core** — `bin/download.py`,
   `bin/preprocess.py`, `bin/subset.py`, `bin/util/pp_utils.py`, ontology
   helpers as needed
3. **Aggregation / evaluation** — `bin/aggregate.py`, `bin/evaluate.py`,
   `bin/extract_cell_types.py`
4. **Non-model envs** — `envs/Dockerfile`, `envs/environment.yml`,
   `envs/download.yml`, `envs/cell_types.yml`, `envs/evaluation.yml`
5. **Model envs** — `envs/models/*.yml`
6. **LLM model scripts** — `models/openrouter_LLM_{blind,unblind}/src/*`
7. **Non-LLM model scripts** — everything else under `models/*/src/*`
8. **Dataset configs** — `datasets/*.yml`
9. **Cache artifacts** — `cache/plots/*.pdf`, `cache/models/predictions/**`,
   `cache/all_cell_types.txt`: low-signal, resolve en bloc (most likely
   `--ours`, regenerate if needed).

Review loop per bucket:

- Use `git diff --name-only --diff-filter=U` to see what's actually conflicted
  inside the bucket.
- For each conflict, present relevant hunks + recommendation based on the
  priorities above. User decides accept / flip / modify.
- Stage (`git add`) after each resolution. No commit until all conflicts are
  resolved.

Ordering: bottom-up by risk. Cache artifacts and datasets first, then envs,
then models, then `bin/`, then `main.nf` + `nextflow.config` last (they gain
the most from the rest of the picture being settled).

Files that exist on only one side (`bin/subset.py`, `docs/plans/*` only on our
side; `envs/Dockerfile`, `envs/environment.yml` only on origin/blind) will
auto-merge without conflict.

## Plan — Section 3: validation and finalization

Smoke test:

```
nextflow run main.nf -profile local \
  --datasets HFRA --subset_size 1000 --subset_strategy random
```

with `nextflow.config` configured so only `celltypist` (blind variant) runs.
Pass = pipeline reaches evaluate/aggregate and produces predictions for HFRA.

If smoke test fails:

- Fix in small named commits on top of `integrate-blind` (do NOT amend the
  merge commit).
- If a wrong resolution is the cause, `git reset --hard <merge-commit>` and
  redo only the affected bucket. Don't abort the entire merge.

Merge commit message (shape):

```
merge: integrate origin/blind into optimize-llm-module

Prioritized from optimize-llm-module:
- openrouter LLM modules
- download -> preprocess -> subset pipeline + storeDir caching

Prioritized from origin/blind:
- Docker/container env handling
- model-side fixes (gene_name extraction, BioMart fallback, celltypist
  model-name fallback, Figshare API fix, etc.)

Smoke-tested: HFRA, subset_size=1000, random strategy, celltypist-only.
```

After successful smoke test:

- `integrate-blind` kept as-is for review.
- Promotion into `optimize-llm-module` (fast-forward vs. PR) is a separate
  decision.
- `blind` (local) and `origin/blind` untouched.

Out-of-scope risks noted, not handled by this plan:

- `origin/blind` moving during the work — re-fetch and merge the delta, or
  restart, depending on scale.
- A different dataset × model combo breaking post-merge — follow-up, not a
  merge blocker.
