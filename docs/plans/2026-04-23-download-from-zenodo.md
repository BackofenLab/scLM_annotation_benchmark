# download_from_zenodo Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `params.download_from_zenodo` flag that replaces the `download → preprocess → subset` chain with a direct fetch of pre-processed, pre-subsetted AnnData files from the public Zenodo record 19707183.

**Architecture:** A new Nextflow process `download_from_zenodo` reads a pinned YAML manifest, fetches the matching `{dataset}_sub{size}_{strategy}.h5ad` via a thin Python script, MD5-verifies it, and emits into `cache/datasets/subset/` via `storeDir` — so downstream processes consume it identically to a `subset`-produced file. The `workflow {}` block gains an upfront validator plus a single `if/else` branch on `params.download_from_zenodo`; everything after `subset_out` is unchanged.

**Tech Stack:** Nextflow DSL2, Python 3.11 (`requests`, `anndata`, `pyyaml`, `tqdm`), the existing `envs/download.yml` conda env.

**Design doc:** `docs/plans/2026-04-23-download-from-zenodo-design.md`

---

## Ordering & principles

- No automated test suite exists for this project — verification is manual smoke tests against the live Zenodo endpoint. Each task ends with an explicit verification command and expected output.
- Each task is independently mergeable. Tasks 1–4 can land without touching `main.nf` at all, so the fetcher can be verified standalone before the workflow flips.
- Commit after each task. Conventional-commit style matching recent history (`feat(x): ...`, `refactor(x): ...`, `docs(x): ...`).

---

## Task 1: Add the config flag

**Files:**
- Modify: `nextflow.config:118-124` (the `params.*` block near the bottom)

**Step 1: Add the flag**

Edit `nextflow.config`. After the line `params.apply_ontology_mapping = true` block (line ~118), add:

```groovy
params.download_from_zenodo = false  // if true, fetch preprocessed+subsetted files from Zenodo instead of running download→preprocess→subset
```

**Step 2: Verify the flag parses**

Run:
```bash
nextflow run main.nf -profile local --download_from_zenodo true --help 2>&1 | head -3
```

Expected: pipeline startup messages or the existing help banner, no Groovy parse error about `download_from_zenodo`. (The flag is read but nothing uses it yet.)

**Step 3: Commit**

```bash
git add nextflow.config
git commit -m "feat(config): add download_from_zenodo flag (unused)"
```

---

## Task 2: Create the Zenodo manifest

**Files:**
- Create: `datasets/zenodo_manifest.yml`

**Step 1: Write the manifest**

Create `datasets/zenodo_manifest.yml` with the exact contents below (copied from the design doc — do not edit MD5s or sizes):

```yaml
record_id: 19707183
base_url: https://zenodo.org/api/records/19707183/files
files:
  HFRA_sub1000_random.h5ad:      {md5: 93c81d9b2d5dd6bad2c17546d08020f1, size: 9957030}
  HFRA_sub1000_stratified.h5ad:  {md5: 5827c0447d357b34718f6978a122c2d2, size: 9952202}
  HFRA_sub10000_random.h5ad:     {md5: ecd1055e65202b3af4dae4070171e159, size: 68381925}
  HFRA_sub10000_stratified.h5ad: {md5: d984db128343217043c7ca1ce530ab7d, size: 67934640}
  HIHA_sub1000_random.h5ad:      {md5: e5b857a6f58cf84ad4c6727614b19d70, size: 12617907}
  HIHA_sub1000_stratified.h5ad:  {md5: 27b3c54c58c388cef04bd01a22cd5d56, size: 12989196}
  HIHA_sub10000_random.h5ad:     {md5: c50f3b07220d63b8a9139ebd46fa39a1, size: 86201739}
  HIHA_sub10000_stratified.h5ad: {md5: 1aed57b7fa68dee0d8040ebbff35ff97, size: 85470007}
  Zeisel_sub1000_random.h5ad:      {md5: 8205caa8f216400a08c717bdef1816ea, size: 12226401}
  Zeisel_sub1000_stratified.h5ad:  {md5: 3053f49a9a3192f03835481d2f35bce7, size: 13009397}
  Zeisel_sub10000_random.h5ad:     {md5: c762bf405fc6683fa0bca99f6b0955c5, size: 82262418}
  Zeisel_sub10000_stratified.h5ad: {md5: d3ea4410f0ef503681eb24b2fdd56d73, size: 80287216}
  human_pancreas_sub1000_random.h5ad:      {md5: e06145ec65c99c321b5309261f9c31c7, size: 11346193}
  human_pancreas_sub1000_stratified.h5ad:  {md5: 38a73f4d6a3e5563ea53f7ddc65e3ac8, size: 11384232}
  human_pancreas_sub10000_random.h5ad:     {md5: 8eea6d1c20d7ddf330ab5c0c57900ba8, size: 101312599}
  human_pancreas_sub10000_stratified.h5ad: {md5: e9a60fc6332fe5e5b0ca02787a56a956, size: 101764721}
```

**Step 2: Verify the manifest parses and keys round-trip**

Run:
```bash
python -c "import yaml; m = yaml.safe_load(open('datasets/zenodo_manifest.yml')); print(m['record_id']); print(len(m['files']), 'files'); print(sorted(m['files'].keys())[0])"
```

Expected output:
```
19707183
16 files
HFRA_sub1000_random.h5ad
```

**Step 3: Verify one URL is reachable (optional sanity)**

Run:
```bash
curl -sI "https://zenodo.org/api/records/19707183/files/HFRA_sub1000_random.h5ad/content" | head -3
```

Expected: `HTTP/2 200` (after any redirect), or at minimum a `200`/`302` status line.

**Step 4: Commit**

```bash
git add datasets/zenodo_manifest.yml
git commit -m "feat(datasets): add zenodo_manifest.yml pinning record 19707183"
```

---

## Task 3: Extract shared HTTP helpers into `bin/util/http.py`

**Rationale:** `bin/fetch_zenodo_subset.py` (Task 4) and `bin/download.py` both need streaming download + session setup. Lift the two helpers into a shared module so the fetcher doesn't duplicate them.

**Files:**
- Create: `bin/util/http.py`
- Modify: `bin/download.py:118-148` (removing `_stream_to_file` and `_new_session`)

**Step 1: Read the current helpers**

Read `bin/download.py` lines 118–148 to see the exact bodies of `_stream_to_file` (lines 118–128) and `_new_session` (lines 143–148). These are the only two we're moving.

**Step 2: Create `bin/util/http.py`**

Write `bin/util/http.py` with:

```python
"""Shared HTTP helpers for downloader scripts."""
from __future__ import annotations
from pathlib import Path

import requests
from tqdm import tqdm


def new_session() -> requests.Session:
    """Return a requests.Session with a reasonable User-Agent."""
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; sclfm-downloader/1.0)"})
    return s


def stream_to_file(resp: requests.Response, dest: Path) -> int:
    """Stream *resp* to *dest* with progress bar; return bytes written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total, written = int(resp.headers.get("content-length", 0)), 0
    with open(dest, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in resp.iter_content(1 << 20):
            if chunk:
                fh.write(chunk)
                written += len(chunk)
                bar.update(len(chunk))
    return written
```

Names are changed from `_stream_to_file` / `_new_session` to `stream_to_file` / `new_session` since they're now part of a module's public API.

**Step 3: Update `bin/download.py` to import from the new module**

In `bin/download.py`:

- After the `import requests` / `from tqdm import tqdm` block near the top, add:
  ```python
  from util.http import new_session, stream_to_file
  ```
- Delete the local `def _new_session(...)` and `def _stream_to_file(...)` definitions.
- Replace all call sites of `_new_session()` with `new_session()` and `_stream_to_file(...)` with `stream_to_file(...)`. There are four call sites: `stream_download` (one `_new_session`, one `_stream_to_file`), `stream_download_figshare` (one `_new_session`, one `_stream_to_file`), and the inside of `stream_download_figshare`'s HTML-retry path (one `_stream_to_file`). Search with `grep -n "_new_session\|_stream_to_file" bin/download.py` to catch all of them.

**Step 4: Verify `download.py` still runs**

Run:
```bash
python -c "import sys; sys.path.insert(0, 'bin'); import download; print('import ok'); print(download.stream_download.__name__)"
```

Expected:
```
import ok
stream_download
```

**Step 5: Verify nothing else imports the old private names**

Run:
```bash
grep -rn "_new_session\|_stream_to_file" bin/ || echo "clean"
```

Expected: `clean`.

**Step 6: Commit**

```bash
git add bin/util/http.py bin/download.py
git commit -m "refactor(bin): lift HTTP helpers into bin/util/http.py"
```

---

## Task 4: Create `bin/fetch_zenodo_subset.py`

**Files:**
- Create: `bin/fetch_zenodo_subset.py`

**Step 1: Write the script**

Create `bin/fetch_zenodo_subset.py` with:

```python
#!/usr/bin/env python
"""
Fetch a preprocessed + subsetted AnnData file from the pinned Zenodo record.

Reads the manifest (`datasets/zenodo_manifest.yml`), resolves the expected
filename from (dataset, subset_size, strategy), downloads it, MD5-verifies,
and confirms the expected annotation column exists. Exits non-zero on any
failure so Nextflow aborts and `-resume` retries cleanly (storeDir never
sees a partial file).
"""
from __future__ import annotations
import argparse, hashlib, sys, yaml
from pathlib import Path

import anndata

# bin/ is on PYTHONPATH when invoked by Nextflow
from util.http import new_session, stream_to_file


def log_info(msg: str) -> None:  print(f"[INFO] {msg}", flush=True)
def log_error(msg: str) -> None: print(f"[ERROR] {msg}", flush=True)


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--subset_size", required=True, type=int)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--annotation_key", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    manifest = yaml.safe_load(Path(args.manifest).read_text())
    filename = f"{args.dataset}_sub{args.subset_size}_{args.strategy}.h5ad"

    entry = manifest.get("files", {}).get(filename)
    if entry is None:
        log_error(f"{filename} not in manifest ({args.manifest})")
        sys.exit(1)

    url = f"{manifest['base_url']}/{filename}/content"
    dest = Path(args.output).resolve()
    log_info(f"Fetching {filename} from {url}")

    session = new_session()
    r = session.get(url, stream=True, allow_redirects=True)
    r.raise_for_status()
    written = stream_to_file(r, dest)
    if written == 0:
        dest.unlink(missing_ok=True)
        log_error(f"Downloaded 0 bytes from {url}")
        sys.exit(1)

    actual = md5sum(dest)
    expected = entry["md5"]
    if actual != expected:
        dest.unlink(missing_ok=True)
        log_error(f"MD5 mismatch for {filename}: got {actual}, expected {expected}")
        sys.exit(1)
    log_info(f"MD5 verified: {actual}")

    ad = anndata.read_h5ad(dest, backed="r")
    try:
        if args.annotation_key not in ad.obs.columns:
            log_error(f"Missing annotation_key '{args.annotation_key}' in {filename}. obs columns: {list(ad.obs.columns)}")
            sys.exit(1)
    finally:
        ad.file.close()

    log_info(f"OK: {dest.name} ready ({written} bytes)")


if __name__ == "__main__":
    main()
```

**Step 2: Make it executable**

Run:
```bash
chmod +x bin/fetch_zenodo_subset.py
```

**Step 3: Smoke-test standalone with the smallest file**

The smallest Zenodo file is `HFRA_sub1000_random.h5ad` (~10 MB). Run from the project root (`bin/` must be on `PYTHONPATH` for the `util` import to resolve — same convention Nextflow uses):

```bash
mkdir -p /tmp/zenodo-smoke && cd /tmp/zenodo-smoke
PYTHONPATH=/Users/<user>/Documents/Arbeit/Projects/Single_Cell_Foundation_Models/sc_language_foundation_models/bin \
  /Users/<user>/Documents/Arbeit/Projects/Single_Cell_Foundation_Models/sc_language_foundation_models/bin/fetch_zenodo_subset.py \
    --manifest /Users/<user>/Documents/Arbeit/Projects/Single_Cell_Foundation_Models/sc_language_foundation_models/datasets/zenodo_manifest.yml \
    --dataset HFRA \
    --subset_size 1000 \
    --strategy random \
    --annotation_key cell_type \
    --output HFRA_sub1000_random.h5ad
```

Expected last lines:
```
[INFO] MD5 verified: 93c81d9b2d5dd6bad2c17546d08020f1
[INFO] OK: HFRA_sub1000_random.h5ad ready (9957030 bytes)
```

(If `cell_type` is not the `annotation_key` in `datasets/HFRA.yml`, substitute the actual key from that YAML. Check with `grep annotation_key datasets/HFRA.yml`.)

**Step 4: Smoke-test the mismatch path**

Run the same command but pass `--subset_size 500` (not in manifest):

```bash
cd /tmp/zenodo-smoke
PYTHONPATH=/Users/<user>/Documents/Arbeit/Projects/Single_Cell_Foundation_Models/sc_language_foundation_models/bin \
  /Users/<user>/Documents/Arbeit/Projects/Single_Cell_Foundation_Models/sc_language_foundation_models/bin/fetch_zenodo_subset.py \
    --manifest /Users/<user>/Documents/Arbeit/Projects/Single_Cell_Foundation_Models/sc_language_foundation_models/datasets/zenodo_manifest.yml \
    --dataset HFRA --subset_size 500 --strategy random \
    --annotation_key cell_type --output would_not_exist.h5ad; echo "exit=$?"
```

Expected:
```
[ERROR] HFRA_sub500_random.h5ad not in manifest (...)
exit=1
```

**Step 5: Clean up smoke artifacts**

```bash
rm -rf /tmp/zenodo-smoke
```

**Step 6: Commit**

```bash
git add bin/fetch_zenodo_subset.py
git commit -m "feat(bin): add fetch_zenodo_subset.py for Zenodo-sourced subsets"
```

---

## Task 5: Add the `download_from_zenodo` Nextflow process

**Files:**
- Modify: `main.nf` (add a new process definition next to `download`)

**Step 1: Add the process**

In `main.nf`, immediately after the `download` process definition (ends at line 41) and before `process preprocess`, insert:

```groovy
process download_from_zenodo {
    container params.container_image
    conda 'envs/download.yml'
    storeDir "${params.cache_dir}/datasets/subset"

    input:
    tuple val(dataset), val(subset_size), val(strategy), val(annotation_key), path(manifest)

    tag { "${dataset}xsub${subset_size}x${strategy}" }

    output:
    tuple(path("${dataset}_sub${subset_size}_${strategy}.h5ad"), val(dataset), val(subset_size), val(strategy))

    script:
    """
    set -euo pipefail
    fetch_zenodo_subset.py \\
      --manifest "${manifest}" \\
      --dataset "${dataset}" \\
      --subset_size ${subset_size} \\
      --strategy "${strategy}" \\
      --annotation_key "${annotation_key}" \\
      --output "${dataset}_sub${subset_size}_${strategy}.h5ad"
    """
}
```

**Step 2: Verify the DSL parses**

Run:
```bash
nextflow inspect main.nf -profile local 2>&1 | tail -20
```

Expected: no Groovy/DSL errors mentioning `download_from_zenodo`. Nextflow may print process summaries or warnings; absence of a stack trace is the passing signal.

**Step 3: Commit**

```bash
git add main.nf
git commit -m "feat(nextflow): add download_from_zenodo process (unwired)"
```

---

## Task 6: Wire up upfront validation and workflow branch

**Files:**
- Modify: `main.nf` (the `workflow {}` block starting at line 229)

**Step 1: Replace the data-pipeline section**

Currently, lines 230–284 of `main.nf` contain:
```
// --- Data pipeline: download -> preprocess -> subset ---
...
subset_out = subset(subset_input)
```

Replace that whole block with the following (the model-preparation block from lines 249–263 must stay; keep it in its current position — we're only changing the data pipeline):

```groovy
    // --- Upfront Zenodo manifest validation ---

    if (params.download_from_zenodo) {
        def zenodo_manifest = loadYaml("datasets/zenodo_manifest.yml")
        def zenodo_available = zenodo_manifest.files.keySet()
        def zenodo_sizes = params.models.collect { resolveSubsetSize(it, params.subset_categories) }.unique()
        def missing = []
        params.datasets.each { ds ->
            zenodo_sizes.each { sz ->
                def fname = "${ds}_sub${sz}_${params.subset_strategy}.h5ad"
                if (!zenodo_available.contains(fname)) missing << fname
            }
        }
        if (missing) {
            log.error "download_from_zenodo=true but these files are not in datasets/zenodo_manifest.yml:"
            missing.each { log.error "  - ${it}" }
            log.error "Either disable download_from_zenodo, or adjust datasets / subset_categories / subset_strategy."
            System.exit(1)
        }
    }

    // --- Model preparation (must run before data pipeline so unique_sizes is available) ---
    //     [keep the existing block that sets up prepare_input, prepared_models, model_sizes, unique_sizes unchanged]

    // --- Data pipeline ---

    if (params.download_from_zenodo) {
        Channel
            .fromList(params.datasets)
            .combine(unique_sizes)
            .map { ds, sz ->
                def cfg = loadYaml("datasets/${ds}.yml")
                tuple(ds, sz, params.subset_strategy, cfg.annotation_key, file("datasets/zenodo_manifest.yml"))
            }
            .set { zenodo_input }

        subset_out = download_from_zenodo(zenodo_input)
    } else {
        Channel
            .fromList(params.datasets)
            .map { dataset -> tuple(dataset, file("datasets/${dataset}.yml")) }
            .set { download_input }

        download_out = download(download_input)

        download_out
            .map { raw_file, dataset ->
                def cfg = loadYaml("datasets/${dataset}.yml")
                def apply_mapping = cfg.containsKey('apply_ontology_mapping') ? cfg.apply_ontology_mapping : params.apply_ontology_mapping
                tuple(raw_file, dataset, file("datasets/${dataset}.yml"), file(params.cache_dir), cfg.annotation_key, apply_mapping)
            }
            .set { preprocess_input }

        preprocess_out = preprocess(preprocess_input)

        preprocess_out
            .combine(unique_sizes)
            .map { pp_file, dataset, annotation_key, subset_size ->
                tuple(pp_file, dataset, file("datasets/${dataset}.yml"), file(params.cache_dir), annotation_key, subset_size, params.min_cells_per_type, params.subset_strategy)
            }
            .set { subset_input }

        subset_out = subset(subset_input)
    }
```

**Important ordering constraint:** the data-pipeline `if/else` depends on `unique_sizes`, which is computed in the model-preparation section. Move the model-preparation block (currently lines 249–275, defining `prepare_input`, `prepared_models`, `model_sizes`, `unique_sizes` plus the OpenRouter API-key warning) so it sits **between** the validator and the data-pipeline branch. The code inside the block doesn't change — only its position.

Everything from `// --- Combine subsets with models ---` (originally line 286) onwards stays exactly as it is and continues to consume `subset_out`.

**Step 2: Inspect the DSL for parse errors**

Run:
```bash
nextflow inspect main.nf -profile local 2>&1 | tail -20
```

Expected: no Groovy errors. Warnings are fine.

**Step 3: Dry-run the upfront validator (negative path — should abort)**

Run with a size not in the manifest:
```bash
nextflow run main.nf -profile local \
  --download_from_zenodo true \
  --datasets '["HIHA"]' \
  --subset_categories '["default": 500]' \
  --subset_strategy stratified \
  --models '["celltypist_blind"]' 2>&1 | grep -A5 "download_from_zenodo=true"
```

Expected: output containing
```
ERROR ~ download_from_zenodo=true but these files are not in datasets/zenodo_manifest.yml:
ERROR ~   - HIHA_sub500_stratified.h5ad
ERROR ~ Either disable download_from_zenodo, or adjust datasets / subset_categories / subset_strategy.
```
Pipeline should exit non-zero without launching any processes.

**Step 4: Dry-run the upfront validator (positive path — should NOT abort)**

Run with a size that IS in the manifest. This may proceed to run model preparation but that's fine — the point is to confirm the validator passes.

```bash
nextflow run main.nf -profile local \
  --download_from_zenodo true \
  --datasets '["HIHA"]' \
  --subset_categories '["default": 1000]' \
  --subset_strategy stratified \
  --models '["celltypist_blind"]' 2>&1 | head -30
```

Expected: no "not in datasets/zenodo_manifest.yml" error. The pipeline should proceed past validation (and into `download_from_zenodo`, then `run_model`, etc.). Let this run to completion if you have time; otherwise `Ctrl-C` after you see `executor >  local (N)` confirming processes have launched.

**Step 5: Commit**

```bash
git add main.nf
git commit -m "feat(nextflow): branch workflow on download_from_zenodo with upfront manifest validation"
```

---

## Task 7: End-to-end smoke test

**No file changes** — this task is purely verification of the completed feature.

**Step 1: Clean slate (optional)**

If `cache/datasets/subset/HIHA_sub1000_stratified.h5ad` already exists from a prior Task 6 run, leave it; re-running should be a cache hit. To force a fresh download, delete it:

```bash
rm -f cache/datasets/subset/HIHA_sub1000_stratified.h5ad
```

**Step 2: Run the pipeline end-to-end on the Zenodo path**

```bash
rm -rf work
nextflow run main.nf -profile local \
  --download_from_zenodo true \
  --datasets '["HIHA"]' \
  --subset_categories '["default": 1000]' \
  --subset_strategy stratified \
  --models '["celltypist_blind"]'
```

Expected:
- No errors from `download_from_zenodo`.
- `cache/datasets/subset/HIHA_sub1000_stratified.h5ad` exists after the run.
- `run_model` and `evaluate_annotation` complete successfully.
- `results/` contains the normal outputs (JSON + aggregated CSV/PNG).

**Step 3: Cache-hit re-run**

Run the exact same command again. Expected: `download_from_zenodo` reports `cached` (no re-download), `run_model` also cache-hits if its prediction CSV exists.

**Step 4: Integrity regression test (one-off)**

Corrupt the cached file and confirm MD5 check fires on a fresh fetch:

```bash
# Flip a byte in the cached file, then force a re-fetch via the work dir
printf 'X' | dd of=cache/datasets/subset/HIHA_sub1000_stratified.h5ad bs=1 count=1 seek=100 conv=notrunc
rm -rf work
nextflow run main.nf -profile local \
  --download_from_zenodo true \
  --datasets '["HIHA"]' \
  --subset_categories '["default": 1000]' \
  --subset_strategy stratified \
  --models '["celltypist_blind"]' 2>&1 | tail -20
```

**Important:** The corruption check above is a quick byte-flip. Because `storeDir` returns the existing cached file without re-running the process, you may need to delete the cached file to force a real re-download before this test can actually exercise the MD5 path. If you want to test a true MD5 mismatch, instead temporarily edit `datasets/zenodo_manifest.yml` to a wrong MD5 for one file, delete that cached file, `rm -rf work`, and re-run — expected: `[ERROR] MD5 mismatch ...` and non-zero exit. **Revert the manifest afterward.**

**Step 5: (Optional) Restore the cache and move on**

```bash
rm -f cache/datasets/subset/HIHA_sub1000_stratified.h5ad  # will re-download on next run
```

No commit for this task — verification only.

---

## Task 8: Document in README

**Files:**
- Modify: `README.md` (the parameter table around lines 9–17, and optionally a new subsection)

**Step 1: Add the flag to the parameter table**

In `README.md`, in the parameter table (currently lines 9–17), add a row for `download_from_zenodo`:

```markdown
| `download_from_zenodo` | If `true`, skip `download`→`preprocess`→`subset` and fetch preprocessed+subsetted files from the Zenodo record listed in `datasets/zenodo_manifest.yml`. All requested `(dataset, subset_size, subset_strategy)` combinations must be in the manifest or the pipeline aborts. |
```

**Step 2: Add a new subsection after `#### Cache invalidation`**

Append:

```markdown
#### Fetching preprocessed data from Zenodo

To skip the full preprocessing and subsetting chain, set `download_from_zenodo=true`
(via `nextflow.config` or `--download_from_zenodo true`). The pipeline will then
fetch files directly from the public Zenodo record pinned in
`datasets/zenodo_manifest.yml` and place them under `cache/datasets/subset/`.

Currently mirrored on Zenodo (16 files): datasets `HFRA`, `HIHA`, `human_pancreas`,
`Zeisel`; subset sizes `1000` and `10000`; strategies `stratified` and `random`.
Requested combinations outside this set cause the pipeline to abort upfront with
a list of the missing files. MD5 checksums are verified after each download.

Zenodo files are ontology-mapped (produced with `apply_ontology_mapping=true`);
the value of `params.apply_ontology_mapping` is ignored for the fetch step but
still forwarded to the evaluation step.
```

**Step 3: Verify it renders**

Run:
```bash
grep -A2 "download_from_zenodo" README.md | head -20
```

Expected: the new row and the first couple lines of the subsection both show up.

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): document download_from_zenodo flag"
```

---

## Rollback notes

Every task is a single commit; reverting `git revert <sha>` undoes that task cleanly. The Zenodo path is fully opt-in (`params.download_from_zenodo = false` by default), so merging without enabling the flag is a no-op for existing users.
