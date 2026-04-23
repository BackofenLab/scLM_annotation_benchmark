# Design: `download_from_zenodo` option

**Date:** 2026-04-23
**Status:** Approved — ready for implementation plan.

## Goal

Add a pipeline mode that skips the `download → preprocess → subset` chain and
fetches pre-processed, pre-subsetted AnnData files directly from the public
Zenodo record [19707183](https://zenodo.org/records/19707183). Controlled by
a new boolean flag `params.download_from_zenodo` in `nextflow.config`.

The Zenodo record mirrors 16 files covering the cross product of:

- datasets: `HFRA`, `HIHA`, `human_pancreas`, `Zeisel`
- subset sizes: `1000`, `10000`
- subset strategies: `stratified`, `random`

Filenames match the pipeline's existing `{dataset}_sub{size}_{strategy}.h5ad`
convention exactly, and all files have already been ontology-mapped (i.e.
produced by running the existing `preprocess` step with
`apply_ontology_mapping=true`).

## Config surface

One new parameter in `nextflow.config`:

```groovy
params.download_from_zenodo = false
```

When `true`, `params.datasets`, `params.subset_categories` and
`params.subset_strategy` select **which** Zenodo files to fetch.
`params.apply_ontology_mapping` is effectively ignored for the fetch step
(Zenodo data is pre-mapped) but is still forwarded to `evaluate_annotation`
unchanged.

README is updated to document the flag and list the 16 mirrored files.

## Manifest

New file `datasets/zenodo_manifest.yml` pins the record and enumerates every
available file with its expected MD5 and size:

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

Download URL per file = `{base_url}/{filename}/content`. Adding a future
Zenodo file = append a row.

## New Nextflow process

Added to `main.nf` next to `download`:

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

Key properties:

- **`storeDir` matches `subset`**, so files land in `cache/datasets/subset/`
  and downstream processes see them identically to subset-produced files.
  Re-runs are cache hits.
- **Output tuple shape matches `subset_out`**, so the `workflow {}` block
  can splice this in with no downstream changes.
- **Reuses `envs/download.yml`** (same deps as `download.py`).

## New Python script `bin/fetch_zenodo_subset.py`

Small fetch-and-verify script (~100 lines), mirroring the structure of
`bin/download.py`. Responsibilities:

1. Load manifest (`datasets/zenodo_manifest.yml`).
2. Resolve the expected filename, look it up in the manifest. If missing,
   exit non-zero with a clear error. (Defense in depth — the workflow-entry
   validator should have caught it earlier.)
3. Stream the file from `{base_url}/{filename}/content` to the work-dir
   output path.
4. Verify the MD5 against the manifest. On mismatch: delete the file and
   exit non-zero. No in-script retry — Nextflow's `storeDir` + `-resume`
   handles retries cleanly because the file is removed before exit.
5. Open the AnnData in backed mode and confirm the expected `annotation_key`
   column is present (mirrors `download.py`'s `verify_columns`).

The `_stream_to_file` and `_new_session` helpers currently living in
`download.py` are lifted into a shared `bin/util/http.py` module so both
scripts use the same HTTP path. No other refactor to `download.py`.

## Workflow branching

Two changes in the `workflow {}` block of `main.nf`.

**1. Upfront validator** — runs at pipeline entry, before any process kicks
off:

```groovy
if (params.download_from_zenodo) {
    def manifest = loadYaml("datasets/zenodo_manifest.yml")
    def available = manifest.files.keySet()

    def sizes = params.models.collect {
        resolveSubsetSize(it, params.subset_categories)
    }.unique()

    def missing = []
    params.datasets.each { ds ->
        sizes.each { sz ->
            def fname = "${ds}_sub${sz}_${params.subset_strategy}.h5ad"
            if (!available.contains(fname)) missing << fname
        }
    }
    if (missing) {
        log.error "download_from_zenodo=true but these files are not in datasets/zenodo_manifest.yml:"
        missing.each { log.error "  - ${it}" }
        log.error "Either disable download_from_zenodo, or adjust datasets / subset_categories / subset_strategy."
        System.exit(1)
    }
}
```

Fast-fail: the user sees every missing combination in one shot.

**2. Branch**:

```groovy
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
    // existing download → preprocess → subset chain, produces subset_out
}
```

Everything downstream of `subset_out` (model combine, `run_model`,
`evaluate_annotation`, `combine_results`) is unchanged. The validator and
the branch share `unique_sizes` as their single source of truth, so
passing validation guarantees the fetch will find every file.

## Error handling

Three levels, each with a distinct message:

1. **Workflow-entry validator** — missing combinations listed, pipeline
   aborts before any process runs.
2. **Manifest lookup in `fetch_zenodo_subset.py`** — redundant check,
   only fires if the validator is bypassed or the manifest changes
   mid-run. Exits with `[ERROR] {filename} not in manifest`.
3. **Download / MD5 failures** — file deleted, non-zero exit. Nextflow's
   default error strategy aborts the run; `-resume` retries cleanly.

No silent fallback from the Zenodo path to the legacy pipeline — mixing
the two would hide config errors.

## Testing

- **Manual smoke test** (primary):
  `nextflow run main.nf -profile local --download_from_zenodo true`
  with `params.datasets = ['HIHA']` and
  `params.subset_categories = ['default': 1000]`. Expect: one Zenodo
  download, cache populated, `run_model` and `evaluate_annotation`
  succeed on the fetched file.
- **Mismatch test**: set `subset_strategy='stratified'` + a size not in
  the manifest (e.g. 500) and confirm the upfront validator aborts with
  the expected message and no processes start.
- **Re-run test**: second run with the same config should be cache hits
  only, no re-downloads.
- **Integrity test**: corrupt one cached file under
  `cache/datasets/subset/` (flip a byte), delete the Nextflow work dir,
  re-run, confirm the MD5 check catches it and the run aborts. (The
  existing pipeline has no equivalent check, so this is new ground.)

No automated unit tests — the project currently has none, and the logic
is mostly Nextflow wiring plus a thin HTTP fetch.

## Non-goals

- Dynamic discovery of Zenodo files via the Zenodo API — manifest is
  hardcoded, explicit, and version-controlled.
- Token/credential handling — the record is public, no auth is required.
- Mixed-mode runs that fall back from Zenodo to the legacy pipeline per
  missing file — either all combinations are on Zenodo or the run aborts.
- Extending `download.py` to cover both paths — a dedicated
  `fetch_zenodo_subset.py` keeps responsibilities clean.
