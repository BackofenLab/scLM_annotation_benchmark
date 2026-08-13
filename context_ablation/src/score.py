#!/usr/bin/env python
"""Score the study's predictions with the benchmark's own two stages.

Deliberately a thin wrapper rather than a reimplementation. ``bin/map_predictions.py`` and
``bin/evaluate.py`` carry the ontology canonicalisation, the review filter and the coverage gate
that produced every number in the main table; scoring this study any other way would put its
results on a different scale and quietly break the comparison the whole design rests on.

Run on the machine that has the core env and ``cache/cell_type_ontology_mappings.csv`` -- the
scoring stage is cheap, so both hosts' predictions are collected and scored in one place.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def prediction_files(cfg, root):
    """Every predictions.csv the runners wrote, with its coordinates recovered from the path."""
    pred_root = root / cfg["output"]["predictions_dir"]
    for path in sorted(pred_root.glob("*/*/*/*/*/predictions.csv")):
        arm, dataset, mode, variant, seed = path.parts[-6:-1]
        yield path, {"arm": arm, "dataset": dataset, "mode": mode,
                     "variant": variant, "seed": seed.removeprefix("seed")}


def score_one(path, coords, cfg, root, python_exe):
    """map_predictions -> evaluate, into a per-run directory. Skips work already done.

    Both stages are cached: the mapped CSV is reused when present, and the JSON is not recomputed.
    Scoring is cheap next to generation, but re-mapping appends to the mappings CSV, so skipping is
    about avoiding side effects as much as about time. That CSV is this study's own copy, never the
    benchmark's -- see `scoring.cache_dir` in config.yml for why that distinction was not optional.
    """
    # Resolved from config, and it must stay LOCAL to this study: map_predictions.py appends every
    # newly seen label to cell_type_ontology_mappings.csv inside this directory. Pointing it at the
    # repo cache means this study mutates the file the benchmark's published numbers rest on.
    cache_dir = (root / cfg["scoring"]["cache_dir"]).resolve()
    if not cache_dir.is_dir():
        raise SystemExit(f"scoring.cache_dir does not exist: {cache_dir}. Seed it with the "
                         f"benchmark's cell_type_ontology_mappings.csv and cl-base.owl first.")

    tag = "__".join([coords["arm"], coords["dataset"], coords["mode"],
                     coords["variant"], f"seed{coords['seed']}"])
    work = root / "results" / "scored" / tag
    work.mkdir(parents=True, exist_ok=True)
    result_json = work / f"{coords['dataset']}_{coords['arm']}.json"
    if result_json.exists():
        return result_json, "cached"

    mapped = work / "predictions_mapped.csv"
    if not mapped.exists():
        subprocess.run(
            [python_exe, str(REPO_ROOT / cfg["scoring"]["map_predictions"].lstrip("./")),
             "--pred", str(path), "--model", coords["arm"], "--dataset", coords["dataset"],
             "--out-dir", str(work), "--cache_dir", str(cache_dir),
             "--apply_ontology_mapping", "true"],
            check=True)
        produced = sorted(work.glob("*_mapped.csv"))
        if not produced:
            raise SystemExit(f"{tag}: map_predictions produced no mapped CSV")
        mapped = produced[0]

    subprocess.run(
        [python_exe, str(REPO_ROOT / cfg["scoring"]["evaluate"].lstrip("./")),
         "--pred", str(mapped), "--model", coords["arm"], "--out", str(result_json),
         "--cache_dir", str(cache_dir), "--max-unmapped-ratio", "0.99",
         "--min-scoring-coverage", str(cfg["scoring"]["min_scoring_coverage"]),
         "--use_text_encoder", "false"],
        check=True)
    return result_json, "scored"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--config", default=str(root / "config.yml"))
    p.add_argument("--python", default=sys.executable,
                   help="interpreter for bin/*.py (needs the evaluation deps, e.g. pronto)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    found = list(prediction_files(cfg, root))
    if not found:
        raise SystemExit(f"no predictions under {root / cfg['output']['predictions_dir']}; "
                         f"run the runners first")

    print(f"{len(found)} prediction file(s) to score")
    if args.dry_run:
        for path, coords in found:
            print(f"  {coords['arm']:22} {coords['dataset']:16} {coords['mode']:3} "
                  f"{coords['variant']:17} seed{coords['seed']}")
        return 0

    counts = {"scored": 0, "cached": 0}
    for path, coords in found:
        _out, status = score_one(path, coords, cfg, root, args.python)
        counts[status] += 1
    print(f"done: {counts['scored']} scored, {counts['cached']} already present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
