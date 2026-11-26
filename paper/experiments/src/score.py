#!/usr/bin/env python
"""Map and score the predictions of a study with the benchmark's scripts
(``bin/map_predictions.py``, ``bin/evaluate.py``), one folder per run under ``<study>/scored``.

Runs that are already scored are skipped. The mapping table is the study's own
(``scoring.cache_dir`` in config.yml), so answers that are new in a study never enter the
benchmark's table.

    python src/score.py --study ablation
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
MAPPINGS = "cell_type_ontology_mappings.csv"


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def prediction_files(pred_root):
    """Every predictions.csv of the runners, with the coordinates of its run."""
    for path in sorted(pred_root.glob("*/*/*/*/*/predictions.csv")):
        arm, dataset, mode, variant, seed = path.parts[-6:-1]
        yield path, {"arm": arm, "dataset": dataset, "mode": mode,
                     "variant": variant, "seed": seed.removeprefix("seed")}


def result_file(work, coords, cfg):
    return work / f"{coords['dataset']}__{coords['arm']}__sub{cfg['eval']['size']}.json"


def score_one(path, coords, cfg, root, python_exe, tree):
    """Map and score one run into its own folder unless that is done already."""
    cache_dir = (root / cfg["scoring"]["cache_dir"]).resolve()
    if not (cache_dir / MAPPINGS).exists():
        raise SystemExit(f"{cache_dir / MAPPINGS} not found. Copy the benchmark's mapping table "
                         "and Cell Ontology release into scoring.cache_dir first.")
    tag = "__".join([coords["arm"], coords["dataset"], coords["mode"],
                     coords["variant"], f"seed{coords['seed']}"])
    work = tree / "scored" / tag
    work.mkdir(parents=True, exist_ok=True)
    if result_file(work, coords, cfg).exists():
        return "cached"
    common = ["--method", coords["arm"], "--dataset", coords["dataset"], "--out-dir", str(work),
              "--mappings", str(cache_dir / MAPPINGS), "--cache_dir", str(cache_dir)]
    mapped = work / "predictions_mapped.csv"
    if not mapped.exists():
        subprocess.run([python_exe, str(REPO_ROOT / cfg["scoring"]["map_predictions"].lstrip("./")),
                        "--pred", str(path), *common], check=True)
    subprocess.run([python_exe, str(REPO_ROOT / cfg["scoring"]["evaluate"].lstrip("./")),
                    "--pred", str(mapped), *common, "--subset-size", str(cfg["eval"]["size"]),
                    "--min-scoring-coverage", str(cfg["scoring"]["min_scoring_coverage"]),
                    "--use_text_encoder", "false"], check=True)
    return "scored"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--config", default=str(root / "config.yml"))
    p.add_argument("--python", default=sys.executable,
                   help="interpreter for the scripts in bin/ (needs the evaluation environment)")
    p.add_argument("--study", default="ablation", help="the study tree (see studies in config.yml)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    tree = root / cfg["studies"][args.study]
    found = list(prediction_files(tree / "predictions"))
    if not found:
        raise SystemExit(f"no predictions under {tree / 'predictions'}; run the runners first")
    print(f"{len(found)} prediction file(s) to score")
    if args.dry_run:
        for _path, coords in found:
            print(f"  {coords['arm']:22} {coords['dataset']:16} {coords['mode']:3} "
                  f"{coords['variant']:17} seed{coords['seed']}")
        return 0
    counts = {"scored": 0, "cached": 0}
    for path, coords in found:
        counts[score_one(path, coords, cfg, root, args.python, tree)] += 1
    print(f"done: {counts['scored']} scored, {counts['cached']} already present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
