#!/usr/bin/env python3
"""Keep a version-tagged history of the aggregated results CSV under ``<cache_dir>/result/``.

Both ``bin/plot_model_comparison.py`` and the Nextflow ``combine_results`` process call this, so
every time the aggregate is (re)generated a dated, versioned copy is retained. The live
``results/aggregated_by_dataset_model.csv`` is always overwritten in place; these snapshots are
not, so they form an audit trail of what the figures were built from.

Filename: ``<stem>_<YYYYMMDD>_v<NN>.csv`` (e.g. ``aggregated_by_dataset_model_20260729_v01.csv``),
where ``NN`` is the next unused two-digit version for the current day. If the newest snapshot for
today is already byte-identical to the source, nothing is written -- so repeated plotting runs on
one day do not pile up duplicate files.

Stdlib only (no pandas), so it runs unchanged inside the pipeline's conda/container env.
"""
from __future__ import annotations

import argparse
import datetime
import filecmp
import re
import shutil
import sys
from pathlib import Path


def snapshot_aggregate(input_csv, cache_dir="cache", today=None):
    """Copy ``input_csv`` into ``<cache_dir>/result/`` under a dated, versioned name.

    Returns ``(dest_path, wrote)``: ``wrote`` is False when an identical snapshot already exists
    for today (the existing path is returned instead of making a duplicate).
    """
    input_csv = Path(input_csv)
    if not input_csv.is_file():
        raise FileNotFoundError(f"aggregate CSV not found: {input_csv}")

    out_dir = Path(cache_dir) / "result"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = input_csv.stem
    day = (today or datetime.date.today()).strftime("%Y%m%d")
    pat = re.compile(rf"^{re.escape(stem)}_{day}_v(\d+)\.csv$")

    versions = []
    for f in out_dir.iterdir():
        match = pat.match(f.name)
        if match:
            versions.append(int(match.group(1)))
    versions.sort()

    # Don't re-snapshot an unchanged aggregate: if today's latest copy already matches, reuse it.
    if versions:
        latest = out_dir / f"{stem}_{day}_v{versions[-1]:02d}.csv"
        if latest.is_file() and filecmp.cmp(input_csv, latest, shallow=False):
            return latest, False

    next_ver = (versions[-1] + 1) if versions else 1
    dest = out_dir / f"{stem}_{day}_v{next_ver:02d}.csv"
    shutil.copy2(input_csv, dest)
    return dest, True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Aggregated results CSV to snapshot.")
    p.add_argument("--cache-dir", default="cache",
                   help="Cache root; the snapshot goes in <cache-dir>/result/ (default: cache).")
    args = p.parse_args(argv)
    try:
        dest, wrote = snapshot_aggregate(args.input, args.cache_dir)
    except FileNotFoundError as exc:
        print(f"snapshot_aggregate: {exc}", file=sys.stderr)
        return 1
    print(f"{'wrote' if wrote else 'up-to-date'} aggregate snapshot: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
