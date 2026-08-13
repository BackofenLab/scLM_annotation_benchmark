#!/usr/bin/env python
"""Drop cache entries that no current prompt asks for.

The response cache is keyed on ``sha256(model | prompt | seed)``, so changing a prompt does not
corrupt anything -- the new prompt simply misses and the old entry is never looked up again. That
property is what makes editing a template safe mid-study.

It does leave orphans behind, though, and orphans are worth removing rather than tolerating. After
the numpy-repr fix regenerated every M2/M4/M5 prompt for the instruction arms, roughly half of A1's
cached responses became unreachable answers to a question the study no longer asks. Left in place
they cost disk, they inflate "how many responses do we have" by a factor that does not correspond to
any result, and -- worst -- an auditor comparing cache size against prompt count would find a
mismatch with no way to tell whether it meant orphans or missing work.

Orphans are identified by reconstructing the set of keys the CURRENT prompt files imply, then
keeping only those. Nothing is inferred from the record itself: a cache entry stores the key, not
the prompt, so the prompt files are the only source of truth for what is still wanted.

Refuses to run while a runner is live, since both append to the same file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner_cache import prompt_key, write_atomic  # noqa: E402
from run_openrouter import parse_prompt_file  # noqa: E402


def wanted_keys(prompts_dir, arm_name, dataset, model):
    """Every key the current prompt files for this (arm, dataset) would look up."""
    keys = set()
    for f in sorted(Path(prompts_dir).glob(f"{dataset}__{arm_name}__*.csv")):
        _ds, _arm, _mode, _variant, seed = parse_prompt_file(f)
        for prompt in pd.read_csv(f)["prompt"]:
            keys.add(prompt_key(model, prompt, seed))
    return keys


def prune(root, cfg, dry_run=True):
    prompts_dir = root / cfg["execution"]["prompts_dir"]
    cache_root = root / "results" / "cache"
    total_kept = total_dropped = 0

    for cache_file in sorted(cache_root.glob("*.jsonl")):
        arm_name, dataset = cache_file.stem.split("__", 1)
        arm = cfg["arms"].get(arm_name)
        if arm is None:
            print(f"  [skip] {cache_file.name}: no such arm in config")
            continue

        keys = wanted_keys(prompts_dir, arm_name, dataset, arm["model"])
        if not keys:
            print(f"  [skip] {cache_file.name}: no prompt files found, refusing to prune blind")
            continue

        kept, dropped = [], 0
        for line in cache_file.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1          # a torn last line is an orphan by definition
                continue
            if rec.get("key") in keys:
                kept.append(line)
            else:
                dropped += 1

        total_kept += len(kept); total_dropped += dropped
        pct = 100 * dropped / max(len(kept) + dropped, 1)
        print(f"  {cache_file.name:44s} keep {len(kept):6d}  drop {dropped:6d} ({pct:4.1f}%)")
        if not dry_run and dropped:
            write_atomic(cache_file, "".join(l + "\n" for l in kept))

    verb = "would drop" if dry_run else "dropped"
    print(f"\n  {verb} {total_dropped} orphan(s); {total_kept} live entries remain")
    return total_dropped


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(root / "config.yml"))
    p.add_argument("--apply", action="store_true", help="write the pruned files (default: dry run)")
    args = p.parse_args(argv)

    import subprocess
    live = subprocess.run(["pgrep", "-f", "run_openrouter|run_local"],
                          capture_output=True, text=True).stdout.strip()
    if live and args.apply:
        raise SystemExit("a runner is still live; stop it before pruning (both append to the cache)")

    cfg = yaml.safe_load(open(args.config))
    prune(root, cfg, dry_run=not args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
