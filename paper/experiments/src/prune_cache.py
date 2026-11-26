#!/usr/bin/env python
"""Drop the entries of the ablation's response cache that no current prompt file asks for.

A cache entry stores the key, not the prompt, so the wanted keys are rebuilt from
``results/prompts/`` and every other entry of ``results/cache/*.jsonl`` is dropped. Without
``--apply`` it only reports; with it, it refuses to write while a runner is running.

    python src/prune_cache.py [--apply]
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
    tree = root / cfg["studies"]["ablation"]
    prompts_dir, cache_root = tree / "prompts", tree / "cache"
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
                dropped += 1          # an unreadable line
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
