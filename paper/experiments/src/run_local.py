#!/usr/bin/env python
"""Answer the prompt files of the local arms with Hugging Face checkpoints, on a GPU if present.

Reads ``<study tree>/prompts/`` like ``run_openrouter.py``; needs torch, transformers and the
checkpoints (``env-local.yml``), but neither cell2sentence nor scanpy. Arms with ``decoding:
greedy`` decode greedily, the others sample with the file's seed. Answers go to the response cache
as each batch finishes, so a stopped job resumes; greedy answers are cached without the seed
(``<arm>__<dataset>__greedy.jsonl``). Writes ``predictions.csv`` and ``parse_stats.json`` as
``run_openrouter.py`` does.

    python src/run_local.py --arms A3_scllm_percell [--datasets ...] [--study lists]
    python src/run_local.py --arms A3_scllm_percell --max-files 2 --limit 64    # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner_cache import ResponseCache, write_atomic  # noqa: E402
from run_openrouter import extract_prediction, parse_prompt_file  # noqa: E402


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def _base_model_from_card(repo):
    """The ``base_model`` of a Hugging Face model card's front matter (its first entry if it is a
    list), or None. The front matter is read line by line, not parsed as YAML."""
    try:
        from huggingface_hub import hf_hub_download
        card = Path(hf_hub_download(repo, "README.md", repo_type="model")).read_text()
    except Exception as exc:  # noqa: BLE001
        print(f"  [verify] could not read {repo} model card ({exc})", file=sys.stderr)
        return None
    lines = card.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":       # end of the front matter
            break
        stripped = line.strip()
        if stripped.startswith("base_model:"):
            value = stripped.split(":", 1)[1].strip().strip("[]'\"")
            if value:                   # `base_model: google/gemma-2-2b`
                return value.split(",")[0].strip()
            # or a list on the next lines: `- google/gemma-2-2b`
            for nxt in lines[i + 1:i + 4]:
                item = nxt.strip().lstrip("-").strip().strip("'\"")
                if item:
                    return item
            return None
    return None


def verify_base_model(arm, cfg):
    """Stop if the model named by ``verify_base_of`` has another base model than ``arm["model"]``.

    The base model comes from the adapted model's card (``base_model``), else from its config's
    ``_name_or_path`` unless that names the adapted model itself. Without either, a warning is
    printed and the run goes on.
    """
    target = arm.get("verify_base_of")
    if not target:
        return

    parent = _base_model_from_card(target)
    if parent:
        print(f"  [verify] {target} model card declares base_model: {parent}")
    else:
        from transformers import AutoConfig
        try:
            adapted = AutoConfig.from_pretrained(target)
        except Exception as exc:  # network, auth, gated repo
            print(f"  [verify] could not read {target} ({exc}); base-model check skipped. "
                  f"Check the base model by hand.", file=sys.stderr)
            return
        candidate = getattr(adapted, "_name_or_path", None)
        # A self-reference names the repository the config was loaded from, not a base model.
        if candidate and Path(str(candidate)).name.lower() != Path(target).name.lower():
            parent = candidate
            print(f"  [verify] no base_model in the model card; falling back to "
                  f"config._name_or_path: {parent}", file=sys.stderr)
    if not parent:
        print(f"  [verify] {target} names no base model in its card or config; cannot confirm "
              f"that {arm['model']} is its base. Check by hand.",
              file=sys.stderr)
        return

    if Path(str(parent)).name.lower() != Path(arm["model"]).name.lower():
        raise SystemExit(
            f"BASE-MODEL MISMATCH: {target} reports its base as '{parent}', but the control arm is "
            f"'{arm['model']}'. Running this pair would measure instruction tuning as much as "
            f"single-cell adaptation. Update `arms.A2_base_percell.model` in config.yml to "
            f"'{parent}', or drop the control claim.")
    print(f"  [verify] OK: {target} reports base '{parent}', matching the control arm")


def load_model(model_id, dtype="bfloat16", device=None):
    """Load model and tokenizer, printing the elapsed time before each step.

    Returns ``(model, tokenizer, device)``; the device is CUDA, else MPS, else the CPU.
    """
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()

    def step(msg):
        print(f"  [load +{time.time() - t0:6.1f}s] {msg}", flush=True)

    if device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    step(f"device={device}; loading tokenizer for {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        # Batches are padded on the left, with the EOS token if there is no pad token.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    step("tokenizer ready; calling from_pretrained (weights -> GPU)")
    # low_cpu_mem_usage avoids a full copy of the checkpoint in host memory; on CUDA, device_map
    # loads the shards straight to the GPU instead of moving the model there afterwards.
    kwargs = {"torch_dtype": getattr(torch, dtype), "low_cpu_mem_usage": True}
    if device == "cuda":
        kwargs["device_map"] = {"": 0}      # all weights on GPU 0
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    step("from_pretrained returned")
    if device != "cuda":
        model = model.to(device)
        step(f"moved to {device}")
    model = model.eval()
    step("eval() done")

    if device == "cuda":
        step(f"weights occupy {torch.cuda.memory_allocated() / 2**30:.1f} GB of "
             f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")
    step(f"MODEL READY: {model_id} on {device} ({dtype})")
    return model, tokenizer, device


def generate(model, tokenizer, device, prompts, seed, greedy=False, max_new_tokens=20,
             batch_size=16, report_every=5):
    """Yield ``(prompt, answer)`` batch by batch: greedy decoding or sampling seeded with ``seed``,
    at most ``max_new_tokens`` new tokens (20, as in the benchmark's C2S-Scale wrapper).

    Prints the progress with rate, remaining time and, on CUDA, peak GPU memory.
    """
    import time

    import torch
    torch.manual_seed(seed)
    done = 0
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    t0 = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for bi, start in enumerate(range(0, len(prompts), batch_size), start=1):
        batch = prompts[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                        max_length=4096).to(device)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=not greedy,
                                 pad_token_id=tokenizer.pad_token_id)
        for prompt, row, prompt_ids in zip(batch, ids, enc["input_ids"]):
            # Only the new tokens, without the prompt.
            yield prompt, tokenizer.decode(row[len(prompt_ids):], skip_special_tokens=True)
        done += len(batch)

        if bi == 1 or bi % report_every == 0 or bi == n_batches:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-9)
            eta = (len(prompts) - done) / max(rate, 1e-9)
            bar_n = int(24 * done / len(prompts))
            mem = ""
            if device == "cuda":
                mem = (f" | VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f}/"
                       f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")
            print(f"    [{'#' * bar_n}{'.' * (24 - bar_n)}] {done}/{len(prompts)} "
                  f"({100 * done / len(prompts):5.1f}%) | {rate:6.1f} prompt/s | "
                  f"tok {enc['input_ids'].shape[1]:4d} | ETA {eta / 60:5.1f} min{mem}",
                  flush=True)

    total = time.time() - t0
    print(f"    done: {len(prompts)} prompts in {total / 60:.1f} min "
          f"({len(prompts) / max(total, 1e-9):.1f}/s)", flush=True)


def run_file(prompt_file, cfg, out_root, cache_root, model_bundle, dry_run=False, limit=None,
             batch_size=None):
    """Answer one prompt file and write its predictions; returns their path, None on a dry run."""
    dataset, arm_name, mode, variant, seed = parse_prompt_file(prompt_file)
    arm = cfg["arms"][arm_name]
    frame = pd.read_csv(prompt_file)
    if limit:
        frame = frame.head(limit)

    # A greedy answer does not depend on the seed, so greedy answers have a cache file of their
    # own, keyed without the seed.
    decoding = arm.get("decoding", "sample")
    greedy = decoding == "greedy"
    suffix = "__greedy" if greedy else ""
    cache = ResponseCache(Path(cache_root) / f"{arm_name}__{dataset}{suffix}.jsonl", arm["model"],
                          seed=None if greedy else seed)
    hits, misses = cache.stats(frame["prompt"].tolist())
    print(f"[{arm_name} {dataset} {mode} {variant} seed{seed}] {decoding}: "
          f"{len(frame)} prompts: {hits} cached, {misses} to generate")
    if dry_run:
        return None

    # Each distinct prompt once.
    todo = list(dict.fromkeys(p for p in frame["prompt"] if cache.get(p) is None))
    if todo:
        # model_bundle loads the model on first use, so a fully cached file needs no checkpoint.
        model, tokenizer, device = model_bundle()
        kw = {"batch_size": batch_size} if batch_size else {}
        print(f"    starting generation: {len(todo)} prompts, batch {batch_size or 16}", flush=True)
        # Cached as each batch arrives, so an interrupted job resumes after its last batch.
        for prompt, text in generate(model, tokenizer, device, todo, seed, greedy=greedy, **kw):
            cache.put(prompt, text)

    frame["predicted_labels"] = [extract_prediction(cache.get(p)) for p in frame["prompt"]]

    # Answer-format counts for parse_stats.json: empty answers and answers of more than 8 words.
    unparseable = int((frame["predicted_labels"] == "").sum())
    long_answers = int((frame["predicted_labels"].str.split().str.len() > 8).sum())

    out = Path(out_root) / arm_name / dataset / mode / variant / f"seed{seed}" / "predictions.csv"
    write_atomic(out, frame[["cell_id", "true_labels", "predicted_labels"]].to_csv(index=False))
    stats = {"arm": arm_name, "dataset": dataset, "mode": mode, "variant": variant, "seed": seed,
             "n": len(frame), "empty": unparseable, "over_8_words": long_answers}
    write_atomic(out.parent / "parse_stats.json", json.dumps(stats, indent=2))
    print(f"    -> {out}  ({unparseable} empty, {long_answers} answers over 8 words)")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--config", default=str(root / "config.yml"))
    p.add_argument("--host", default="cluster")
    p.add_argument("--arms", nargs="+")
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--modes", nargs="+", help="only these context modes (conditions)")
    p.add_argument("--study", default="ablation", help="the study tree (see `studies` in config)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int,
                   help="only the first N prompts per file (smoke test). Results are diverted to "
                        "results/smoke/ and the cache is kept separate, so a partial run can never "
                        "be mistaken for a real one")
    p.add_argument("--max-files", type=int, help="stop after N prompt files (smoke test)")
    p.add_argument("--batch-size", type=int,
                   help="generation batch size; the default of 16 is very conservative for an H100")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    arms = args.arms or cfg["execution"]["hosts"][args.host]
    tree = root / cfg["studies"][args.study]
    prompts_dir = tree / "prompts"
    if not prompts_dir.exists():
        raise SystemExit(f"no prompts at {prompts_dir}; copy results/prompts/ from the host that "
                         f"ran sentences.py --materialize")

    # A smoke test keeps its predictions and cache apart from those of a full run.
    if args.limit or args.max_files:
        out_root = tree / "smoke" / "predictions"
        cache_root = tree / "smoke" / "cache"
        print(f"smoke test: limit={args.limit} max_files={args.max_files}\n"
              f"  writing to {out_root}; delete this folder before the full run", file=sys.stderr)
    else:
        out_root = tree / "predictions"
        cache_root = tree / "cache"

    def selected(prompt_file, arm_name):
        dataset, arm, mode, _variant, _seed = parse_prompt_file(prompt_file)
        return (arm == arm_name and (not args.datasets or dataset in args.datasets)
                and (not args.modes or mode in args.modes))

    for arm_name in arms:
        arm = cfg["arms"][arm_name]
        if arm["kind"] != "local":
            print(f"[skip] {arm_name}: not a local arm")
            continue
        files = sorted(f for f in prompts_dir.glob("*.csv") if selected(f, arm_name))
        if not files:
            print(f"[skip] {arm_name}: no prompt files")
            continue
        if args.max_files:
            files = files[:args.max_files]

        bundle = None
        if not args.dry_run:
            verify_base_model(arm, cfg)
            cached_bundle = []

            def bundle(_arm=arm):
                if not cached_bundle:
                    cached_bundle.append(load_model(_arm["model"]))
                return cached_bundle[0]
        # Progress over the files of this arm.
        t_arm = time.time()
        for i, f in enumerate(files, start=1):
            el = time.time() - t_arm
            eta = (el / (i - 1) * (len(files) - i + 1)) if i > 1 else float("nan")
            print(f"\n[file {i}/{len(files)}] {f.name}  "
                  f"(elapsed {el / 60:.1f} min, shard ETA {eta / 60:.1f} min)", flush=True)
            run_file(f, cfg, out_root, cache_root, bundle, dry_run=args.dry_run,
                     limit=args.limit, batch_size=args.batch_size)
        print(f"\n[{arm_name}] {len(files)} files in {(time.time() - t_arm) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
