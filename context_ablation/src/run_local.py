#!/usr/bin/env python
"""Run the local-weight arms (A2 base Gemma, A3 C2S-Scale) over materialised prompts.

Consumes the same prompt files as the OpenRouter runner, so this machine needs neither
``cell2sentence`` nor ``scanpy`` nor the 1 GB context h5ads -- only torch, transformers and the two
checkpoints. See ``execution`` in config.yml for why prompts are built once elsewhere.

Caching is per prompt via ``runner_cache``, exactly as for the API arms. It matters here for a
different reason: these are ~144,000 local generations, so a job killed by a scheduler wall-clock
limit must resume rather than restart.
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
    """`base_model:` from a HuggingFace model card's YAML frontmatter, or None.

    Read as plain text rather than parsed as YAML: the frontmatter of a real model card often
    contains tags and nested blocks that a strict parser trips over, and one field is all that is
    needed here. Returns the first entry if the field is a list.
    """
    try:
        from huggingface_hub import hf_hub_download
        card = Path(hf_hub_download(repo, "README.md", repo_type="model")).read_text()
    except Exception as exc:  # noqa: BLE001 - network, auth, missing card: all just mean "unknown"
        print(f"  [verify] could not read {repo} model card ({exc})", file=sys.stderr)
        return None
    lines = card.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":       # closing fence: stop before the prose body
            break
        stripped = line.strip()
        if stripped.startswith("base_model:"):
            value = stripped.split(":", 1)[1].strip().strip("[]'\"")
            if value:                   # `base_model: google/gemma-2-2b`
                return value.split(",")[0].strip()
            # or a YAML list:  base_model:\n  - google/gemma-2-2b
            for nxt in lines[i + 1:i + 4]:
                item = nxt.strip().lstrip("-").strip().strip("'\"")
                if item:
                    return item
            return None
    return None


def verify_base_model(arm, cfg):
    """Check that the base arm really is what the adapted model was fine-tuned from.

    This resolves the study's one genuinely open design decision. A2 is only a CONTROL if it shares
    C2S-Scale's starting point: if C2S-Scale-2B was fine-tuned from ``gemma-2-2b`` and we run
    ``gemma-2-2b-it``, the comparison measures instruction tuning as much as single-cell adaptation.

    The parent is read from the adapted model's MODEL CARD (`base_model:` in the YAML frontmatter),
    not from its config. An earlier version trusted ``config._name_or_path`` and it produced a false
    mismatch that killed all four A2 jobs: transformers sets that field to the repo it LOADED FROM,
    so C2S-Scale reports itself as its own base. It is a provenance field for the load, never for
    the fine-tuning ancestry. A self-reference therefore carries no information and must not be read
    as disagreement -- it is ignored below rather than treated as a mismatch.

    A genuine mismatch still stops the run, because silently producing the wrong control is worse
    than not running.
    """
    target = arm.get("verify_base_of")
    if not target:
        return

    # Authoritative source. Verified by hand 2026-08-12: C2S-Scale-Gemma-2-2B declares
    # `base_model: google/gemma-2-2b`, i.e. the BASE model, not the instruction-tuned `-it`.
    parent = _base_model_from_card(target)
    if parent:
        print(f"  [verify] {target} model card declares base_model: {parent}")
    else:
        from transformers import AutoConfig
        try:
            adapted = AutoConfig.from_pretrained(target)
        except Exception as exc:  # network, auth, gated repo
            print(f"  [verify] could not read {target} ({exc}); base-model check SKIPPED. "
                  f"Resolve before claiming A2 as a control.", file=sys.stderr)
            return
        candidate = getattr(adapted, "_name_or_path", None)
        # Self-reference means "loaded from here", not "descended from here".
        if candidate and Path(str(candidate)).name.lower() != Path(target).name.lower():
            parent = candidate
            print(f"  [verify] no base_model in the model card; falling back to "
                  f"config._name_or_path: {parent}", file=sys.stderr)
    if not parent:
        print(f"  [verify] {target} declares no base model in config OR model card; cannot confirm "
              f"that {arm['model']} is its base. Check by hand before claiming a control.",
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
    """Load weights and tokenizer, narrating each step.

    The narration is not decoration. At 27B every step here takes minutes, and a silent minute is
    indistinguishable from a hang -- which cost an hour of wall-clock on 2026-08-13 when a stall
    after the weights landed could not be located because the function printed nothing until it
    returned. Each stage announces itself with a timestamp BEFORE it starts, so the last line in the
    log names the operation that is currently blocking.
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
        # Needed for left-padded batching; Gemma ships no pad token.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    step("tokenizer ready; calling from_pretrained (weights -> GPU)")
    # Two flags, both required at 27B, for different reasons.
    #
    # low_cpu_mem_usage: a job gets 62 GB of CPU RAM (DefMemPerGPU=62000) and the plain path
    # materialises the whole checkpoint in RAM first. gemma-2-27b ships FP32 weights totalling
    # 101 GB, so without this it is OOM-killed on startup, surfacing as SLURM's "Kill task failed".
    #
    # device_map: loads each shard STRAIGHT TO THE GPU instead of building the model in host memory
    # and moving it with `.to(device)` afterwards. That distinction is what makes 27B viable here.
    # The weights live on /scratch/global, which is NFS, and safetensors mmaps its files -- so the
    # `.to()` pass walks the whole 51 GB as random page faults over the network. Measured
    # 2026-08-13: three shards sat in uninterruptible D state for over an hour, 43 s of CPU between
    # them, GPU memory frozen at 24.8 GB of an expected 54 GB, and no batch ever started. It reads
    # as a hang, not as slow I/O. device_map turns that into a sequential shard-by-shard read.
    kwargs = {"torch_dtype": getattr(torch, dtype), "low_cpu_mem_usage": True}
    if device == "cuda":
        kwargs["device_map"] = {"": 0}      # the single GPU this job was allocated
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    step("from_pretrained returned")
    if device != "cuda":
        model = model.to(device)
        step(f"moved to {device}")
    model = model.eval()
    step("eval() done")

    # NO `sum(p.numel() for p in model.parameters())` here. It was added purely to print a
    # parameter count and is the prime suspect for the 2026-08-13 stall: under `device_map`,
    # accelerate wraps the model in dispatch hooks, and walking every parameter can interact with
    # them in ways a plainly-placed model does not. A diagnostic that may hang the run is worse than
    # no diagnostic. Memory allocated on the device says the same thing and costs one CUDA call.
    if device == "cuda":
        step(f"weights occupy {torch.cuda.memory_allocated() / 2**30:.1f} GB of "
             f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")
    step(f"MODEL READY: {model_id} on {device} ({dtype})")
    return model, tokenizer, device


def generate(model, tokenizer, device, prompts, seed, max_new_tokens=24, batch_size=16,
             report_every=5):
    """Greedy-free sampling, seeded per repeat so the three repeats genuinely differ.

    Progress is reported as a rate and an ETA rather than a bare counter. On a GPU job the only
    question that matters mid-run is "will this finish inside the walltime", and a raw
    `256/21000` cannot answer it. Peak VRAM is reported alongside because it is what says whether
    --batch-size can be raised: a 2B model in bf16 leaves most of a 96 GB card unused, and the
    only honest way to size the batch is to measure the first one.

    A GENERATOR, yielding ``(prompt, text)`` as each batch lands rather than returning the whole
    file at the end. That lets the caller persist every batch immediately. Returning one list meant
    a job killed at the walltime lost every generation for that file -- up to 1,000 prompts of GPU
    time -- which is exactly the file-granularity flaw the response cache exists to avoid.
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
            ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                 pad_token_id=tokenizer.pad_token_id)
        for prompt, row, prompt_ids in zip(batch, ids, enc["input_ids"]):
            # Only the continuation, never the echoed prompt.
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
    dataset, arm_name, mode, variant, seed = parse_prompt_file(prompt_file)
    arm = cfg["arms"][arm_name]
    frame = pd.read_csv(prompt_file)
    if limit:
        frame = frame.head(limit)

    cache = ResponseCache(Path(cache_root) / f"{arm_name}__{dataset}.jsonl", arm["model"], seed=seed)
    hits, misses = cache.stats(frame["prompt"].tolist())
    print(f"[{arm_name} {dataset} {mode} {variant} seed{seed}] "
          f"{len(frame)} prompts: {hits} cached, {misses} to generate")
    if dry_run:
        return None

    # Deduplicated for the same reason the API runner is: identical prompts recur (every cell in
    # M5 shares one), and generating them twice wastes GPU time as surely as it wastes money.
    todo = list(dict.fromkeys(p for p in frame["prompt"] if cache.get(p) is None))
    if todo:
        # Resolved only when there is something to generate. model_bundle is a CALLABLE, not a
        # loaded model: a fully cached shard then needs no checkpoint at all, so re-deriving
        # predictions after a parser change costs seconds on any machine instead of a 5 GB load
        # on a GPU. That mattered immediately -- the <ctrl100> fix had to be applied to 128
        # already-generated responses.
        model, tokenizer, device = model_bundle()
        kw = {"batch_size": batch_size} if batch_size else {}
        print(f"    starting generation: {len(todo)} prompts, batch {batch_size or 16}", flush=True)
        # Written as each batch arrives, and ResponseCache fsyncs every entry, so an interrupted
        # job resumes from the last completed BATCH rather than restarting the file.
        for prompt, text in generate(model, tokenizer, device, todo, seed, **kw):
            cache.put(prompt, text, meta={"seed": seed})

    frame["predicted_labels"] = [extract_prediction(cache.get(p)) for p in frame["prompt"]]

    # Base models are not instruction-tuned to emit a bare label, so they answer in prose. Report
    # the rate: a low score that is really a parsing failure would otherwise be read as the base
    # model being biologically worse, inflating the apparent value of fine-tuning.
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

    # A truncated run must not write where a complete one would. Sharing the tree would leave a
    # 32-row predictions.csv sitting exactly where score.py expects 1,000, and a cache whose
    # entries are real but whose coverage is not -- the later full run would look "already done".
    # Diverting both is what makes the smoke test safe to throw away.
    if args.limit or args.max_files:
        out_root = tree / "smoke" / "predictions"
        cache_root = tree / "smoke" / "cache"
        print(f"SMOKE TEST: limit={args.limit} max_files={args.max_files}\n"
              f"  writing to {out_root} -- delete this tree before the real run", file=sys.stderr)
    else:
        out_root = tree / "predictions"
        cache_root = tree / "cache"

    for arm_name in arms:
        arm = cfg["arms"][arm_name]
        if arm["kind"] != "local":
            print(f"[skip] {arm_name}: not a local arm")
            continue
        files = sorted(f for f in prompts_dir.glob("*.csv")
                       if parse_prompt_file(f)[1] == arm_name
                       and (not args.datasets or parse_prompt_file(f)[0] in args.datasets))
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
        # Shard-level progress. A shard is 21 files, so the per-batch bar inside generate() only
        # ever shows one twenty-first of the job; without this line there is no way to tell from
        # the log whether a shard is nearly done or barely started.
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
