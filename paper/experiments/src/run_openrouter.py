#!/usr/bin/env python
"""Answer the prompt files of the OpenRouter arms through the OpenRouter API.

Reads ``<study tree>/prompts/`` (from ``sentences.py --materialize`` and ``cluster.py``) and sends
each distinct prompt once, with the arm's model, the file's seed and the ``provider``,
``reasoning`` and ``repeats`` settings of config.yml. Answers go to the response cache
(``<study tree>/cache/``), so a run resumes where it stopped. Writes
``<study tree>/predictions/<arm>/<dataset>/<mode>/<variant>/seed<seed>/predictions.csv`` and
``parse_stats.json``. Needs ``OPENROUTER_API_KEY``; every call is billed.

    python src/run_openrouter.py --arms A1_llm_percell [--study lists] [--dry-run]
    python src/run_openrouter.py --provider-mix
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner_cache import ResponseCache, write_atomic  # noqa: E402

API_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def build_payload(model, prompt, cfg, seed):
    """The request body: model, prompt, seed, temperature if set, provider and reasoning."""
    provider_cfg = cfg.get("provider", {})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "seed": seed,
    }
    temperature = cfg["repeats"].get("temperature")
    if temperature is not None:
        payload["temperature"] = temperature

    provider = {"allow_fallbacks": bool(provider_cfg.get("allow_fallbacks", False))}
    if provider_cfg.get("only"):
        provider["only"] = list(provider_cfg["only"])
    if provider_cfg.get("sort"):
        provider["sort"] = provider_cfg["sort"]
    payload["provider"] = provider

    reasoning = reasoning_block(model, cfg.get("reasoning") or {})
    if reasoning:
        payload["reasoning"] = reasoning
    return payload


# Vendors whose models get no reasoning field, vendors whose models get a token budget instead of
# an effort level, and models that always reason; as in the benchmark's OpenRouter wrapper.
NON_REASONING_PREFIXES = ("meta-llama", "mistralai")
MAX_TOKEN_REASONING_PREFIXES = ("anthropic", "google")
MANDATORY_REASONING_PATTERNS = ("deepseek-r1",)


def reasoning_block(model, cfg):
    """The ``reasoning`` field of a request to ``model``, or ``None`` if none is sent."""
    if not cfg:
        return None
    prefix = model.split("/")[0] if "/" in model else ""
    suffix = model.split("/")[-1] if "/" in model else model

    if prefix in NON_REASONING_PREFIXES:
        return None
    if any(p in suffix for p in MANDATORY_REASONING_PATTERNS):
        # These models accept only `exclude`.
        return {"exclude": cfg["exclude"]} if cfg.get("exclude") is not None else None

    block = {}
    if cfg.get("enabled") is not None:
        block["enabled"] = cfg["enabled"]
    has_effort, has_budget = cfg.get("effort") is not None, cfg.get("max_tokens") is not None
    if has_effort and has_budget:
        # With both set, the vendor decides which one is sent.
        if prefix in MAX_TOKEN_REASONING_PREFIXES:
            block["max_tokens"] = cfg["max_tokens"]
        else:
            block["effort"] = cfg["effort"]
    elif has_effort:
        block["effort"] = cfg["effort"]
    elif has_budget:
        block["max_tokens"] = cfg["max_tokens"]
    if cfg.get("exclude") is not None:
        block["exclude"] = cfg["exclude"]
    return block or None


def call_once(session, model, prompt, cfg, seed, api_key, max_retries=8, backoff=2.0):
    """One request with retries. Returns ``(text, meta)``, with ``text`` None if no answer came.

    ``meta["transient"]`` marks an error that a later run may not meet again (not cached), and
    ``meta["fatal"]`` an error of the key or account (401, 402, 403) that stops the run. The wait
    before attempt n + 1 is ``backoff * 2**(n - 1)`` seconds, or the 429's ``retry-after``.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = build_payload(model, prompt, cfg, seed)

    def wait(attempt, retry_after=None):
        """Sleep for the wait times a random factor between 0.7 and 1.3, so that the workers do
        not retry at the same moment."""
        base = float(retry_after) if retry_after else backoff * (2 ** (attempt - 1))
        time.sleep(base * random.uniform(0.7, 1.3))

    for attempt in range(1, max_retries + 1):
        try:
            r = session.post(API_URL, headers=headers, json=payload, timeout=120)
        except requests.RequestException as exc:
            if attempt == max_retries:
                return None, {"error": f"request failed: {exc}", "transient": True}
            wait(attempt)
            continue

        if r.status_code == 200:
            # A 200 whose body is not JSON is retried.
            try:
                body = r.json()
            except ValueError as exc:
                if attempt == max_retries:
                    return None, {"error": f"200 with unparseable body: {exc}",
                                  "body_head": r.text[:200], "transient": True}
                wait(attempt)
                continue
            choice = (body.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content")
            return text, {
                # The provider that served this call.
                "provider": body.get("provider"),
                "finish_reason": choice.get("finish_reason"),
                "usage": body.get("usage"),
            }

        if r.status_code == 429:
            wait(attempt, r.headers.get("retry-after"))
            continue
        if 500 <= r.status_code < 600:
            wait(attempt)
            continue
        # 401 (key), 402 (credit) and 403 (guardrail) fail every later call alike: not cached, and
        # the run stops.
        if r.status_code in (401, 402, 403):
            return None, {"error": f"HTTP {r.status_code}: {r.text[:300]}", "transient": True,
                          "fatal": True}
        # Other errors, such as 400 or 404, belong to the request and are cached.
        return None, {"error": f"HTTP {r.status_code}: {r.text[:300]}", "transient": False}

    return None, {"error": f"exhausted {max_retries} attempts", "transient": True}


#: Tokens in angle brackets that ``skip_special_tokens`` keeps, such as the ``<ctrl100>`` that
#: C2S-Scale-Gemma-2-2B puts after its answers.
CONTROL_TOKEN_RE = re.compile(r"<[^>\s]{1,32}>")

#: Control characters other than tab and newline, which can stand before a label in the answers of
#: the local models.
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

#: Lines that introduce an answer, such as ``Cell1:`` or ``Cell 1:``, which the local models write
#: before a label. A bare ``cell`` (the ontology root) is an answer and does not match.
ANSWER_HEADER_RE = re.compile(r"^(?!cell$)(cell\s*\w{0,8}\s*:?|:)$", re.I)


def extract_prediction(text):
    """The label of a response: its first non-empty line that is not a header, without control
    tokens, control characters, quotes and a final period."""
    if not text:
        return ""
    cleaned = CONTROL_CHAR_RE.sub("", CONTROL_TOKEN_RE.sub("", str(text)))
    for line in cleaned.splitlines():
        line = line.strip().strip('"').strip("'").rstrip(".").strip()
        if not line or ANSWER_HEADER_RE.match(line):
            continue          # a header such as `Cell1:`
        return line
    return ""


def parse_prompt_file(path):
    """``(dataset, arm, mode, variant, seed)`` from the name of a prompt file."""
    stem = Path(path).stem
    dataset, arm, mode, variant, seed = stem.split("__")
    return dataset, arm, mode, variant, int(seed.removeprefix("seed"))


def _pooled_session(workers):
    """A Session whose connection pool holds at least one connection per worker.

    ``requests.Session()`` keeps 10 connections per host, and urllib3 discards the connections of
    further workers instead of reusing them.
    """
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=max(workers, 10),
        pool_maxsize=max(workers, 10),
        # With pool_block=True a worker would wait for a free connection without a timeout.
        pool_block=False,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def run_file(prompt_file, cfg, api_key, out_root, cache_root, dry_run=False):
    """Answer one prompt file and write its predictions; returns their path, None on a dry run."""
    dataset, arm_name, mode, variant, seed = parse_prompt_file(prompt_file)
    arm = cfg["arms"][arm_name]
    model = arm["model"]
    frame = pd.read_csv(prompt_file)

    cache = ResponseCache(Path(cache_root) / f"{arm_name}__{dataset}.jsonl", model, seed=seed)
    hits, _misses = cache.stats(frame["prompt"].tolist())
    # Calls to make, counted per distinct prompt.
    pending = len({p for p in frame["prompt"] if cache.get(p) is None})
    print(f"[{arm_name} {dataset} {mode} {variant} seed{seed}] "
          f"{len(frame)} rows, {frame['prompt'].nunique()} unique: {hits} rows cached, "
          f"{pending} call(s) to make")
    if dry_run:
        return None

    session = _pooled_session(int(cfg.get("provider", {}).get("max_workers", 16)))
    # Each distinct prompt once, in file order; the cells of one cluster share a prompt.
    todo = list(dict.fromkeys(p for p in frame["prompt"] if cache.get(p) is None))
    transient = 0
    if todo:
        workers = int(cfg.get("provider", {}).get("max_workers", 16))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(call_once, session, model, p, cfg, seed, api_key): p
                       for p in todo}
            done = 0
            for fut in as_completed(futures):
                prompt = futures[fut]
                # An exception in a worker is a transient error of its prompt.
                try:
                    text, meta = fut.result()
                except Exception as exc:  # noqa: BLE001
                    text, meta = None, {"error": f"worker raised {type(exc).__name__}: {exc}",
                                        "transient": True}
                if meta.get("fatal"):
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise SystemExit(f"{arm_name} {dataset} {mode} seed{seed}: {meta['error']}. "
                                     f"Stopped; nothing of it was cached and no file was written.")
                # Transient failures are not cached, so that a rerun asks again.
                if meta.get("transient"):
                    transient += 1
                else:
                    cache.put(prompt, text, meta=meta)
                done += 1
                if done % 200 == 0:
                    print(f"    {done}/{len(todo)}")

    frame["predicted_labels"] = [extract_prediction(cache.get(p)) for p in frame["prompt"]]
    unanswered = int((frame["predicted_labels"] == "").sum())

    out = (Path(out_root) / arm_name / dataset / mode / variant / f"seed{seed}" /
           "predictions.csv")
    write_atomic(out, frame[["cell_id", "true_labels", "predicted_labels"]].to_csv(index=False))

    # Answer-format counts, as run_local.py writes them.
    long_answers = int((frame["predicted_labels"].astype(str).str.split().str.len() > 8).sum())
    stats = {"arm": arm_name, "dataset": dataset, "mode": mode, "variant": variant, "seed": seed,
             "n": len(frame), "empty": unanswered, "over_8_words": long_answers}
    write_atomic(out.parent / "parse_stats.json", json.dumps(stats, indent=2))
    print(f"    -> {out}  ({unanswered} unanswered, scored as wrong)")
    # The file has empty answers for these prompts until a rerun fills them in.
    if transient:
        print(f"    {transient} call(s) failed transiently and were not cached; run this arm "
              f"again before scoring to fill them in.", file=sys.stderr)
    return out


def provider_mix(root, cfg, arms, prompts_dir, cache_root):
    """Print the share of cached answers that each provider served, per arm and mode."""
    counts = {}
    for f in sorted(prompts_dir.glob("*.csv")):
        dataset, arm_name, mode, variant, seed = parse_prompt_file(f)
        if arm_name not in arms:
            continue
        cache = ResponseCache(Path(cache_root) / f"{arm_name}__{dataset}.jsonl",
                              cfg["arms"][arm_name]["model"], seed=seed)
        if not len(cache):
            continue
        for prompt in pd.read_csv(f)["prompt"]:
            rec = cache.record(prompt)
            if rec:
                counts.setdefault((arm_name, mode), {})
                key = rec.get("provider") or "unknown"
                counts[(arm_name, mode)][key] = counts[(arm_name, mode)].get(key, 0) + 1

    if not counts:
        print("no cached responses yet; nothing to audit")
        return
    print(f"\n{'arm':20s} {'mode':5s} {'n':>7s}  provider split")
    for (arm_name, mode) in sorted(counts):
        by_prov = counts[(arm_name, mode)]
        total = sum(by_prov.values())
        share = ", ".join(f"{p} {100 * n / total:.0f}%"
                          for p, n in sorted(by_prov.items(), key=lambda kv: -kv[1]))
        print(f"{arm_name:20s} {mode:5s} {total:7d}  {share}")
    print("\nProviders can serve a model at different quantizations, so modes served by "
          "different providers\nare not strictly comparable.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--config", default=str(root / "config.yml"))
    p.add_argument("--host", default="laptop", help="which host's arms to run (see config.yml)")
    p.add_argument("--arms", nargs="+", help="override the host's arm list")
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--study", default="ablation", help="the study tree (see `studies` in config)")
    p.add_argument("--dry-run", action="store_true",
                   help="report cache hits and what would be called, then exit")
    p.add_argument("--provider-mix", action="store_true",
                   help="report which provider served each (arm, mode) and exit; run this before "
                        "using any number, since routing is unrestricted and quantization varies")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("API_KEY")
    if not api_key and not (args.dry_run or args.provider_mix):
        raise SystemExit("set OPENROUTER_API_KEY (or API_KEY) in the environment")

    if not cfg.get("provider", {}).get("only"):
        print("provider routing is unrestricted (provider.only is empty), so the quantization can\n"
              "differ between calls of one arm; --provider-mix shows which providers answered.",
              file=sys.stderr)

    arms = args.arms or cfg["execution"]["hosts"][args.host]
    tree = root / cfg["studies"][args.study]
    prompts_dir = tree / "prompts"
    if not prompts_dir.exists():
        raise SystemExit(f"no prompts at {prompts_dir}; run sentences.py --materialize first")

    if args.provider_mix:
        provider_mix(root, cfg, arms, prompts_dir, tree / "cache")
        return 0

    def wanted(f):
        dataset, arm, _mode, _variant, seed = parse_prompt_file(f)
        return (arm in arms and (not args.datasets or dataset in args.datasets)
                and (not args.seeds or seed in args.seeds))

    files = [f for f in sorted(prompts_dir.glob("*.csv")) if wanted(f)]
    if not files:
        raise SystemExit(f"no prompt files for arms {arms} in {prompts_dir}")

    for f in files:
        run_file(f, cfg, api_key, tree / "predictions", tree / "cache", dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
