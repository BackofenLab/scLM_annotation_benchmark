#!/usr/bin/env python
"""Run the OpenRouter arms (A1, A4, A5) over materialised prompts.

Reads prompt files written by ``sentences.py --materialize`` and ``cluster.py``, so it builds no
sentences itself -- see ``execution`` in config.yml for why that separation exists.

Every call goes through ``runner_cache.ResponseCache``, keyed on the prompt text, so an interrupted
run resumes where it stopped rather than re-paying for answers it already has. Retry and backoff
conventions mirror ``models/openrouter_LLM_blind/src/predict.py`` (honour ``retry-after`` on 429,
exponential backoff on 5xx) so behaviour under load matches the benchmark that produced the numbers
this study is compared against.
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
    """Request body. Provider routing is PINNED, unlike the benchmark's.

    The benchmark leaves `sort: price` with `allow_fallbacks: true` and an empty whitelist, so
    requests land on whichever provider is cheapest at that moment -- and providers serve different
    quantizations of the same weights. That is tolerable for the benchmark's large effects but not
    for an ablation measuring what a tissue string buys, where a silent quantization change
    mid-run would be indistinguishable from the effect being measured.
    """
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


# Vendors whose models take no reasoning block at all, and those that take a token budget rather
# than an effort level. Mirrors models/openrouter_LLM_blind/src/predict.py so this study asks for
# the same reasoning behaviour the benchmark got.
NON_REASONING_PREFIXES = ("meta-llama", "mistralai")
MAX_TOKEN_REASONING_PREFIXES = ("anthropic", "google")
MANDATORY_REASONING_PATTERNS = ("deepseek-r1",)


def reasoning_block(model, cfg):
    """The `reasoning` field for one model, or ``None``.

    Sending nothing is NOT equivalent to sending a default: GPT-OSS are reasoning models and the
    benchmark set this explicitly, so omitting it would leave the provider's own default in force
    and the anchor check would compare two different reasoning budgets.
    """
    if not cfg:
        return None
    prefix = model.split("/")[0] if "/" in model else ""
    suffix = model.split("/")[-1] if "/" in model else model

    if prefix in NON_REASONING_PREFIXES:
        return None
    if any(p in suffix for p in MANDATORY_REASONING_PATTERNS):
        # Reasoning cannot be turned off or budgeted for these; only `exclude` is honoured.
        return {"exclude": cfg["exclude"]} if cfg.get("exclude") is not None else None

    block = {}
    if cfg.get("enabled") is not None:
        block["enabled"] = cfg["enabled"]
    has_effort, has_budget = cfg.get("effort") is not None, cfg.get("max_tokens") is not None
    if has_effort and has_budget:
        # Both configured: the vendor decides which one it actually accepts.
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
    """One completion, with the benchmark's retry semantics. Returns ``(text, meta)``.

    A call that never succeeds returns ``None`` text rather than raising: the study scores an
    unanswered cell as wrong (as the benchmark does), and one dead prompt must not abort a run of
    a hundred thousand.

    Errors carry a ``transient`` flag so the caller knows whether the answer is worth caching. A
    400 (malformed request) will fail identically forever and is worth remembering; a 429 says only
    that the provider was busy at that instant, and remembering it would permanently mark a cell
    unanswered over a momentary condition.

    Retry budget is deliberately long. With `1.5 * 2**(n-1)` over 5 attempts the window was 22s,
    which a sustained upstream rate limit outlasts trivially -- observed in practice: DeepInfra
    rate-limited gpt-oss-120b and 199 calls exhausted their retries inside two minutes. At
    `2.0 * 2**(n-1)` over 8 attempts the window is ~4min, which rides out a typical throttle.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = build_payload(model, prompt, cfg, seed)

    def wait(attempt, retry_after=None):
        """Backoff with jitter.

        The jitter is not cosmetic. Every worker in the pool hits the same upstream limit at the
        same moment, and without jitter they all sleep the identical interval and retry in lockstep
        -- reconverging into the burst that caused the throttle. Randomising spreads them out.
        """
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
            # A 200 does not guarantee JSON. Observed in practice: a provider returned a 200 whose
            # body failed to parse, r.json() raised, and the exception propagated out of the thread
            # pool and killed a 74,346-call run over one bad response. Treated as retryable, since
            # it is far more likely to be a transient provider hiccup than a permanent condition.
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
                # Recorded per call so the caption can state which provider actually served the
                # study, rather than which one the config asked for.
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
        # Anything else (400 malformed, 401 bad key, 404 unknown model) is a property of the
        # request, not of the moment: NOT transient, so it is cached and never re-paid for.
        return None, {"error": f"HTTP {r.status_code}: {r.text[:300]}", "transient": False}

    return None, {"error": f"exhausted {max_retries} attempts", "transient": True}


#: Vocabulary tokens that survive `skip_special_tokens=True` because the tokenizer does not class
#: them as special. C2S-Scale-Gemma-2-2B terminates every answer with `<ctrl100>`, so its raw output
#: is `'\nneural cell.<ctrl100>'`. Left in place, no prediction would ever match a Cell Ontology
#: term and the whole arm would score at zero -- looking like a model failure rather than a parsing
#: bug. Found by the smoke test, in 128 of 128 responses.
CONTROL_TOKEN_RE = re.compile(r"<[^>\s]{1,32}>")

#: Raw non-printable bytes, which the angle-bracket pattern above does not catch. C2S-Scale emitted
#: `'\n\x12monary gland epithelial cell'` -- a DC2 byte fused to the label. Only 2 predictions in
#: 318,966 carry one and both are truncated beyond rescue, so removing them changes no current
#: number. It is here because the failure mode is silent and asymmetric: a leading control byte
#: turns an otherwise exact ontology match into an unmappable string, and that would be scored as a
#: wrong answer rather than as a parsing artefact.
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

#: Header lines that introduce an answer rather than being one. C2S-Scale answers in a multi-cell
#: transcript format even when asked about a single cell:
#:
#:     Cell1:
#:     retinal rod cell.
#:     Cell:
#:     Mueller cell.
#:
#: Taking the first non-empty line therefore captured `Cell1:` and threw the actual prediction away.
#: It cost 761 A3 predictions that do resolve to a Cell Ontology term -- read as the model failing
#: when it had in fact answered.
#:
#: The negative lookahead is essential: a bare `cell` is a LEGITIMATE answer (the ontology root, and
#: what nonsense is mapped to), so it must never be skipped as a header.
#: `\s*` after `cell` matters: the model writes both `Cell1:` and `Cell 1:`, and an earlier version
#: without it let `Cell 1:` and `Cell 2` through as if they were answers.
ANSWER_HEADER_RE = re.compile(r"^(?!cell$)(cell\s*\w{0,8}\s*:?|:)$", re.I)


def extract_prediction(text):
    """First non-empty line, stripped. Mirrors how the benchmark parses a label out.

    Also removes angle-bracket vocabulary tokens. That is a no-op for the API arms, whose responses
    never contain them, and essential for the local ones.
    """
    if not text:
        return ""
    cleaned = CONTROL_CHAR_RE.sub("", CONTROL_TOKEN_RE.sub("", str(text)))
    for line in cleaned.splitlines():
        line = line.strip().strip('"').strip("'").rstrip(".").strip()
        if not line or ANSWER_HEADER_RE.match(line):
            continue          # a `Cell1:` header introduces the answer; it is not the answer
        return line
    return ""


def parse_prompt_file(path):
    """(dataset, arm, mode, variant, seed) from the materialised filename."""
    stem = Path(path).stem
    dataset, arm, mode, variant, seed = stem.split("__")
    return dataset, arm, mode, variant, int(seed.removeprefix("seed"))


def _pooled_session(workers):
    """A Session whose connection pool is at least as large as the thread pool.

    ``requests.Session()`` defaults to ``pool_maxsize=10``. With 48 workers, urllib3 discards every
    connection beyond the tenth instead of reusing it, and the discarded sockets sit in CLOSE_WAIT
    until something reaps them. Observed after ~9 hours of a 74,346-call run: 9 sockets in
    CLOSE_WAIT, one ESTABLISHED, zero throughput, and the process still alive -- a hang that looks
    exactly like a provider outage while the API is in fact answering normally.

    Sizing the pool to the worker count removes the churn. `pool_block` makes a worker WAIT for a
    free connection rather than open an unpooled one, so the pool size is a real bound rather than
    a hint, and the failure mode becomes slowness instead of leaked sockets.
    """
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=max(workers, 10),
        pool_maxsize=max(workers, 10),
        # pool_block STAYS FALSE. Setting it True was an over-correction that turned a socket leak
        # into a hard deadlock: `block=True` waits for a free connection with NO timeout, so once
        # connections go stale in the pool every worker parks forever. Observed on A4 -- 43 sockets
        # in CLOSE_WAIT, one ESTABLISHED, 8 seconds of CPU across 32 minutes, zero progress.
        #
        # Sizing the pool to the worker count is what actually fixes the churn; blocking adds
        # nothing on top of it, because with maxsize >= workers there is never contention to block
        # on in the first place. If a connection does go bad, discarding it and opening another is
        # the behaviour we want.
        pool_block=False,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def run_file(prompt_file, cfg, api_key, out_root, cache_root, dry_run=False):
    dataset, arm_name, mode, variant, seed = parse_prompt_file(prompt_file)
    arm = cfg["arms"][arm_name]
    model = arm["model"]
    frame = pd.read_csv(prompt_file)

    cache = ResponseCache(Path(cache_root) / f"{arm_name}__{dataset}.jsonl", model, seed=seed)
    hits, _misses = cache.stats(frame["prompt"].tolist())
    # Reported AFTER deduplication, because rows and calls differ by ~28x on the cluster arm and the
    # row figure reads as a cost estimate that is wildly wrong. "966 to call" for a file that issues
    # seven requests is not a rounding error, it is a different quantity.
    pending = len({p for p in frame["prompt"] if cache.get(p) is None})
    print(f"[{arm_name} {dataset} {mode} {variant} seed{seed}] "
          f"{len(frame)} rows, {frame['prompt'].nunique()} unique: {hits} rows cached, "
          f"{pending} call(s) to make")
    if dry_run:
        return None

    session = _pooled_session(int(cfg.get("provider", {}).get("max_workers", 16)))
    # DEDUPLICATED, and this is not a micro-optimisation. The cluster arm writes one row per eval
    # cell but every cell in a cluster carries the identical prompt, so a naive pass would issue
    # ~999 calls where ~30 distinct prompts exist -- a 30x overcharge, since nothing is cached yet
    # when the list is built. dict.fromkeys preserves order.
    todo = list(dict.fromkeys(p for p in frame["prompt"] if cache.get(p) is None))
    transient = 0  # defined here, not inside the branch: a fully-cached file skips the pool
    if todo:
        workers = int(cfg.get("provider", {}).get("max_workers", 16))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(call_once, session, model, p, cfg, seed, api_key): p
                       for p in todo}
            done = 0
            for fut in as_completed(futures):
                prompt = futures[fut]
                # Second guard, deliberately belt-and-braces with the one inside call_once. Any
                # exception escaping a worker would otherwise re-raise here and abort the entire
                # run -- which is exactly what happened once. A prompt that cannot be answered is
                # recorded as an error and scored as wrong, like any other unanswered cell.
                try:
                    text, meta = fut.result()
                except Exception as exc:  # noqa: BLE001 - one bad call must not kill the run
                    text, meta = None, {"error": f"worker raised {type(exc).__name__}: {exc}"}
                # Permanent failures are cached (a malformed request fails identically forever, so
                # remembering it saves paying twice). TRANSIENT ones are NOT: a 429 says the
                # provider was busy at that instant, and caching it would permanently mark a cell
                # unanswered over a momentary condition. That is not hypothetical -- a DeepInfra
                # throttle wrote 199 cached failures that no rerun would ever have retried, and
                # they would have scored as wrong in the final numbers without ever showing up as
                # a run failure.
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

    # Written for the API arms too, in the same shape run_local.py uses, so the answer-format rate
    # is comparable ACROSS arms rather than existing only for the local ones. Without it the base
    # model's 39.5% prose rate has no baseline to be judged against, and "A2 scores lower" cannot be
    # separated from "A2 does not answer in the required format".
    long_answers = int((frame["predicted_labels"].astype(str).str.split().str.len() > 8).sum())
    stats = {"arm": arm_name, "dataset": dataset, "mode": mode, "variant": variant, "seed": seed,
             "n": len(frame), "empty": unanswered, "over_8_words": long_answers}
    write_atomic(out.parent / "parse_stats.json", json.dumps(stats, indent=2))
    print(f"    -> {out}  ({unanswered} unanswered, scored as wrong)")
    # Loud, because the file on disk looks complete either way. Transient failures were left out of
    # the cache on purpose, so simply re-running this arm fills them in and rewrites the file --
    # but nothing else will tell you it needs doing.
    if transient:
        print(f"    !! {transient} call(s) failed transiently and were NOT cached. This file is "
              f"INCOMPLETE -- re-run this arm to fill them in before scoring.", file=sys.stderr)
    return out


def provider_mix(root, cfg, arms, prompts_dir, cache_root):
    """Report which provider actually served each (arm, mode).

    Routing is unrestricted (see config.yml), so quantization varies within an arm and is a live
    confound on effects this study calls small. The confound cannot be removed after the fact, but
    it can be measured: if one mode was served mostly at fp4 and another mostly at bf16, a mode
    difference is partly a hardware difference and the numbers must not be read at face value.

    An even split across modes is the reassuring outcome; a lopsided one is a finding in itself.
    """
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
    print("\nRead this before using any number: modes served by different providers are not "
          "strictly comparable,\nbecause the endpoints differ in quantization (fp4/fp8/bf16/fp16).")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--config", default=str(root / "config.yml"))
    p.add_argument("--host", default="laptop", help="which host's arms to run (see config.yml)")
    p.add_argument("--arms", nargs="+", help="override the host's arm list")
    p.add_argument("--datasets", nargs="+")
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
        # Expected, not accidental: the pin was dropped by decision after DeepInfra throttled the
        # run. Still printed every time, because it changes how the results may be read.
        print("NOTE: provider routing is unrestricted, so quantization varies within an arm and is\n"
              "      an uncontrolled confound. Run --provider-mix before using any number.",
              file=sys.stderr)

    arms = args.arms or cfg["execution"]["hosts"][args.host]
    prompts_dir = root / cfg["execution"]["prompts_dir"]
    if not prompts_dir.exists():
        raise SystemExit(f"no prompts at {prompts_dir}; run sentences.py --materialize first")

    if args.provider_mix:
        provider_mix(root, cfg, arms, prompts_dir, root / "results" / "cache")
        return 0

    files = sorted(f for f in prompts_dir.glob("*.csv")
                   if parse_prompt_file(f)[1] in arms
                   and (not args.datasets or parse_prompt_file(f)[0] in args.datasets))
    if not files:
        raise SystemExit(f"no prompt files for arms {arms} in {prompts_dir}")

    out_root = root / cfg["output"]["predictions_dir"]
    cache_root = root / "results" / "cache"
    for f in files:
        run_file(f, cfg, api_key, out_root, cache_root, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
