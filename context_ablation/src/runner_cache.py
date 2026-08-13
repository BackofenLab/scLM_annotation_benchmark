#!/usr/bin/env python
"""Per-prompt response cache shared by both runners.

Caching is not a nicety here. The full study is ~150,000 generations split across two machines and
an API that charges per call; a run that dies at 80% and cannot resume wastes real money and a day
of wall-clock. The benchmark's own runners cache at FILE granularity -- `runModelScript` in main.nf
copies a whole `predictions.csv` if it exists -- which is fine when a task is one model on one
dataset, but useless here: a single (arm, mode, variant, seed) file is 1,000 calls, and losing the
last one throws away the other 999.

So the cache is keyed per PROMPT, on everything that changes the request:

    key = sha256(model | prompt | seed)

Keying on content rather than on position means the cache is correct by construction under the
things that actually change during a study like this. Reordering cells, adding a dataset, changing
`n_calls` for M5, or re-running one mode all reuse whatever was already answered. And crucially,
EDITING A TEMPLATE INVALIDATES EXACTLY THE PROMPTS IT CHANGED -- no manual cache-busting, and no
risk of silently serving answers to a question we no longer ask, which is the failure mode that
would quietly corrupt an ablation.

The model id is in the key because the same prompt sent to two models is two different results.
The seed is in the key for the same reason -- it is sent in the request body. An earlier version
left it out, arguing the seed had already changed the prompt via the shuffled candidate list. That
holds only for the modes that HAVE a candidate list: in M1 and M3 the prompt is seed-independent,
so all three repeats were served from one set of calls and showed zero variance by construction.
That reads as determinism rather than as the artefact it is, which is exactly the kind of quiet
wrongness this cache is supposed to prevent.

Storage is one JSONL per (arm, dataset) appended as answers arrive, so a kill -9 loses at most the
in-flight batch. JSONL rather than a database because it is append-only, human-greppable, and
survives being copied between the two machines with `scp`.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path


def prompt_key(model: str, prompt: str, seed=None) -> str:
    """Content-addressed key over everything that changes the REQUEST.

    Changing a template changes the key, which is the point.

    The seed is part of the key. An earlier version left it out, reasoning that the seed had
    already changed the prompt because the candidate list is shuffled per cell -- but that only
    holds for the modes that HAVE a candidate list. In M1 and M3 the prompt is seed-independent, so
    excluding the seed served all three repeats from one set of calls: the repeats showed zero
    variance by construction and looked like determinism rather than an artefact. Since the seed is
    sent in the request body, two identical prompts at different seeds are different requests.
    """
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(prompt.encode("utf-8"))
    if seed is not None:
        digest.update(b"\x00")
        digest.update(str(seed).encode("utf-8"))
    return digest.hexdigest()


class ResponseCache:
    """Append-only JSONL cache of prompt -> response, safe for concurrent writers.

    Usage::

        cache = ResponseCache(path, model)
        if (hit := cache.get(prompt)) is None:
            hit = call_the_model(prompt)
            cache.put(prompt, hit, meta={"provider": ...})
    """

    def __init__(self, path, model, seed=None):
        self.path = Path(path)
        self.model = model
        self.seed = seed
        self._lock = threading.Lock()
        self._entries = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        """Read what is already there, tolerating a truncated final line.

        A run killed mid-write leaves a partial JSON object on the last line. Dropping it is right:
        that one prompt is simply re-asked, whereas refusing to load would discard every answer
        before it.
        """
        if not self.path.exists():
            return
        kept = dropped = 0
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._entries[rec["key"]] = rec
                    kept += 1
                except (json.JSONDecodeError, KeyError):
                    dropped += 1
        if dropped:
            print(f"  [cache] {self.path.name}: {kept} entries loaded, {dropped} unreadable line(s) "
                  f"ignored (a killed run leaves a partial last line; those prompts re-run)")

    def __len__(self):
        return len(self._entries)

    def get(self, prompt):
        rec = self._entries.get(prompt_key(self.model, prompt, self.seed))
        return rec["response"] if rec else None

    def record(self, prompt):
        """The whole cached record, or ``None``. Used by the provider-mix audit, which needs the
        `provider` field rather than the response text."""
        return self._entries.get(prompt_key(self.model, prompt, self.seed))

    def put(self, prompt, response, meta=None):
        """Record one answer. Flushed immediately so a crash loses at most the in-flight call."""
        rec = {"key": prompt_key(self.model, prompt, self.seed), "model": self.model,
               "seed": self.seed, "response": response, **(meta or {})}
        with self._lock:
            self._entries[rec["key"]] = rec
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def stats(self, prompts):
        """(hits, misses) for a prompt list, so a run can report what it will actually do."""
        hits = sum(1 for p in prompts if prompt_key(self.model, p, self.seed) in self._entries)
        return hits, len(prompts) - hits


def write_atomic(path, text):
    """Write a result file so an interrupted run never leaves a half-written CSV behind.

    A truncated predictions file is worse than a missing one: the missing one is obvious, while the
    truncated one scores silently on fewer cells.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _self_test():
    import shutil
    tmp = Path(tempfile.mkdtemp())
    try:
        c = ResponseCache(tmp / "c.jsonl", "m1", seed=1)
        assert c.get("p") is None
        c.put("p", "answer")
        assert c.get("p") == "answer"

        # Survives a reopen.
        assert ResponseCache(tmp / "c.jsonl", "m1", seed=1).get("p") == "answer"

        # A different SEED is a different request, not a cache hit. Without this, modes whose
        # prompt does not depend on the seed (M1, M3) would serve all three repeats from one call.
        assert ResponseCache(tmp / "c.jsonl", "m1", seed=2).get("p") is None

        # A different model is a different result, not a cache hit.
        assert ResponseCache(tmp / "c.jsonl", "m2", seed=1).get("p") is None

        # Editing the prompt invalidates exactly that prompt: the property that keeps an ablation
        # from serving stale answers after a template change.
        assert c.get("p ") is None

        # A truncated final line loses only its own entry.
        with open(tmp / "c.jsonl", "a") as fh:
            fh.write('{"key": "broken", "resp')
        assert ResponseCache(tmp / "c.jsonl", "m1", seed=1).get("p") == "answer"

        # A different SEED is a different request, not a cache hit. Without this, modes whose
        # prompt does not depend on the seed (M1, M3) would serve all three repeats from one call.
        assert ResponseCache(tmp / "c.jsonl", "m1", seed=2).get("p") is None

        write_atomic(tmp / "out.csv", "a,b\n1,2\n")
        assert (tmp / "out.csv").read_text() == "a,b\n1,2\n"
        assert not list(tmp.glob("*.tmp")), "atomic write left a temp file behind"
        print("runner_cache self-test: OK")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    _self_test()
