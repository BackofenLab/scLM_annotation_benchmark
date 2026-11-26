#!/usr/bin/env python
"""Response cache of the runners, kept as append-only JSONL.

Every answer is stored under ``sha256(model, prompt, seed)``, so an interrupted run resumes with
the prompts that have no answer yet, and a changed prompt is asked again. The runners keep one
file per arm and dataset in ``<study tree>/cache/``. ``write_atomic`` writes result files.

    python src/runner_cache.py    # self-test
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path


def prompt_key(model: str, prompt: str, seed=None) -> str:
    """sha256 of the model id, the prompt and the seed (left out if None)."""
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(prompt.encode("utf-8"))
    if seed is not None:
        digest.update(b"\x00")
        digest.update(str(seed).encode("utf-8"))
    return digest.hexdigest()


class ResponseCache:
    """Append-only JSONL cache of the response to each prompt; ``put`` is thread-safe.

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
        """Load the stored entries, skipping unreadable lines such as a last line cut off by a
        killed run."""
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
        """The whole cached record of a prompt, with its metadata, or ``None``."""
        return self._entries.get(prompt_key(self.model, prompt, self.seed))

    def put(self, prompt, response, meta=None):
        """Append one answer and sync it to disk."""
        rec = {"key": prompt_key(self.model, prompt, self.seed), "model": self.model,
               "seed": self.seed, "response": response, **(meta or {})}
        with self._lock:
            self._entries[rec["key"]] = rec
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def stats(self, prompts):
        """``(hits, misses)`` for a list of prompts."""
        hits = sum(1 for p in prompts if prompt_key(self.model, p, self.seed) in self._entries)
        return hits, len(prompts) - hits


def write_atomic(path, text):
    """Write ``text`` to ``path`` through a temporary file, so an interrupted run leaves no
    partial file."""
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

        # Another seed is another request.
        assert ResponseCache(tmp / "c.jsonl", "m1", seed=2).get("p") is None

        # Another model is another request.
        assert ResponseCache(tmp / "c.jsonl", "m2", seed=1).get("p") is None

        # A changed prompt is not a hit.
        assert c.get("p ") is None

        # A truncated final line loses only its own entry.
        with open(tmp / "c.jsonl", "a") as fh:
            fh.write('{"key": "broken", "resp')
        assert ResponseCache(tmp / "c.jsonl", "m1", seed=1).get("p") == "answer"

        # Another seed is another request.
        assert ResponseCache(tmp / "c.jsonl", "m1", seed=2).get("p") is None

        write_atomic(tmp / "out.csv", "a,b\n1,2\n")
        assert (tmp / "out.csv").read_text() == "a,b\n1,2\n"
        assert not list(tmp.glob("*.tmp")), "atomic write left a temp file behind"
        print("runner_cache self-test: OK")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    _self_test()
