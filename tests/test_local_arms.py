"""Checks for the local-weight arms of the context ablation (context_ablation/src/run_local.py):
the C2S template keeps the benchmark wrapper's sorted candidate list, and greedy arms generate a
prompt once whatever the seed, in a cache of their own.

Run with pytest, or directly in the evaluation environment: python tests/test_local_arms.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "context_ablation" / "src")]
import prompts  # noqa: E402
import run_local  # noqa: E402

LABELS = ["b cell", "T cell", "acinar cell", "CD4-positive, alpha-beta T cell"]


def test_c2s_keeps_the_wrappers_list_order():
    rng = np.random.default_rng(1)
    state = rng.bit_generator.state
    assert prompts.ordered_candidates("c2s", LABELS, rng) == sorted(LABELS)
    # No draw, so whatever else this cell draws (M6's gene shuffle) is unchanged.
    assert rng.bit_generator.state == state
    assert (prompts.ordered_candidates("instruction", LABELS, np.random.default_rng(1))
            == prompts.candidate_list(LABELS, np.random.default_rng(1)))


def _run(decoding):
    """Two prompt files with the same prompts at two seeds; returns the generate() calls."""
    arm = {"kind": "local", "model": "m"}
    if decoding:
        arm["decoding"] = decoding
    cfg = {"arms": {"A3_scllm_percell": arm}}
    calls = []

    def fake_generate(model, tokenizer, device, todo, seed, greedy=False, **kw):
        calls.append({"n": len(todo), "seed": seed, "greedy": greedy, **kw})
        for prompt in todo:
            yield prompt, "B cell."

    original = run_local.generate
    run_local.generate = fake_generate
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            outputs = []
            for seed in (1234, 5678):
                prompt_file = tmp / f"HFRA__A3_scllm_percell__M1__expression_ranked__seed{seed}.csv"
                pd.DataFrame({"cell_id": ["c1", "c2"], "true_labels": ["t", "t"],
                              "prompt": ["p1", "p2"]}).to_csv(prompt_file, index=False)
                outputs.append(run_local.run_file(prompt_file, cfg, tmp / "out", tmp / "cache",
                                                  lambda: (None, None, "cpu")))
            caches = sorted(p.name for p in (tmp / "cache").iterdir())
            labels = [pd.read_csv(o)["predicted_labels"].tolist() for o in outputs]
    finally:
        run_local.generate = original
    return calls, caches, labels


def test_greedy_answers_are_generated_once_across_seeds():
    calls, caches, labels = _run("greedy")
    assert calls == [{"n": 2, "seed": 1234, "greedy": True}]
    assert caches == ["A3_scllm_percell__HFRA__greedy.jsonl"]
    assert labels == [["B cell", "B cell"]] * 2


def test_sampled_answers_stay_keyed_by_seed():
    calls, caches, _labels = _run(None)
    assert [(c["seed"], c["greedy"]) for c in calls] == [(1234, False), (5678, False)]
    assert caches == ["A3_scllm_percell__HFRA.jsonl"]


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
