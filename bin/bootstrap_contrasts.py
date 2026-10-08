#!/usr/bin/env python
"""Paired bootstrap intervals for the differences in balanced accuracy that the Results state.

Every method annotated the same cells of a dataset: a fixed number per cell type, drawn at random
within the type (``bin/subset.py``). The bootstrap draws cells the same way, with replacement
within each true type, and recomputes balanced accuracy for every run on the same drawn cells, so a
difference between two runs keeps its pairing. A comparison between the best runs of two groups
picks the best again in every replicate, so the run that won on the observed cells is not favoured.

The interval covers which cells were sampled, not a rerun of a stochastic model. In the
prompt-context ablation a rerun with another seed moved balanced accuracy by a median standard
deviation of 0.009, so a difference is called only when its interval excludes zero and it is at
least NOISE_FLOOR. With four datasets, the text counts the datasets on which a difference is
called; nothing is tested across datasets.

Checked 2026-10-07: the per-cell outcomes reproduce the snapshot's balanced accuracy to 5e-5 in
every run, and the 60 runs scored at both subset sizes moved by no more than the bootstrap
predicts once the cells the two samples share are accounted for.

Reads the per-cell outcomes ``bin/per_class_scores.py`` writes (``--cells-out``). Writes every
comparison to a CSV and the comparisons the text states to a supplementary LaTeX table. With
``--lists-cells`` it compares instead each condition of the imperfect-list study with the exact
list (``list_contrasts``).

Usage:
    python bin/bootstrap_contrasts.py --input cache/result/aggregated_by_dataset_model_<v>.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manuscript_figures import DATASET_ORDER, _run_label, dataset_label  # noqa: E402
from plot_model_comparison import (  # noqa: E402
    EXCLUDED_BASES, categorize_model, derive_mode, model_base)

DEFAULT_CELLS = "cache/cell_hits_1000_stratified.csv"
DEFAULT_OUT = "cache/bootstrap_contrasts_1000_stratified.csv"
REPLICATES = 10_000
SEED = 20261007
# The smallest difference called, above the run-to-run noise of a stochastic model (see above).
NOISE_FLOOR = 0.03
# Runs compared with the top run of Figure 2A on the mean over the four datasets.
TOP_RUNS = 5


def _group(family, mode):
    return lambda m: categorize_model(m) == family and derive_mode(m) == mode


# The comparisons the text states, as (key, label, group A, group B): the best run of A minus the
# best run of B, per dataset, both in the same setting.
CONTRASTS = [
    ("llm_c-scllm_c", "Constrained LLMs vs constrained scLLMs",
     _group("LLM", "unblind"), _group("scLLM", "unblind")),
    ("llm_c-baseline_c", "Constrained LLMs vs constrained baselines",
     _group("LLM", "unblind"), _group("classical", "unblind")),
    ("scllm_c-baseline_c", "Constrained scLLMs vs constrained baselines",
     _group("scLLM", "unblind"), _group("classical", "unblind")),
    ("llm_u-baseline_u", "Unconstrained LLMs vs unconstrained baselines",
     _group("LLM", "blind"), _group("classical", "blind")),
    ("scllm_u-baseline_u", "Unconstrained scLLMs vs unconstrained baselines",
     _group("scLLM", "blind"), _group("classical", "blind")),
    ("scllm_u-llm_u", "Unconstrained scLLMs vs unconstrained LLMs",
     _group("scLLM", "blind"), _group("LLM", "blind")),
]
FAMILY_NAME = {"LLM": "General-purpose LLMs", "scLLM": "scLLMs", "classical": "Baselines"}


def replicates(cells, n, rng, listed=None):
    """Balanced accuracy of every run of one dataset on ``n`` redraws of its cells, with
    replacement within each true type, as an ``n`` x runs frame, and the observed values.
    ``listed`` (true types x runs, booleans) restricts a run's balanced accuracy to some types;
    by default every type counts."""
    hits = cells.pivot(index="cell_id", columns="model", values="hit")
    if hits.isna().any().any():
        raise SystemExit(f"{cells['dataset'].iloc[0]}: the runs were not scored on the same cells")
    true = cells.drop_duplicates("cell_id").set_index("cell_id")["true_label"].reindex(hits.index)
    keep = (pd.DataFrame(True, index=sorted(true.unique()), columns=hits.columns)
            if listed is None else listed.reindex(columns=hits.columns).astype(bool))
    total = np.zeros((n, hits.shape[1]))
    for cell_type, index in true.groupby(true).groups.items():
        weights = rng.multinomial(len(index), np.full(len(index), 1 / len(index)), size=n)
        total += (weights @ hits.loc[index].to_numpy(float) / len(index)
                  * keep.loc[cell_type].to_numpy(float))
    per_type = hits.groupby(true).mean()
    observed = per_type.mean() if listed is None else per_type.where(keep).mean()
    return pd.DataFrame(total / keep.sum().to_numpy(float), columns=hits.columns), observed


def verdict(difference, low, high):
    """Ahead, behind or level. The floor is applied to the difference as printed (three
    decimals), so a reader sees what was judged."""
    if abs(round(difference, 3)) < NOISE_FLOOR or low <= 0 <= high:
        return "level"
    return "ahead" if difference > 0 else "behind"


def _row(contrast, label, dataset, run_a, run_b, observed, draws):
    low, high = np.percentile(draws, [2.5, 97.5])
    difference = observed[run_a] - observed[run_b]
    return {"contrast": contrast, "label": label, "dataset": dataset,
            "run_a": run_a, "balanced_accuracy_a": observed[run_a],
            "run_b": run_b, "balanced_accuracy_b": observed[run_b],
            "difference": difference, "ci_low": low, "ci_high": high,
            "verdict": verdict(difference, low, high)}


def contrasts(cells, n=REPLICATES, seed=SEED):
    """Every comparison of CONTRASTS per dataset, the effect of the constrained setting for each
    method, and the top of Figure 2A on the mean over the datasets."""
    rng = np.random.default_rng(seed)
    rows, draws, observed = [], {}, {}
    for dataset in [d for d in DATASET_ORDER if d in set(cells["dataset"])]:
        draws[dataset], observed[dataset] = replicates(cells[cells["dataset"] == dataset], n, rng)
    for dataset, rep in draws.items():
        obs = observed[dataset]
        for key, label, in_a, in_b in CONTRASTS:
            a = [m for m in rep.columns if in_a(m)]
            b = [m for m in rep.columns if in_b(m)]
            rows.append(_row(key, label, dataset, obs[a].idxmax(), obs[b].idxmax(), obs,
                             rep[a].max(axis=1) - rep[b].max(axis=1)))
        for model in rep.columns:
            unconstrained = model.replace("_unblind", "_blind")
            if model.endswith("_unblind") and unconstrained in rep.columns:
                rows.append(_row("list_effect", "Constrained minus unconstrained, same model",
                                 dataset, model, unconstrained, obs,
                                 rep[model] - rep[unconstrained]))
    # The mean over the datasets, each redrawn independently.
    common = sorted(set.intersection(*(set(rep.columns) for rep in draws.values())))
    mean_draws = sum(draws[d][common] for d in draws) / len(draws)
    mean_obs = (sum(observed[d][common] for d in draws) / len(draws)).sort_values(ascending=False)
    top = mean_obs.index[0]
    for other in mean_obs.index[1:TOP_RUNS + 1]:
        rows.append(_row("top", "Top of Figure 2A, mean over the four datasets", "all four",
                         top, other, mean_obs, mean_draws[top] - mean_draws[other]))
    return pd.DataFrame(rows)


def list_contrasts(cells, n=REPLICATES, seed=SEED):
    """Each condition of the imperfect-list study minus the exact list E, per arm and dataset.

    Both sides count only the types the condition still lists, so a condition that removes types
    is not charged for the cells it could not name. All conditions and seeds of an arm and dataset
    share each redraw, and the difference is averaged over the seeds. Gemini answers per cluster,
    so its cells are not independent and its intervals are too narrow.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for (arm, dataset), group in cells.groupby(["arm", "dataset"], sort=False):
        exact = dict(tuple(group[group["condition"] == "E"].groupby("seed")))
        runs, masks = [], []
        for (condition, s), run in group[group["condition"] != "E"].groupby(["condition", "seed"]):
            listed = run.groupby("true_label")["listed"].first()
            for name, frame in ((f"{condition}:{s}", run), (f"E|{condition}:{s}", exact[s])):
                runs.append(frame.assign(model=name))
                masks.append(listed.rename(name))
        draws, observed = replicates(pd.concat(runs), n, rng, pd.concat(masks, axis=1))
        for condition in sorted(set(group["condition"]) - {"E"}):
            a = [c for c in draws.columns if c.startswith(f"{condition}:")]
            b = [f"E|{c}" for c in a]
            means = pd.Series({f"{arm} {condition}": observed[a].mean(),
                               f"{arm} E": observed[b].mean()})
            rows.append(_row(f"list_{condition}", f"{condition} minus E, on the types listed",
                             dataset, f"{arm} {condition}", f"{arm} E", means,
                             (draws[a].to_numpy() - draws[b].to_numpy()).mean(axis=1)))
    return pd.DataFrame(rows)


def _signed(value):
    return f"{value:+.3f}".replace("-", "$-$")


def write_tex(table, out_path, n=REPLICATES):
    """The comparisons the text states, as a supplementary table; the list effect as counts."""
    lines = [
        "% GENERATED by bin/bootstrap_contrasts.py -- do not edit by hand.",
        "% Requires \\usepackage{booktabs}.",
        "",
        "\\begin{table}[!ht]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llrlrr}",
        "\\toprule",
        "Dataset & Best of the first group & Bal.\\ acc. & Best of the second group & Bal.\\ acc."
        " & Difference (95\\% interval) \\\\",
    ]

    def cells_of(r, dataset):
        diff = f"{_signed(r.difference)} ({_signed(r.ci_low)}, {_signed(r.ci_high)})"
        if r.verdict != "level":
            diff = f"\\textbf{{{diff}}}"
        return (f"{dataset} & {_run_label(r.run_a)} & {r.balanced_accuracy_a:.3f} & "
                f"{_run_label(r.run_b)} & {r.balanced_accuracy_b:.3f} & {diff} \\\\")

    for key, label, *_ in CONTRASTS:
        block = table[table["contrast"] == key]
        lines += ["\\midrule", f"\\multicolumn{{6}}{{l}}{{\\textbf{{{label}}}}} \\\\"]
        lines += [cells_of(r, dataset_label(r.dataset)) for r in block.itertuples()]
    top = table[table["contrast"] == "top"]
    lines += ["\\midrule", "\\multicolumn{6}{l}{\\textbf{Top of Figure~\\ref{fig:benchmark}A, "
              "mean over the four datasets}} \\\\"]
    lines += [cells_of(r, "all four") for r in top.itertuples()]
    lines += ["\\midrule", "\\multicolumn{6}{l}{\\textbf{Constrained minus unconstrained, same "
              "model}} \\\\"]
    effect = table[table["contrast"] == "list_effect"].copy()
    effect["family"] = effect["run_a"].map(categorize_model)
    for family in ("LLM", "scLLM", "classical"):
        group = effect[effect["family"] == family]
        counts = group["verdict"].value_counts()
        text = (f"ahead in {counts.get('ahead', 0)}, level in {counts.get('level', 0)}, behind in "
                f"{counts.get('behind', 0)} of {len(group)} model--dataset pairs "
                f"({_signed(group['difference'].min())} to {_signed(group['difference'].max())})")
        lines.append(f"\\multicolumn{{6}}{{l}}{{{FAMILY_NAME[family]}: {text}}} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\captionsetup{position=below}",
        "\\caption{\\textbf{Paired bootstrap intervals for the comparisons in the text.} Balanced "
        "accuracy at 1,000 cells. Each comparison takes the best run of each group on a dataset; "
        "C and U mark the constrained and unconstrained settings. The interval resamples the "
        "cells with replacement within each cell type, as they were sampled, and scores every run "
        f"on the same resampled cells ({n:,} replicates); the best run of a group is chosen again "
        "in every replicate. A difference is in bold when its interval excludes zero and it is at "
        f"least {NOISE_FLOOR:.2f}, the noise of rerunning a model; the others count as level "
        "(Methods). The last rows count how often the constrained setting changed the balanced "
        "accuracy of the same method, by the same rule.}",
        "\\label{supptab:contrasts}",
        "\\end{table}",
        "",
    ]
    out_path.write_text("\n".join(lines))
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="Results/aggregated_by_dataset_model.csv",
                   help="aggregate snapshot, to check the per-cell outcomes against")
    p.add_argument("--cells", default=DEFAULT_CELLS)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--tex", default="scLLMs_benchmark_paper/bioRxiv_template/"
                                    "supplementary_contrasts_generated.tex")
    p.add_argument("--replicates", type=int, default=REPLICATES)
    p.add_argument("--lists-cells", type=Path,
                   help="per-cell outcomes of the imperfect-list study (lists.py --score); "
                        "compares its conditions with the exact list instead")
    args = p.parse_args(argv)

    if args.lists_cells:
        table = list_contrasts(pd.read_csv(args.lists_cells, dtype={"cell_id": str}),
                               args.replicates)
        out = args.lists_cells.with_name("bootstrap_lists.csv")
        table.to_csv(out, index=False)
        for r in table.itertuples():
            print(f"{r.dataset:15s} {r.run_a:30s} {r.difference:+.3f} "
                  f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}] {r.verdict}")
        print(f"-> {out}")
        return 0

    cells = pd.read_csv(args.cells, dtype={"cell_id": str})
    keep = (~cells["model"].map(model_base).isin(EXCLUDED_BASES)
            & (cells["model"].map(categorize_model) != "simple"))
    cells = cells[keep]
    # The per-cell outcomes must reproduce the scores of the snapshot before anything is resampled.
    snapshot = pd.read_csv(args.input)
    snapshot = snapshot[snapshot["subset_size"] == 1000].set_index(["model", "dataset"])
    observed = (cells.groupby(["model", "dataset", "true_label"])["hit"].mean()
                .groupby(["model", "dataset"]).mean())
    gap = (observed - snapshot["balanced_accuracy"].reindex(observed.index)).abs()
    if gap.isna().any() or gap.max() > 1e-4:
        raise SystemExit(f"per-cell outcomes do not reproduce the snapshot: max gap {gap.max()}")
    print(f"per-cell outcomes reproduce the snapshot in {len(gap)} runs (max gap {gap.max():.1e})")

    table = contrasts(cells, args.replicates)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    write_tex(table, Path(args.tex), args.replicates)
    stated = table[table["contrast"] != "list_effect"]
    for r in stated.itertuples():
        print(f"{r.contrast:16s} {r.dataset:15s} {r.run_a:32s} {r.run_b:32s} "
              f"{r.difference:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}] {r.verdict}")
    print(f"-> {args.out}\n-> {args.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
