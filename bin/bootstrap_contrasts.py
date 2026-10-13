#!/usr/bin/env python
"""Paired bootstrap intervals for the differences in balanced accuracy that the Results state.

Every method annotated the same cells of a dataset, drawn at random within each cell type in
proportion to its share of the dataset, at least 20 per type (``bin/subset.py``). The bootstrap
draws cells the same way, with replacement within each true type, and recomputes balanced accuracy
for every run on the same drawn cells, so a difference between two runs keeps its pairing. A
comparison between the best runs of two groups picks the best again in every replicate, so the run
that won on the observed cells is not favoured.

The interval covers which cells were sampled, not a rerun of a stochastic model. In the
prompt-context ablation a rerun of a general-purpose LLM (the API arms, which sample) with another
seed moved balanced accuracy by a median standard deviation of 0.010, so a difference is called
only when its interval excludes zero and it is at least NOISE_FLOOR. With four datasets, the text
counts the datasets on which a difference is called; nothing is tested across datasets.

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
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_table import NUMBER_WORDS  # noqa: E402
from ablation_tables import LIST_CONDITION_NAME  # noqa: E402
from manuscript_figures import (ABLATION_ARMS, DATASET_ORDER, STUDY_ARM_MODEL,  # noqa: E402
                                _display, _run_label, dataset_label)
from plot_model_comparison import (  # noqa: E402
    EXCLUDED_BASES, categorize_model, derive_mode, model_base, model_tier)

DEFAULT_CELLS = "cache/cell_hits_1000_stratified.csv"
DEFAULT_OUT = "cache/bootstrap_contrasts_1000_stratified.csv"
REPLICATES = 10_000
SEED = 20261007
# The smallest difference called, above the run-to-run noise of a stochastic model (see above).
NOISE_FLOOR = 0.03
# Runs compared with the top run of Figure 2A on the mean over the four datasets: the two the
# text calls level with it.
TOP_RUNS = 2


def _group(family, mode):
    return lambda m: categorize_model(m) == family and derive_mode(m) == mode


# The comparisons the text states, as (key, label, group A, group B): the best run of A minus the
# best run of B, per dataset, both in the same setting.
CONTRASTS = [
    ("llm_c-scllm_c", "Constrained LLMs vs constrained scLLMs",
     _group("LLM", "unblind"), _group("scLLM", "unblind")),
    ("llm_c-baseline_c", "Constrained LLMs vs constrained baselines",
     _group("LLM", "unblind"), _group("classical", "unblind")),
    ("llm_u-baseline_u", "Unconstrained LLMs vs unconstrained baselines",
     _group("LLM", "blind"), _group("classical", "blind")),
    ("scllm_u-baseline_u", "Unconstrained scLLMs vs unconstrained baselines",
     _group("scLLM", "blind"), _group("classical", "blind")),
    ("scllm_u-llm_u", "Unconstrained scLLMs vs unconstrained LLMs",
     _group("scLLM", "blind"), _group("LLM", "blind")),
]
# Rows of the list effect, as (family, subcategory or None for all, name): the groups the text
# describes, with the scLLMs split into embedding-based and generative ones.
LIST_EFFECT_ROWS = [("LLM", None, "General-purpose LLMs"),
                    ("scLLM", "retrieval", "Embedding-based scLLMs"),
                    ("scLLM", "generative", "Generative scLLMs"),
                    ("classical", None, "Baselines")]


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
                rows.append(_row("list_effect", "Constrained minus unconstrained, same method",
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


# R2 on the benchmark's cells: the prompt-context ablation's arms at their default variant, the
# imperfect-list study, and the benchmark's unconstrained runs as the no-list reference of the
# methods that take no prompt.
ABLATION_VARIANT = {"A1_llm_percell": "expression_ranked", "A4_llm_percluster": "res1.0",
                    "A8_gemini_percluster": "res1.0", "A3_scllm_percell": "expression_ranked"}
# The ablation's effects, as (key, label, arms, condition minus condition).
ABLATION_EFFECTS = [
    ("abl_list", "+ list minus genes only", list(ABLATION_VARIANT), "M2", "M1"),
    ("abl_tissue", "+ tissue minus genes only", list(ABLATION_VARIANT), "M3", "M1"),
    ("abl_both", "+ tissue + list minus + list", list(ABLATION_VARIANT), "M4", "M2"),
    ("abl_order", "genes only minus shuffled genes", ["A1_llm_percell", "A3_scllm_percell"],
     "M1", "M6"),
]
LIST_GROUPS = [("LLM", "scLLM"), ("LLM", "classical"), ("scLLM", "classical")]


def r2_contrasts(ablation, lists, bench, n=REPLICATES, seed=SEED):
    """The comparisons R2 states, per dataset, on the same redraws of the cells:
    the ablation's effects and per cluster against per cell; each imperfect list against the
    exact list and against no list; and, at the exact list and at L4, the best method of each
    family against the best of another and C2S-Scale against GPT-OSS.

    Balanced accuracy over all true types, so a type missing from a list counts as missed.
    Conditions run with three seeds are averaged over them. Gemini, the per-cluster GPT-OSS,
    scType and SCSA label clusters, so their cells are not independent and their intervals are
    too narrow.
    """
    rng = np.random.default_rng(seed)
    ablation = ablation[ablation["variant"] == ablation["arm"].map(ABLATION_VARIANT)]
    arms = list(dict.fromkeys(lists["arm"]))
    prompted = [a for a in arms if a in ABLATION_VARIANT]
    none = {a: f"A|{a}|M1" if a in prompted else f"B|{STUDY_ARM_MODEL[a]}_blind" for a in arms}
    gemini_cell = "B|gemini-3.1-pro-preview_unblind"
    bench = bench[bench["model"].isin({r[2:] for r in none.values() if r.startswith("B|")}
                                      | {gemini_cell[2:]})]
    runs = pd.concat([
        ablation.assign(model="A|" + ablation["arm"] + "|" + ablation["mode"] + "|"
                        + ablation["seed"].astype(str)),
        lists.assign(model="L|" + lists["arm"] + "|" + lists["condition"] + "|"
                     + lists["seed"].astype(str)),
        bench.assign(model="B|" + bench["model"])])
    rows = []
    for dataset in [d for d in DATASET_ORDER if d in set(runs["dataset"])]:
        draws, observed = replicates(runs.loc[runs["dataset"] == dataset,
                                              ["dataset", "model", "cell_id", "true_label",
                                               "hit"]], n, rng)

        def run(name):
            """The draws and the observed value of a run, averaged over its seeds."""
            cols = [c for c in draws.columns if c == name or c.startswith(name + "|")]
            return draws[cols].mean(axis=1), observed[cols].mean()

        def add(key, label, a, b):
            (draws_a, obs_a), (draws_b, obs_b) = run(a), run(b)
            rows.append(_row(key, label, dataset, a, b, {a: obs_a, b: obs_b}, draws_a - draws_b))

        for key, label, effect_arms, cond_a, cond_b in ABLATION_EFFECTS:
            for arm in effect_arms:
                add(key, label, f"A|{arm}|{cond_a}", f"A|{arm}|{cond_b}")
        add("abl_unit", "per cluster minus per cell, + list", "A|A4_llm_percluster|M2",
            "A|A1_llm_percell|M2")
        add("abl_unit", "per cluster minus per cell, + list", "A|A8_gemini_percluster|M2",
            gemini_cell)
        for arm in arms:
            conditions = sorted(set(lists.loc[lists["arm"] == arm, "condition"]) - {"E"})
            for condition in conditions:
                add("list_condition", "condition minus exact list", f"L|{arm}|{condition}",
                    f"L|{arm}|E")
            for condition in ("E", "L4"):
                add("list_none", "condition minus no list", f"L|{arm}|{condition}", none[arm])
        for condition in ("E", "L4"):
            best = {}
            for arm in arms:
                best.setdefault(categorize_model(STUDY_ARM_MODEL[arm]), {})[arm] = run(
                    f"L|{arm}|{condition}")
            for fam_a, fam_b in LIST_GROUPS:
                a, b = best[fam_a], best[fam_b]
                obs_a = pd.Series({arm: o for arm, (_d, o) in a.items()})
                obs_b = pd.Series({arm: o for arm, (_d, o) in b.items()})
                run_a, run_b = f"L|{obs_a.idxmax()}|{condition}", f"L|{obs_b.idxmax()}|{condition}"
                rows.append(_row(f"list_{fam_a}-{fam_b}", "best of one family minus best of "
                                 "another", dataset, run_a, run_b,
                                 {run_a: obs_a.max(), run_b: obs_b.max()},
                                 pd.concat({k: d for k, (d, _o) in a.items()}, axis=1).max(axis=1)
                                 - pd.concat({k: d for k, (d, _o) in b.items()}, axis=1)
                                 .max(axis=1)))
            add("list_c2s-gptoss", "C2S-Scale minus GPT-OSS", f"L|A3_scllm_percell|{condition}",
                f"L|A1_llm_percell|{condition}")
    return pd.DataFrame(rows)


# R2's table: the comparisons in the order the text makes them, as (contrast key, heading).
R2_SECTIONS = [
    ("abl_list", "Ablation: + list minus genes only"),
    ("abl_tissue", "Ablation: + tissue minus genes only"),
    ("abl_both", "Ablation: + tissue + list minus + list"),
    ("abl_order", "Ablation: genes only minus shuffled genes"),
    ("abl_unit", "Ablation: per cluster minus per cell, + list"),
    ("list_condition", "List study: condition minus the exact list"),
    ("list_none", "List study: condition minus no list"),
    ("list_between", "List study: between methods"),
]
FAMILY_SHORT = {"LLM": "LLM", "scLLM": "scLLM", "classical": "baseline"}


def _r2_names(row):
    """(method, condition) as the table prints them."""
    if row.contrast.startswith("abl_"):
        arm = row.run_a.split("|")[1]
        if row.contrast == "abl_unit":
            return ABLATION_ARMS[arm][3].split(",")[0], ""
        return ABLATION_ARMS[arm][3], ""
    _kind, arm, condition = row.run_a.split("|")
    name = LIST_CONDITION_NAME[condition]
    if row.contrast == "list_c2s-gptoss":
        return "C2S-Scale 2B minus GPT-OSS-120B", name
    if row.contrast.startswith("list_") and "-" in row.contrast:
        fam_a, fam_b = row.contrast.removeprefix("list_").split("-")
        return f"best {FAMILY_SHORT[fam_a]} minus best {FAMILY_SHORT[fam_b]}", name
    return _display(STUDY_ARM_MODEL[arm]), name


def write_r2_tex(table, out_path, n=REPLICATES):
    """R2's comparisons as a longtable: one row per comparison, its difference per dataset, bold
    where it is called."""
    table = table.assign(section=table["contrast"].where(
        ~table["contrast"].str.contains("-"), "list_between"))
    header = "Method & Condition & HFRA & HIHA & Zeisel & human pancreas \\\\"
    lines = [
        "% GENERATED by bin/bootstrap_contrasts.py --r2 -- do not edit by hand.",
        "% Requires \\usepackage{longtable} and booktabs.", "",
        "\\clearpage", "\\scriptsize", "\\captionsetup{position=below}",
        "\\begin{longtable}{llrrrr}", "\\toprule", header, "\\midrule", "\\endfirsthead",
        "\\multicolumn{6}{l}{\\emph{Comparisons of R2, continued}}\\\\", "\\toprule", header,
        "\\midrule", "\\endhead", "\\midrule",
        "\\multicolumn{6}{r}{\\emph{continued on the next page}}\\\\", "\\endfoot", "\\bottomrule",
        "\\caption{\\textbf{Paired bootstrap intervals for the comparisons of R2.} Balanced accuracy "
        "over all cell types, a type missing from a list counting as missed, at 1,000 cells. Each "
        "value is the first run minus the second, averaged over the three seeds where a condition "
        "ran with three; it is in bold when its 95\\% interval excludes zero and it is at least "
        f"{NOISE_FLOOR:.2f}, as in Supplementary Table~\\ref{{supptab:contrasts}} ({n:,} "
        "replicates, the cells resampled within each cell type). The ablation's arms are those of "
        "Figure~\\ref{fig:context}A; GPT-OSS-120B and Gemini 3.1 Pro per cluster are compared with "
        "their own runs per cell, Gemini's from the benchmark. Without a list, the methods that "
        "take a prompt are the ablation's genes-only runs and the others the benchmark's "
        "unconstrained runs. A best of a family is chosen again in every replicate. Gemini 3.1 "
        "Pro, the per-cluster GPT-OSS-120B, scType and SCSA label clusters, so their cells are not "
        "independent and their intervals are too narrow. The intervals are in the supplementary "
        "data file \\texttt{supp\\_r2\\_contrasts.csv}.}\\label{supptab:r2_contrasts}\\\\",
        "\\endlastfoot",
    ]
    for key, heading in R2_SECTIONS:
        sub = table[table["section"] == key]
        if sub.empty:
            continue
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textbf{{{heading}}}}} \\\\")
        # One row per pair of runs; between methods, per comparison and condition, since the best
        # run of a family differs between datasets.
        keys = ([sub["contrast"], sub["run_a"].str.split("|").str[2]] if key == "list_between"
                else ["run_a", "run_b"])
        for _key, group in sub.groupby(keys, sort=False):
            method, condition = _r2_names(group.iloc[0])
            lines.append(" & ".join([method, condition] + _r2_cells(group)) + " \\\\")
    lines += ["\\end{longtable}", ""]
    out_path.write_text("\n".join(lines))
    return out_path


def _r2_cells(group):
    """The difference per dataset, in dataset order, bold where called."""
    by = group.set_index("dataset")
    cells = []
    for dataset in DATASET_ORDER:
        if dataset not in by.index:
            cells.append("--")
            continue
        r = by.loc[dataset]
        value = _signed(r["difference"])
        cells.append(value if r["verdict"] == "level" else f"\\textbf{{{value}}}")
    return cells


def _signed(value):
    return f"{value:+.3f}".replace("-", "$-$")


def _accuracy(value):
    """Balanced accuracy as the results tables print it: kept to four decimals, as the snapshot
    stores it, then rounded half up to three."""
    return str(Decimal(str(round(value, 4))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


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
        return (f"{dataset} & {_run_label(r.run_a)} & {_accuracy(r.balanced_accuracy_a)} & "
                f"{_run_label(r.run_b)} & {_accuracy(r.balanced_accuracy_b)} & {diff} \\\\")

    for key, label, *_ in CONTRASTS:
        block = table[table["contrast"] == key]
        lines += ["\\midrule", f"\\multicolumn{{6}}{{l}}{{\\textbf{{{label}}}}} \\\\"]
        lines += [cells_of(r, dataset_label(r.dataset)) for r in block.itertuples()]
    top = table[table["contrast"] == "top"]
    lines += ["\\midrule", "\\multicolumn{6}{l}{\\textbf{Top of Figure~\\ref{fig:benchmark}A, "
              "mean over the four datasets}} \\\\"]
    lines += [cells_of(r, "all four") for r in top.itertuples()]
    lines += ["\\midrule", "\\multicolumn{6}{l}{\\textbf{Constrained minus unconstrained, same "
              "method}} \\\\"]
    effect = table[table["contrast"] == "list_effect"].copy()
    effect["family"] = effect["run_a"].map(categorize_model)
    effect["tier"] = effect["run_a"].map(model_tier)
    for family, tier, name in LIST_EFFECT_ROWS:
        group = effect[(effect["family"] == family) & ((effect["tier"] == tier) if tier else True)]
        counts = group["verdict"].value_counts()
        text = (f"ahead in {counts.get('ahead', 0)}, level in {counts.get('level', 0)} and behind "
                f"in {counts.get('behind', 0)} of {len(group)} pairs of method and dataset "
                f"({_signed(group['difference'].min())} to {_signed(group['difference'].max())})")
        lines.append(f"\\multicolumn{{6}}{{l}}{{{name} {text}}} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\captionsetup{position=below}",
        "\\caption{\\textbf{Paired bootstrap intervals for the comparisons in the text.} The "
        f"first {NUMBER_WORDS[len(CONTRASTS)]} blocks compare two groups of runs, named in their "
        "headings, on each dataset. Dataset names the test set, or all four for the mean over the "
        "four datasets. The best run of each group on a dataset is named with its balanced "
        "accuracy at 1,000 cells (Bal.\\ acc.), and C and U mark the constrained and "
        "unconstrained settings. Difference gives the first minus the second, with its 95\\% "
        "interval in parentheses. The interval resamples the cells with replacement within each "
        "cell type, as they were sampled, and scores every run on the same resampled cells "
        f"({n:,} replicates). The best run of each group is chosen again in every replicate. The "
        "block headed with the top of Figure~\\ref{fig:benchmark}A compares the run with the "
        f"highest mean over the four datasets with the next {NUMBER_WORDS[TOP_RUNS]}, on that "
        "mean, each dataset resampled independently. A difference is in bold when its interval "
        f"excludes zero and it is at least {NOISE_FLOOR:.2f}. The last block counts, for the "
        "general-purpose LLMs, the embedding-based and the generative scLLMs and the baselines, "
        "the pairs of method and dataset in which the constrained setting was ahead of the "
        "unconstrained one by this rule, level or behind, with the range of the differences in "
        "parentheses.}",
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
    p.add_argument("--r2", type=Path, metavar="STUDY_DIR",
                   help="the context_ablation directory: the comparisons R2 states, from both "
                        "studies' per-cell outcomes and the benchmark's (--cells)")
    args = p.parse_args(argv)

    if args.r2:
        read = lambda path: pd.read_csv(path, dtype={"cell_id": str})  # noqa: E731
        table = r2_contrasts(read(args.r2 / "results" / "cell_outcomes.csv"),
                             read(args.r2 / "results_lists" / "cell_outcomes.csv"),
                             read(args.cells), args.replicates)
        out = args.r2 / "results_lists" / "bootstrap_r2.csv"
        table.to_csv(out, index=False)
        tex = Path(args.tex).with_name("supplementary_r2_contrasts_generated.tex")
        write_r2_tex(table, tex, args.replicates)
        table.to_csv(tex.parent / "tables" / "supp_r2_contrasts.csv", index=False)
        print(f"{len(table)} comparisons -> {out}\n-> {tex}")
        return 0

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
