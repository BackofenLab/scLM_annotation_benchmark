#!/usr/bin/env python
"""Generate the supplementary tables and the prompts note for the prompt-context ablation and the
imperfect-list study (Results R2).

Five typeset tables, the prompts note and two reusable CSVs:

* ``tables/supp_ablation_results.csv`` -- one row per (arm, gene variant, condition, dataset),
  every metric the ablation scores, with the mean and the standard deviation over the three seeds.
  This is the artifact a reader reuses and the one converted for submission.
* the design table -- which arm ran which model, where, at which granularity, under which prompt
  template, and over which conditions and gene variants. The grid is READ FROM THE DATA, not
  transcribed, so a condition that was configured but never run cannot appear here as though it
  had been.
* the condition table -- what each of the five prompt conditions supplies, and which two of them
  reproduce the benchmark's own two regimes.
* the results table -- every arm x variant x condition on the metrics the section argues from.
* ``tables/supp_list_results.csv`` and two tables of the imperfect-list study -- its conditions,
  read from context_ablation/config.yml, and its results per method and condition.
* the prompts note -- every prompt of the benchmark and the two studies, imported from
  context_ablation/src/prompts.py or built by the benchmark wrappers' own functions.

Nothing is hand-written, so these cannot drift from Figure 3 (style Rule 6).

Usage:
    python bin/ablation_tables.py
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import io
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manuscript_figures import (DATASET_ORDER, DEFAULT_ABLATION_CSV,  # noqa: E402
                                DEFAULT_LIST_RUNS_CSV, STUDY_ARM_MODEL, _display,
                                _list_study_arms, dataset_label)

# Per-arm facts that are NOT in the results CSV: the checkpoint or endpoint, where it ran, and
# whose prompt template it used. Everything else in the design table is derived from the data.
# Sourced from context_ablation/config.yml; if that changes, change this.
ARM_META = {
    "A1_llm_percell": ("GPT-OSS-120B", "openai/gpt-oss-120b", "hosted API", "cell",
                       "benchmark instruction"),
    "A2_base_percell": ("Gemma-2-2B", "google/gemma-2-2b", "local GPU", "cell",
                        "benchmark instruction"),
    "A3_scllm_percell": ("C2S-Scale 2B", "vandijklab/C2S-Scale-Gemma-2-2B", "local GPU", "cell",
                         "C2S-Scale's own"),
    "A4_llm_percluster": ("GPT-OSS-120B", "openai/gpt-oss-120b", "hosted API", "cluster",
                          "cluster marker genes"),
    "A8_gemini_percluster": ("Gemini 3.1 Pro", "google/gemini-3.1-pro-preview", "hosted API",
                             "cluster", "cluster marker genes"),
}
ARM_ORDER = ["A1_llm_percell", "A2_base_percell", "A3_scllm_percell",
             "A4_llm_percluster", "A8_gemini_percluster"]

# Condition naming and order for the MANUSCRIPT, 2026-08-15.
#
# The study's own code, its cached artifacts and context_ablation.csv use M1-M6, assigned in the
# order the conditions were designed. Those names are NOT renamed -- the manuscript relabels at the
# last step before writing, the same convention manuscript_figures.py uses for blind/unblind ->
# unconstrained/constrained, so no code or cached result has to move.
#
# The manuscript order is Figure 3A's x axis, which is the information ladder from least supplied
# to most, not the order the conditions were numbered in. C1..C5 are therefore assigned along that
# ladder, so a reader moves left to right across the figure and top to bottom down the table.
CONDITION_ORDER = ["M6", "M1", "M3", "M2", "M4"]
CONDITION_LABEL = {"M6": "C1", "M1": "C2", "M3": "C3", "M2": "C4", "M4": "C5"}


def _cond(mode):
    """Manuscript label for an internal condition name. Unknown names pass through unchanged."""
    return CONDITION_LABEL.get(mode, mode)


def _cond_key(mode):
    """Sort key placing conditions in Figure 3A's order."""
    return CONDITION_ORDER.index(mode) if mode in CONDITION_ORDER else len(CONDITION_ORDER)


# What each condition supplies, and which benchmark regime it reproduces. Rows follow
# CONDITION_ORDER. The prompt templates for the two reproducing conditions are copied verbatim from
# the benchmark's model configurations, so those two rows are what license comparing any number in
# R3 to the main table.
CONDITIONS = [
    ("C1", "yes", "shuffled", "--", "--", "--"),
    ("C2", "yes", "expression-ranked", "--", "--", "unconstrained"),
    ("C3", "yes", "expression-ranked", "yes", "--", "--"),
    ("C4", "yes", "expression-ranked", "--", "yes", "constrained"),
    ("C5", "yes", "expression-ranked", "yes", "yes", "--"),
]

VARIANT_LABEL = {"expression_ranked": "all genes", "hvg": "5,000 variable genes",
                 "res0.5": "resolution 0.5", "res1.0": "resolution 1.0"}

# Metrics the typeset results table carries, in the order R3 argues from them. The CSV keeps every
# column the scorer emits; these are the ones a reader checks against the figure.
#
# PLAIN ACCURACY DROPPED 2026-08-15. Over the 36 rows of this table it correlates with balanced
# accuracy at Spearman 0.985 and Pearson 0.995, so it reorders almost nothing, while Lin correlates
# at 0.953 and does carry independent information. It remains in supp_ablation_results.csv, so
# nothing is lost.
TEX_METRICS = [
    ("balanced_accuracy", "Bal.\\ acc."),
    ("agreement_score", "Agree."),
    ("lin_avg_similarity", "Lin"),
    ("non_cl_ratio", "non-CL"),
    ("frac_empty", "Empty"),
    ("frac_over_8_words", "$>$8 words"),
]


# The imperfect-list study: its conditions as Figure 3B names them, in the study's order. The
# conditions themselves are read from context_ablation/config.yml, never transcribed.
LIST_CONDITION_NAME = {"E": "exact list", "EU": "exact list + uncertain", "L1": "A1",
                       "L2-2": "A2 (2 types)", "L2-5": "A2", "L3": "A3", "L4": "A1 + A2 + A3"}
FRACTION = {0.5: "half", 0.25: "a quarter"}
PROMPTED_ARMS = {"A1_llm_percell", "A3_scllm_percell", "A8_gemini_percluster"}
# Columns of the list-study table: balanced accuracy over all types and over the types still listed
# (where types are removed), the coarser answers mixed levels draw, and "uncertain" where offered.
LIST_METRICS = [("balanced_accuracy", "Bal.\\ acc."), ("agreement", "Agree."), ("lin", "Lin"),
                ("balanced_accuracy_listed", "Listed"), ("all_coarser", "Coarser"),
                ("abstained_removed", "Unc.\\ missing"), ("abstained_listed", "Unc.\\ listed")]


def _tex(text):
    return str(text).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def _fmt(value, digits=3):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def load(path=DEFAULT_ABLATION_CSV):
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"No ablation results at {path}. Run the study first: see context_ablation/README.md.")
    return pd.read_csv(path)


def per_dataset_frame(raw):
    """One row per (arm, variant, condition, dataset): mean and SD over the three seeds."""
    metrics = [c for c in raw.columns
               if pd.api.types.is_numeric_dtype(raw[c]) and c not in ("seed",)]
    grouped = raw.groupby(["arm", "variant", "mode", "dataset"])[metrics]
    out = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_sd"))
    out["n_seeds"] = grouped.size()
    out = out.reset_index()
    # Carry BOTH names. `mode` is what the study's code and cached artifacts use; `condition` is
    # what the manuscript prints. Without the second column a reader moving from Supplementary
    # Table S8 to this file has no way to find the row they were looking at.
    out.insert(out.columns.get_loc("mode") + 1, "condition", out["mode"].map(_cond))
    return out.sort_values(["arm", "variant", out["mode"].map(_cond_key).name or "mode"],
                           key=lambda s: s.map(_cond_key) if s.name == "mode" else s)


def overall_frame(raw):
    """One row per (arm, variant, condition): mean over datasets of the seed means, and the SD
    ACROSS DATASETS of those means.

    The spread reported is the between-dataset one, not the between-seed one, and deliberately so:
    the seed-to-seed SD within a dataset is roughly 27x smaller, so quoting it would suggest a
    precision the four-dataset design does not have. n = 4 datasets, matching the paper.
    """
    per_ds = (raw.groupby(["arm", "variant", "mode", "dataset"])
              .mean(numeric_only=True).reset_index())
    grouped = per_ds.groupby(["arm", "variant", "mode"])
    metrics = [c for c, _h in TEX_METRICS]
    out = grouped[metrics].mean().add_suffix("_mean").join(
        grouped[metrics].std(ddof=1).add_suffix("_sd"))
    out["n_datasets"] = grouped.size()
    return out.reset_index()


def _condition_ranges(labels):
    """The conditions as a list, a run of three or more consecutive ones as a range:
    C1, C2, C3, C4, C5 -> C1--C5. Labels that are not C<number> are listed as they are."""
    groups = []
    for label in labels:
        n = int(label[1:]) if re.fullmatch(r"C\d+", label) else None
        if n is not None and groups and groups[-1][2] == n - 1:
            groups[-1][2] = n
        else:
            groups.append([label, n, n])
    parts = []
    for label, first, last in groups:
        if first is None or last == first:
            parts.append(label)
        elif last == first + 1:
            parts += [label, f"C{last}"]
        else:
            parts.append(f"{label}--C{last}")
    return ", ".join(parts)


def _stacked(items):
    """Several entries one under another in one table cell."""
    if len(items) == 1:
        return items[0]
    return "\\begin{tabular}[t]{@{}l@{}}" + "\\\\".join(items) + "\\end{tabular}"


def design_rows(raw):
    """The realized grid, read from the data rather than from the configuration.

    An arm configured but never run therefore cannot appear, which is the point: the study's
    config.yml also defines a GPT-OSS-20B per-cell arm that was never executed, and a design table
    transcribed from the config would have claimed it.
    """
    rows = []
    for arm in ARM_ORDER:
        sub = raw[raw["arm"] == arm]
        if sub.empty:
            continue
        name, ident, host, gran, template = ARM_META[arm]
        # Figure 3A's order, relabelled C1..C5. Superseded the earlier numeric M1..M6 sort on
        # 2026-08-15: with the conditions renamed along the information ladder, C-order and reading
        # order are the same thing, so the table and the figure no longer disagree.
        conditions = [_cond(m) for m in sorted(set(sub["mode"]), key=_cond_key)]
        variants = [VARIANT_LABEL.get(v, v) for v in
                    sorted(set(sub["variant"]), key=lambda v: list(VARIANT_LABEL).index(v))]
        rows.append((name, ident, host, gran, template,
                     _condition_ranges(conditions), variants,
                     int(sub["seed"].nunique()), int(sub["dataset"].nunique())))
    return rows


def write_tex(raw, out_path):
    lines = [
        "% GENERATED by bin/ablation_tables.py -- do not edit by hand.",
        "% Regenerate after any change to context_ablation/results/context_ablation.csv so these",
        "% tables cannot drift from Figure 3. Requires \\usepackage{longtable} and booktabs.",
        "",
    ]

    # ---- design ---------------------------------------------------------
    lines += [
        "\\begin{table}[!ht]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lllllll}", "\\toprule",
        "Model & Identifier & Run on & Unit & Prompt template & Conditions & Gene variants \\\\",
        "\\midrule",
    ]
    # A little space between the arms, whose gene variants stack over two lines.
    for i, (name, ident, host, gran, template, conds, variants, _ns, _nd) in enumerate(
            design_rows(raw)):
        lines += ([] if i == 0 else ["\\addlinespace[3pt]"]) + [
            " & ".join([_tex(name), f"\\texttt{{{_tex(ident)}}}", host, gran, _tex(template),
                        _tex(conds), _stacked([_tex(v) for v in variants])]) + " \\\\"]
    lines += [
        "\\bottomrule", "\\end{tabular}",
        "\\captionsetup{position=below}",
        "\\caption{\\textbf{Arms of the prompt-context ablation.} "
        "Each row is one arm. \\emph{Unit} is what one model call annotated, a single cell or a "
        "whole cluster. \\emph{Prompt template} names whose wording the arm used: the benchmark's "
        "own template for the general-purpose models, C2S-Scale's for the model it was released "
        "with, and a marker-gene template for the cluster-level arms. \\emph{Conditions} are the "
        "prompt conditions of Supplementary Table~\\ref{supptab:ablation_conditions}. "
        "\\emph{Gene variants} are the gene vocabularies for the per-cell arms and the clustering "
        "resolutions for the per-cluster arms; the per-cell arms ran the 5,000 variable genes in "
        "conditions C2 and C5 only. Every run was repeated with three seeds on each of the four "
        "datasets, at 1,000 cells per dataset.}",
        "\\label{supptab:ablation_design}", "\\end{table}", "",
    ]

    # ---- conditions -----------------------------------------------------
    lines += [
        "\\begin{table}[!ht]", "\\centering", "\\footnotesize",
        "\\begin{tabular}{llllll}", "\\toprule",
        "Condition & Gene list & Gene order & Tissue and species & Expected cell types "
        "& Benchmark regime \\\\",
        "\\midrule",
    ]
    for row in CONDITIONS:
        lines.append(" & ".join(_tex(c) for c in row) + " \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}",
        "\\captionsetup{position=below}",
        "\\caption{\\textbf{What each prompt condition supplies.} "
        "Conditions are in the order of the x axes of Figure~\\ref{fig:context}A (C2--C5) and "
        "Supplementary Figure~\\ref{suppfig:context_per_dataset}A--D. "
        "A dash marks information the condition withholds. \\emph{Gene order} is whether the gene "
        "list keeps its descending-expression ranking or is permuted while holding the same genes, "
        "the same count and the same surrounding prompt text. \\emph{Benchmark regime}: the "
        "setting of the main benchmark whose prompt wording the per-cell arms repeat in that "
        "condition (Supplementary Note~\\ref{suppnote:prompts}).}",
        "\\label{supptab:ablation_conditions}", "\\end{table}", "",
    ]

    # ---- results --------------------------------------------------------
    overall = overall_frame(raw)
    order = {a: i for i, a in enumerate(ARM_ORDER)}
    overall["_a"] = overall["arm"].map(order)
    overall["_c"] = overall["mode"].map(_cond_key)
    overall = overall.sort_values(["_a", "variant", "_c"])

    header = ["Model", "Unit", "Genes", "Cond."] + [h for _c, h in TEX_METRICS]
    # A longtable does not float: \\clearpage puts the floating tables above out first, so the
    # tables appear in the order of their numbers.
    lines += [
        "\\clearpage", "\\scriptsize",
        "\\captionsetup{position=below}",
        "\\begin{longtable}{llll" + "r" * len(TEX_METRICS) + "}",
        "\\toprule", " & ".join(header) + " \\\\", "\\midrule", "\\endfirsthead",
        "\\multicolumn{%d}{l}{\\emph{Ablation results, continued}}\\\\" % len(header),
        "\\toprule", " & ".join(header) + " \\\\", "\\midrule", "\\endhead",
        "\\midrule",
        "\\multicolumn{%d}{r}{\\emph{continued on the next page}}\\\\" % len(header),
        "\\endfoot", "\\bottomrule",
        "\\caption{\\textbf{Complete results of the prompt-context ablation.} "
        "One row per arm, gene variant and prompt condition, with conditions in the order of "
        "Supplementary Table~\\ref{supptab:ablation_conditions}. Each value is the mean over the "
        "four datasets of that dataset's mean over three seeds, at 1,000 cells per dataset, so "
        "$n = 4$ datasets and 12 runs per row. Bold marks the best value within each arm, over its "
        "conditions and gene variants: the highest balanced accuracy, agreement score and Lin "
        "similarity and the lowest non-CL share. \\emph{Agree.} is the agreement score and "
        "\\emph{Lin} the Lin similarity (Supplementary Table~\\ref{supptab:metrics}). "
        "\\emph{non-CL} is the share of answers that are, as written, not "
        "the name or identifier of a Cell Ontology term, \\emph{Empty} the share without an "
        "answer, and \\emph{$>$8 words} the share running past eight words. Conditions are "
        "defined in Supplementary "
        "Table~\\ref{supptab:ablation_conditions} and arms in Supplementary "
        "Table~\\ref{supptab:ablation_design}.}"
        "\\label{supptab:ablation_results}\\\\",
        "\\endlastfoot",
    ]
    # Best value per score column, bolded WITHIN EACH ARM rather than across the whole table.
    #
    # A single global maximum would compare a 2B base model against a frontier model against a
    # fine-tuned scLLM, at two different annotation units and two gene vocabularies, and mark one
    # cell the winner. That is not a comparison this table supports, and a reader would read the
    # bold as "best configuration" when it only means "largest number present".
    # Within an arm the model and the unit are fixed, so the bold answers the question the section
    # actually asks: which condition wins for this model. To bold globally instead, group on
    # nothing and take the column max over `overall`.
    #
    # Three columns are bolded, and non-CL runs the OTHER WAY: lowest is best there.
    #
    # bin/supplementary_tables.py bolds none of its diagnostics, and for a share like this that is
    # right -- marking the LARGEST fraction would flag the method most dependent on our manual
    # mapping as though it had won. Marking the SMALLEST is the opposite and is meaningful:
    # it is the run whose raw output was already closest to the ontology, which is the mechanism
    # R3.3 argues from. So the direction, not the column, is what made the earlier exclusion right.
    #
    # Empty and >8 words stay unbolded. They are at or near zero for every arm except the base
    # model, so a bold there would mark a tie between runs that never had the failure mode.
    BOLD_MAX = {"balanced_accuracy", "agreement_score", "lin_avg_similarity"}
    BOLD_MIN = {"non_cl_ratio"}
    best = {}
    for arm, sub in overall.groupby("arm"):
        for col in BOLD_MAX | BOLD_MIN:
            values = pd.to_numeric(sub[f"{col}_mean"], errors="coerce")
            if values.notna().any():
                # Compare on the PRINTED string, as the benchmark tables do: two runs differing in
                # the fifth decimal print identically, and bolding only one reads as a typo.
                best[(arm, col)] = _fmt(values.min() if col in BOLD_MIN else values.max())

    for _, r in overall.iterrows():
        name, _ident, _host, gran, _template = ARM_META[r["arm"]]
        cells = [_tex(name), gran, _tex(VARIANT_LABEL.get(r["variant"], r["variant"])),
                 _tex(_cond(r["mode"]))]
        for col, _h in TEX_METRICS:
            text = _fmt(r.get(f"{col}_mean"))
            if col in (BOLD_MAX | BOLD_MIN) and text != "--" \
                    and best.get((r["arm"], col)) == text:
                text = f"\\textbf{{{text}}}"
            cells.append(text)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\end{longtable}", ""]

    out_path.write_text("\n".join(lines))
    return out_path


def list_frame(runs):
    """One row per (method, condition, dataset) of the list study: mean and SD over the seeds."""
    metrics = [c for c in runs.columns if pd.api.types.is_numeric_dtype(runs[c]) and c != "seed"]
    grouped = runs.groupby(["arm", "condition", "dataset"])[metrics]
    out = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_sd"))
    out["n_seeds"] = grouped.size()
    out = out.reset_index()
    out.insert(1, "method", out["arm"].map(lambda a: _display(STUDY_ARM_MODEL[a])))
    out.insert(3, "condition_name", out["condition"].map(LIST_CONDITION_NAME))
    return out


def write_list_tex(runs, conditions):
    """The list study's conditions, read from its configuration, and its results per method."""
    lines = [
        "\\begin{table}[!ht]", "\\centering", "\\footnotesize",
        # The list column wraps: the combined condition names all three changes.
        "\\begin{tabular}{@{}l>{\\raggedright\\arraybackslash}p{6.4cm}ll@{}}", "\\toprule",
        "Condition & List & ``uncertain'' offered & Methods \\\\", "\\midrule",
    ]
    for name, spec in conditions.items():
        parts = []
        if spec.get("relatives"):
            share = FRACTION[spec["relatives"]]
            parts.append(f"a parent or a child added for {share} of the types")
        if spec.get("extra"):
            parts.append(f"{spec['extra']} absent types added")
        if spec.get("remove"):
            parts.append(f"{FRACTION[spec['remove']]} of the types removed")
        lines.append(" & ".join([_tex(LIST_CONDITION_NAME[name]),
                                 "; ".join(parts) or "the dataset's cell types",
                                 "yes" if spec.get("uncertain") else "--",
                                 str(runs.loc[runs["condition"] == name, "arm"].nunique())])
                     + " \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}", "\\captionsetup{position=below}",
        "\\caption{\\textbf{Conditions of the imperfect-list study.} Each condition replaces the "
        "list of expected cell types with a list drawn from it, once per dataset and seed. "
        "A relative of a type is a parent of it in the Cell Ontology, never a broad one, or a "
        "child of it. An absent type is a child of such a parent that is neither a type of the "
        "dataset nor above or below one. Added types are terms of the Cell Ontology's CELLxGENE "
        "subset without species-qualified names. Shares are rounded, and at least one type is "
        "removed; the cells of a removed type stay in the data. A2 (2 types) adds the first two of "
        "A2's five types, and A1 + A2 + A3 combines the relatives, absent types and removals of "
        "A1, A2 and A3. ``uncertain'' is offered to the methods that receive a prompt. "
        "\\emph{Methods} counts the methods that ran a condition: exact list + uncertain ran only "
        "for the three methods that receive a prompt, since for the others it equals the exact "
        "list.}",
        "\\label{supptab:list_conditions}", "\\end{table}", "",
    ]

    per_ds = runs.groupby(["arm", "condition", "dataset"]).mean(numeric_only=True)
    overall = per_ds.groupby(["arm", "condition"]).mean()
    header = ["Method", "Condition"] + [h for _c, h in LIST_METRICS]
    lines += [
        "\\clearpage", "\\scriptsize", "\\captionsetup{position=below}",
        "\\begin{longtable}{ll" + "r" * len(LIST_METRICS) + "}",
        "\\toprule", " & ".join(header) + " \\\\", "\\midrule", "\\endfirsthead",
        "\\multicolumn{%d}{l}{\\emph{Imperfect-list results, continued}}\\\\" % len(header),
        "\\toprule", " & ".join(header) + " \\\\", "\\midrule", "\\endhead",
        "\\midrule",
        "\\multicolumn{%d}{r}{\\emph{continued on the next page}}\\\\" % len(header),
        "\\endfoot", "\\bottomrule",
        "\\caption{\\textbf{Results of the imperfect-list study.} One row per method and "
        "condition, each value the mean over the four datasets of that dataset's mean over three "
        "seeds, at 1,000 cells per dataset. \\emph{Bal.\\ acc.} is balanced accuracy over all "
        "cell types, a type missing from the list counting as missed, and \\emph{Agree.} and "
        "\\emph{Lin} are the agreement score and Lin similarity over all cell types "
        "(Supplementary Table~\\ref{supptab:metrics}). \\emph{Listed} is balanced "
        "accuracy over the types still listed, given where a condition removes types. "
        "\\emph{Coarser} is the share of cells named with a coarser type that is not broad, the "
        "class `coarser, specific' of Figure~\\ref{fig:benchmark}H. "
        "\\emph{Unc.\\ missing} and \\emph{Unc.\\ listed} are the shares of the missing and of "
        "the listed types' cells answered ``uncertain'', given where it is offered. Shares are "
        "averaged over cell types, as balanced accuracy is. Gemini 3.1 Pro, scType and SCSA "
        "annotated clusters, the other methods single cells. Conditions are defined in "
        "Supplementary Table~\\ref{supptab:list_conditions}.}"
        "\\label{supptab:list_results}\\\\",
        "\\endlastfoot",
    ]
    for arm in _list_study_arms(runs):
        for name, spec in conditions.items():
            if (arm, name) not in overall.index:
                continue
            row = overall.loc[(arm, name)]
            shown = {"balanced_accuracy": True, "agreement": True, "lin": True,
                     "all_coarser": True,
                     "balanced_accuracy_listed": bool(spec.get("remove")),
                     "abstained_removed": arm in PROMPTED_ARMS and bool(spec.get("uncertain"))
                     and bool(spec.get("remove")),
                     "abstained_listed": arm in PROMPTED_ARMS and bool(spec.get("uncertain"))}
            lines.append(" & ".join([_tex(_display(STUDY_ARM_MODEL[arm])),
                                     _tex(LIST_CONDITION_NAME[name])]
                                    + [_fmt(row[c]) if shown[c] else "--"
                                       for c, _h in LIST_METRICS]) + " \\\\")
    lines += ["\\end{longtable}", ""]
    return lines


# --------------------------------------------------------------------------- #
# Prompt templates, never transcribed: a hand copy could silently stop matching what was sent.
# * The general-purpose LLM, C2S-Scale and per-cluster templates are IMPORTED from
#   context_ablation/src/prompts.py. With {context} empty, its LLM and C2S-Scale templates equal
#   the benchmark's (models/openrouter_LLM_*/src/model_cfg.yml and models/c2s_scale_*/src/
#   predict.py, checked byte for byte 2026-10-13), so one block serves the benchmark and the study.
# * InstructCell's prompt and LangCell's text for a candidate are built by the benchmark wrappers'
#   own functions, called with the slots as arguments.
# M5's two templates exist in prompts.py and are deliberately NOT listed: the mode was removed from
# the study on 2026-08-13 together with its prompts, predictions and scored runs.
# --------------------------------------------------------------------------- #
WRAPPERS = Path(__file__).resolve().parents[1] / "models"
DATASET_CONFIGS = Path(__file__).resolve().parents[1] / "datasets"
# (heading, where it was used, prompts.py constant). Conditions use the manuscript's labels (see
# CONDITION_LABEL): M1->C2, M2->C4, M3->C3, M4->C5, M6->C1.
# Headings name the template, not a model group: in the ablation the per-cell template also served
# the Gemma-2-2B base model, and Supplementary Table S9 names the arm behind each template.
SHARED_BLOCKS = [
    ("Per-cell template, unconstrained",
     "the benchmark by the general-purpose LLMs and in conditions C1, C2 and C3",
     "INSTRUCTION_NO_TYPES"),
    ("Per-cell template, constrained",
     "the benchmark by the general-purpose LLMs and in conditions C4 and C5",
     "INSTRUCTION_WITH_TYPES"),
    ("C2S-Scale's template, unconstrained",
     "the benchmark by C2S-Scale 2B and 27B and in conditions C1, C2 and C3", "C2S_NO_TYPES"),
    ("C2S-Scale's template, constrained",
     "the benchmark by C2S-Scale 2B and 27B and in conditions C4 and C5", "C2S_WITH_TYPES"),
]
CLUSTER_BLOCKS = [
    ("Per-cluster template, unconstrained", "conditions C2 and C3", "MARKER_NO_TYPES"),
    ("Per-cluster template, constrained", "conditions C4 and C5", "MARKER_WITH_TYPES"),
]


def _wrapper_function(relpath, name, **scope):
    """Function ``name`` of a benchmark wrapper, compiled on its own. The wrappers import their
    model stacks at the top, which this environment lacks, and the prompt code needs none of it."""
    path = WRAPPERS / relpath
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def benchmark_texts():
    """InstructCell's prompt in both settings and LangCell's text for a candidate, with and
    without a definition, as the wrappers build them."""
    scope = {"Dict": Dict, "List": List, "Optional": Optional,
             "logger": logging.getLogger("instructcell")}
    meta = {"species": "{species}", "tissue": "{tissue}"}
    unconstrained = _wrapper_function("instructcell_blind/src/predict.py", "build_prompt",
                                      **scope)(meta)
    constrained = _wrapper_function("instructcell_unblind/src/predict.py", "build_prompt",
                                    **scope)(meta, ["{unique_cell_types}"])
    text = _wrapper_function("langcell_blind/src/predict.py", "gettextfromname")
    with contextlib.redirect_stdout(io.StringIO()):  # the wrapper prints every text it builds
        defined = text("{name}", {"{name}": "id"}, {}, {"id": {"name": "{name}",
                                                              "def": "{definition}"}})
        bare = text("{candidate}", {}, {}, {})
    return unconstrained, constrained, defined.rstrip(), bare


def _dataset_configs():
    return {ds: yaml.safe_load((DATASET_CONFIGS / f"{ds}.yml").read_text()) for ds in DATASET_ORDER}


def _listed(items):
    return ", ".join(items[:-1]) + " and " + items[-1]


def _context_phrases(P, cfg):
    """The species-and-tissue phrase each dataset filled the ablation's {context} with, built by
    the study's own function from the configuration it read (context_ablation/src/sentences.py)."""
    phrases = []
    for ds, c in cfg.items():
        phrase = P.context_phrase({"tissue_species": True}, "instruction", tissue=c.get("tissue"),
                                  species=c.get("species") or c.get("organism"),
                                  override=c.get("context_phrase")).strip()
        phrases.append(f"\\texttt{{{phrase}}} for {dataset_label(ds)}")
    return _listed(phrases)


def _instructcell_slots(cfg):
    """The species and tissue InstructCell received, from the dataset configurations."""
    species = sorted({c["species"] for c in cfg.values()})
    return " or ".join(species), _listed([f"\\texttt{{{_tex(c['tissue'])}}} for "
                                          f"{dataset_label(ds)}" for ds, c in cfg.items()])


def write_prompts_tex(out_path):
    """Every prompt the benchmark and the two studies sent, verbatim, with the slots explained."""
    src = Path(__file__).resolve().parents[1] / "context_ablation" / "src"
    sys.path.insert(0, str(src))
    import prompts as P  # noqa: E402

    cfg = _dataset_configs()
    phrases = _context_phrases(P, cfg)
    lines = [
        "% GENERATED by bin/ablation_tables.py -- do not edit by hand.",
        "% The templates are imported from context_ablation/src/prompts.py or built by the",
        "% benchmark wrappers' own functions, so they cannot drift from what was sent.",
        "",
        # A \label inside an unnumbered \subsection* attaches to whatever counter was stepped last
        # and resolves to an EMPTY string, so "Supplementary Note~\ref{...}" renders with nothing
        # after it and hyperref warns about an empty link. A dedicated counter, stepped with
        # \refstepcounter so \ref picks it up, is what makes the cross-reference work. The counter
        # is defined in sections/supplementary_notes.tex, so the notes can be ordered by first
        # mention around this file.
        "\\refstepcounter{suppnote}",
        "\\paragraph{Supplementary Note \\thesuppnote: Prompt templates.}\\label{suppnote:prompts}",
        "",
        "The prompt templates of the benchmark, the prompt-context ablation and the "
        "imperfect-list study are reproduced below, with the substitution slots in braces. "
        "\\texttt{num\\_genes} caps the gene list at 200 per cell and 25 per cluster. "
        "\\texttt{cell\\_sentence} and \\texttt{marker\\_genes} carry the gene names, and "
        "\\texttt{organism} names the dataset's organism, Homo sapiens or Mus musculus. "
        "\\texttt{unique\\_cell\\_types} is the list of expected cell types. The per-cell "
        "templates receive it as a Python list, permuted independently for each cell, the per-cluster templates separated by commas and permuted "
        "for each cluster, and the templates of C2S-Scale and InstructCell separated by commas and "
"sorted. \\texttt{context} is empty in the benchmark. The prompt-context ablation fills it "
        f"in conditions C3 and C5 with the species and tissue, {phrases}, and with the tissue "
        "alone in C2S-Scale's template, which already names the organism (Supplementary "
        "Tables~\\ref{supptab:ablation_design} "
        "and~\\ref{supptab:ablation_conditions}). The benchmark sent each prompt to the "
        "general-purpose LLMs with an empty system message, and the ablation's arms through an "
        "API sent it as the only message.",
        "",
    ]

    def block(heading, used_in, body, note=""):
        return [f"\\paragraph{{{heading}.}} Used in {used_in}.{note}",
                "\\begin{quote}\\footnotesize\\ttfamily\\raggedright",
                _verbatimish(body.rstrip("\n")), "\\end{quote}", ""]

    instruct_u, instruct_c, langcell_text, langcell_bare = benchmark_texts()
    species, tissues = _instructcell_slots(cfg)
    bare = _verbatimish(langcell_bare).removesuffix(" \\\\")
    for heading, used_in, const in SHARED_BLOCKS:
        lines += block(heading, used_in, getattr(P, const))
    lines += block("InstructCell's template, unconstrained", "the benchmark", instruct_u,
                   f" \\texttt{{species}} is {species}, \\texttt{{tissue}} is {tissues}, and "
                   "InstructCell replaces \\texttt{input} with embeddings of the cell that its expression "
                   "encoder computes from the raw counts. The "
                   "template is the example prompt of the InstructCell repository without its "
                   "sequencing method, which the benchmark did not pass.")
    lines += block("InstructCell's template, constrained", "the benchmark", instruct_c,
                   " It adds the expected cell types in the benchmark's own wording, not in one "
                   "of the option phrasings InstructCell was trained with.")
    lines += block("Candidate texts of the embedding-based scLLMs",
                   "the benchmark and the imperfect-list study", langcell_text,
                   " These models compare each cell with a text for every candidate cell type, "
                   "all 3,169 Cell Ontology names when unconstrained, the expected cell types when "
                   "constrained and the drawn list in the imperfect-list study. CellWhisperer and "
                   "mmContext take the candidate names as "
                   "written. LangCell takes the text below, with the name and definition from the "
                   "Cell Ontology file its authors provide, and "
                   f"\\texttt{{{bare}}} for a candidate that file lacks.")
    for heading, used_in, const in CLUSTER_BLOCKS:
        lines += block(heading, used_in, getattr(P, const))
    lines += [
        "\\paragraph{Imperfect-list study.} The study used the templates of condition C4, the "
        "per-cell template for GPT-OSS-120B, C2S-Scale's own for C2S-Scale 2B and the per-cluster "
        "template for Gemini 3.1 Pro, with the drawn list in place of the dataset's own. The "
        "conditions that offer the answer uncertain (Supplementary "
        "Table~\\ref{supptab:list_conditions}) add the rule below, as rule 5 of the per-cell and "
        "per-cluster templates and directly after the list in C2S-Scale's template.",
        "\\begin{quote}\\footnotesize\\ttfamily\\raggedright",
        _verbatimish(P.UNCERTAIN_RULE),
        "\\end{quote}",
        "",
    ]
    out_path.write_text("\n".join(lines))
    return out_path


# Characters a prompt block must show as typed. \symbol takes the glyph from the typewriter font
# itself, which \{ and \_ do not in the OT1 encoding of the BiB build. The typewriter fonts have no
# em dash (OT1 gives "|", and "---" shows three hyphens), so it comes from the roman font.
_TT_CHARS = {"\\": "\\symbol{92}", "{": "\\symbol{123}", "}": "\\symbol{125}",
             "_": "\\symbol{95}", "^": "\\symbol{94}", "~": "\\symbol{126}", "&": "\\&",
             "%": "\\%", "#": "\\#", "$": "\\$", "—": "{\\rmfamily\\textemdash}"}


def _verbatimish(text):
    """Typeset prompt text in a monospaced block without a verbatim environment.

    ``verbatim`` cannot be used: the templates contain braces, which the reader must SEE because
    they mark the substitution slots, and they contain an em dash that a verbatim environment
    passes through unprocessed. Escaping by hand and forcing the line breaks keeps both visible
    and keeps the block inside a ``quote``, which ``verbatim`` would not allow.
    """
    return "\n".join("".join(_TT_CHARS.get(c, c) for c in line) + " \\\\"
                     for line in text.split("\n"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(DEFAULT_ABLATION_CSV))
    p.add_argument("--lists", default=str(DEFAULT_LIST_RUNS_CSV))
    p.add_argument("--outdir", default="scLLMs_benchmark_paper/bioRxiv_template")
    args = p.parse_args(argv)

    raw = load(args.input)
    outdir = Path(args.outdir)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "tables" / "supp_ablation_results.csv"
    per_dataset_frame(raw).to_csv(csv_path, index=False)
    print(f"[csv]  {csv_path}")

    runs = pd.read_csv(args.lists)
    conditions = yaml.safe_load((Path(__file__).resolve().parents[1] / "context_ablation"
                                 / "config.yml").read_text())["lists"]["conditions"]
    list_path = outdir / "tables" / "supp_list_results.csv"
    list_frame(runs).to_csv(list_path, index=False)
    print(f"[csv]  {list_path}")

    tex_path = write_tex(raw, outdir / "supplementary_ablation_tables_generated.tex")
    print(f"[tex]  {tex_path}")
    # The list study's tables in a file of their own: the supplement places R2's bootstrap table
    # between the two studies, in the order the text first cites them.
    list_tex = outdir / "supplementary_list_tables_generated.tex"
    list_tex.write_text("\n".join(["% GENERATED by bin/ablation_tables.py -- do not edit by hand.",
                                   "% The imperfect-list study. Requires \\usepackage{longtable} "
                                   "and booktabs.", ""] + write_list_tex(runs, conditions)))
    print(f"[tex]  {list_tex}")

    prompts_path = write_prompts_tex(outdir / "supplementary_prompts_generated.tex")
    print(f"[tex]  {prompts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
