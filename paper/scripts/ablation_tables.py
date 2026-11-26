#!/usr/bin/env python
"""Generate the supplementary tables of the prompt-context ablation and the imperfect-list study,
and the note with every prompt.

Writes to --outdir:

* ``tables/supp_ablation_results.csv``: one row per arm, gene variant, condition and dataset, with
  the mean and the standard deviation over the seeds of every metric the ablation scores;
* ``tables/supp_list_results.csv``: the same for the list study, per method, condition and
  dataset;
* ``supplementary_ablation_tables_generated.tex``: the ablation's arms, read from the results,
  and its results;
* ``supplementary_list_tables_generated.tex``: the results of the list study per method and
  condition, in the order of paper/experiments/config.yml;
* ``supplementary_prompts_generated.tex``: every prompt of the benchmark and of the two studies,
  imported from paper/experiments/src/prompts.py or built by the benchmark wrappers' own
  functions.

    python paper/scripts/ablation_tables.py [--input CSV] [--lists CSV] [--outdir DIR]
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import io
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

import paths
from table_format import best_as_printed, bold, fmt, join, tex
from methods import (DATASET_ORDER, MODE_LABELS, MODE_ORDER, STUDY_ARM_MODEL, _display,
                     _list_study_arms, dataset_label)

# The facts of each arm that are not in the results, as in paper/experiments/config.yml: name,
# model identifier and unit.
ARM_META = {
    "A1_llm_percell": ("GPT-OSS-120B", "openai/gpt-oss-120b", "cell"),
    "A2_base_percell": ("Gemma-2-2B", "google/gemma-2-2b", "cell"),
    "A3_scllm_percell": ("C2S-Scale 2B", "vandijklab/C2S-Scale-Gemma-2-2B", "cell"),
    "A4_llm_percluster": ("GPT-OSS-120B", "openai/gpt-oss-120b", "cluster"),
    "A8_gemini_percluster": ("Gemini 3.1 Pro", "google/gemini-3.1-pro-preview", "cluster"),
}
ARM_ORDER = list(ARM_META)
# The study's code and files name the conditions M1 to M6; the tables print the figures' labels.
CONDITION_LABEL = {m: MODE_LABELS[m].replace("\n", " ") for m in MODE_ORDER}


def _cond(mode):
    """Label of a condition."""
    return CONDITION_LABEL[mode]


def _cond_key(mode):
    """Sort key that puts the conditions in the order of MODE_ORDER."""
    return MODE_ORDER.index(mode)


VARIANT_LABEL = {"expression_ranked": "all genes", "hvg": "5,000 variable genes",
                 "res0.5": "resolution 0.5", "res1.0": "resolution 1.0"}

# The metrics of the typeset results table; the CSV keeps every column the scorer writes.
TEX_METRICS = [
    ("balanced_accuracy", "Bal.\\ acc."),
    ("agreement_score", "Agree."),
    ("lin_avg_similarity", "Lin"),
    ("non_cl_ratio", "non-CL"),
    ("frac_over_8_words", "$>$8 words"),
]


# The conditions of the list study as the figures name them. The conditions themselves are read
# from paper/experiments/config.yml.
LIST_CONDITION_NAME = {"E": "exact list", "EU": "exact list + uncertain", "L1": "A1",
                       "L2-2": "A2 (2 types)", "L2-5": "A2", "L3": "A3", "L4": "A1 + A2 + A3"}
PROMPTED_ARMS = {"A1_llm_percell", "A3_scllm_percell", "A8_gemini_percluster"}
# Columns of the list-study table: balanced accuracy over all types and over the types still listed
# (where types are removed), the coarser and finer answers that added relatives draw, and
# "uncertain" where offered.
LIST_METRICS = [("balanced_accuracy", "Bal.\\ acc."), ("agreement", "Agree."), ("lin", "Lin"),
                ("balanced_accuracy_listed", "Listed"), ("all_coarser", "Coarser"),
                ("all_finer", "Finer"), ("abstained_removed", "Unc.\\ missing"),
                ("abstained_listed", "Unc.\\ listed")]


def per_dataset_frame(raw):
    """One row per (arm, variant, condition, dataset): mean and SD over the three seeds."""
    metrics = [c for c in raw.columns
               if pd.api.types.is_numeric_dtype(raw[c]) and c not in ("seed",)]
    grouped = raw.groupby(["arm", "variant", "mode", "dataset"])[metrics]
    out = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_sd"))
    out["n_seeds"] = grouped.size()
    out = out.reset_index()
    # The study's name of the condition and the tables' label.
    out.insert(out.columns.get_loc("mode") + 1, "condition", out["mode"].map(_cond))
    return out.sort_values(["arm", "variant", "mode"],
                           key=lambda s: s.map(_cond_key) if s.name == "mode" else s)


def overall_frame(raw):
    """One row per arm, variant and condition: the mean over the datasets of the means over the
    seeds, and the standard deviation of those means across the datasets."""
    per_ds = (raw.groupby(["arm", "variant", "mode", "dataset"])
              .mean(numeric_only=True).reset_index())
    grouped = per_ds.groupby(["arm", "variant", "mode"])
    metrics = [c for c, _h in TEX_METRICS]
    out = grouped[metrics].mean().add_suffix("_mean").join(
        grouped[metrics].std(ddof=1).add_suffix("_sd"))
    out["n_datasets"] = grouped.size()
    return out.reset_index()


def _stacked(items):
    """Several entries one under another in one table cell."""
    if len(items) == 1:
        return items[0]
    return "\\begin{tabular}[t]{@{}l@{}}" + "\\\\".join(items) + "\\end{tabular}"


def design_rows(raw):
    """Per arm, as the results hold it: name, identifier, unit, whether the arm also ran the
    shuffled gene order, and its variants."""
    rows = []
    for arm in ARM_ORDER:
        sub = raw[raw["arm"] == arm]
        name, ident, gran = ARM_META[arm]
        # Every arm ran the four conditions with genes in their order.
        modes = set(sub["mode"])
        assert {"M1", "M2", "M3", "M4"} <= modes, arm
        variants = [VARIANT_LABEL[v] for v in
                    sorted(set(sub["variant"]), key=lambda v: list(VARIANT_LABEL).index(v))]
        rows.append((name, ident, gran, "yes" if "M6" in modes else "--", variants))
    return rows


def write_tex(raw, out_path):
    lines = [
        "% GENERATED by paper/scripts/ablation_tables.py -- do not edit by hand.",
        "% Regenerate after any change to paper/experiments/results/context_ablation.csv so",
        "% these tables cannot drift from Figure 3. Requires booktabs.",
        "",
    ]

    lines += [
        "\\begin{table}[!ht]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lllll}", "\\toprule",
        "Model & Identifier & Unit & Shuffled order & Variants \\\\",
        "\\midrule",
    ]
    # A little space between the arms, whose variants stack over two lines.
    for i, (name, ident, gran, shuffled, variants) in enumerate(design_rows(raw)):
        lines += ([] if i == 0 else ["\\addlinespace[3pt]"]) + [
            " & ".join([tex(name), f"\\texttt{{{tex(ident)}}}", gran,
                        tex(shuffled), _stacked([tex(v) for v in variants])]) + " \\\\"]
    lines += [
        "\\bottomrule", "\\end{tabular}",
        "\\captionsetup{position=below}",
        "\\caption{\\textbf{Arms of the prompt-context ablation.} Each row is one arm. "
        "\\emph{Identifier} is the model's identifier on OpenRouter for GPT-OSS-120B and Gemini "
        "3.1 Pro and on Hugging Face for Gemma-2-2B and C2S-Scale 2B, and \\emph{Unit} is what "
        "one model call annotated, a single cell or a cluster. "
        "Every arm received genes only, genes + tissue, genes + list and genes + tissue + list, "
        "and \\emph{Shuffled order} marks the arms that also received the genes of genes only in "
        "shuffled order. \\emph{Variants} are the gene vocabularies of the per-cell arms and the "
        "clustering resolutions of the per-cluster arms, and the per-cell arms ran the variant "
        "with 5,000 variable genes in genes only and in genes + tissue + list.}",
        "\\label{supptab:ablation_design}", "\\end{table}", "",
    ]

    overall = overall_frame(raw)
    order = {a: i for i, a in enumerate(ARM_ORDER)}
    overall["_a"] = overall["arm"].map(order)
    overall["_c"] = overall["mode"].map(_cond_key)
    overall = overall.sort_values(["_a", "variant", "_c"])

    header = ["Model", "Unit", "Variant", "Condition"] + [h for _c, h in TEX_METRICS]
    # One page holds the 36 rows, so the table floats like the arms table.
    lines += [
        "\\begin{table}[!ht]", "\\centering", "\\scriptsize",
        "\\begin{tabular}{llll" + "r" * len(TEX_METRICS) + "}",
        "\\toprule", " & ".join(header) + " \\\\", "\\midrule",
    ]
    # The best value of each score is bold within each arm, where model and unit are fixed: the
    # highest balanced accuracy, agreement score and Lin similarity and the lowest share of answers
    # that are not Cell Ontology terms. Values are compared as printed.
    BOLD_MAX = {"balanced_accuracy", "agreement_score", "lin_avg_similarity"}
    BOLD_MIN = {"non_cl_ratio"}
    best = {(arm, col): best_as_printed(sub[f"{col}_mean"], lowest=col in BOLD_MIN)
            for arm, sub in overall.groupby("arm") for col in BOLD_MAX | BOLD_MIN}

    # The model, unit and variant where they change, and a little space between the arms.
    previous = (None, None)
    for _, r in overall.iterrows():
        name, _ident, gran = ARM_META[r["arm"]]
        new_arm = r["arm"] != previous[0]
        new_variant = new_arm or r["variant"] != previous[1]
        if new_arm and previous[0] is not None:
            lines.append("\\addlinespace[3pt]")
        previous = (r["arm"], r["variant"])
        cells = [tex(name) if new_arm else "", gran if new_arm else "",
                 tex(VARIANT_LABEL[r["variant"]]) if new_variant else "",
                 tex(_cond(r["mode"]))]
        for col, _h in TEX_METRICS:
            cells.append(bold(fmt(r.get(f"{col}_mean")), best.get((r["arm"], col))))
        lines.append(" & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}",
        "\\captionsetup{position=below}",
        "\\caption{\\textbf{Results of the prompt-context ablation.} One row per arm, variant "
        "and condition. \\emph{Model} and \\emph{Unit} name the arm, and \\emph{Variant} is the "
        "gene vocabulary of a per-cell arm or the clustering resolution of a per-cluster arm. "
        "Each value is the mean over the four datasets, at 1,000 cells each, of the mean over "
        "three seeds per dataset. \\emph{Bal.\\ acc.} is balanced accuracy and \\emph{Agree.} the "
        "agreement score, which gives a cell 1 for its true type, 0.5 for a finer type or for a "
        "coarser type with at most 250 terms at or below it in the Cell Ontology and 0 "
        "otherwise, both averaged over the cells of each type and then over the types. "
        "\\emph{Lin} is the Lin similarity between the answer and the true type in the Cell "
        "Ontology, averaged over cells. \\emph{non-CL} is the share of answers that are, as "
        "written, not the name or identifier of a Cell Ontology term, and \\emph{$>$8 words} the "
        "share that runs past eight words. Within each arm, the highest balanced accuracy, "
        "agreement score and Lin similarity and the lowest non-CL share are highlighted in bold.}",
        "\\label{supptab:ablation_results}", "\\end{table}", "",
    ]

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
    """The list study's results per method and condition, in the order of its configuration."""
    per_ds = runs.groupby(["arm", "condition", "dataset"]).mean(numeric_only=True)
    overall = per_ds.groupby(["arm", "condition"]).mean()
    header = ["Method", "Condition"] + [h for _c, h in LIST_METRICS]
    lines = [
        "\\clearpage", "\\scriptsize", "\\captionsetup{position=below}",
        "\\begin{longtable}{ll" + "r" * len(LIST_METRICS) + "}",
        "\\toprule", " & ".join(header) + " \\\\", "\\midrule", "\\endfirsthead",
        "\\multicolumn{%d}{l}{\\emph{Imperfect-list results, continued}}\\\\" % len(header),
        "\\toprule", " & ".join(header) + " \\\\", "\\midrule", "\\endhead",
        "\\midrule",
        "\\multicolumn{%d}{r}{\\emph{continued on the next page}}\\\\" % len(header),
        "\\endfoot", "\\bottomrule",
        "\\caption{\\textbf{Results of the imperfect-list analysis.} One row per method and "
        "condition. The conditions are the exact list, the exact list with the answer uncertain "
        "(exact list + uncertain), a parent or a child added for half of the types (A1), two or "
        "five types absent from the data added (A2 (2 types) or A2), a quarter of the types "
        "removed (A3) and all three errors (A1 + A2 + A3). Each value is the mean over the four "
        "datasets, at 1,000 cells each, of the mean over three seeds per dataset. "
        "\\emph{Bal.\\ acc.} is balanced accuracy and \\emph{Agree.} the agreement score, which "
        "gives a cell 1 for its true type, 0.5 for a finer type or for a coarser type with at most "
        "250 terms at or below it in the Cell Ontology and 0 otherwise, both averaged over the "
        "cells of each type and then over all true types, including those removed from the list. "
        "\\emph{Lin} is the Lin similarity between the answer and the true type in the Cell "
        "Ontology, averaged over cells. \\emph{Listed} is balanced accuracy over the types still "
        "on the list, given where types were removed. \\emph{Coarser} and \\emph{Finer} are the "
        "shares of cells answered with a coarser type that has at most 250 terms at or below it "
        "and with a finer type. "
        "\\emph{Unc.\\ missing} and \\emph{Unc.\\ listed} are the shares of the cells of the "
        "missing and of the listed types that were answered uncertain, given where that answer "
        "was offered. Shares are averaged over cell types. Gemini 3.1 Pro, scType and SCSA "
        "annotated clusters, the other methods single cells.}"
        "\\label{supptab:list_results}\\\\",
        "\\endlastfoot",
    ]
    for arm in _list_study_arms(runs):
        for name, spec in conditions.items():
            if (arm, name) not in overall.index:
                continue
            row = overall.loc[(arm, name)]
            shown = {"balanced_accuracy": True, "agreement": True, "lin": True,
                     "all_coarser": True, "all_finer": True,
                     "balanced_accuracy_listed": bool(spec.get("remove")),
                     "abstained_removed": arm in PROMPTED_ARMS and bool(spec.get("uncertain"))
                     and bool(spec.get("remove")),
                     "abstained_listed": arm in PROMPTED_ARMS and bool(spec.get("uncertain"))}
            lines.append(" & ".join([tex(_display(STUDY_ARM_MODEL[arm])),
                                     tex(LIST_CONDITION_NAME[name])]
                                    + [fmt(row[c]) if shown[c] else "--"
                                       for c, _h in LIST_METRICS]) + " \\\\")
    lines += ["\\end{longtable}", ""]
    return lines


# The prompt templates are not transcribed. The general-purpose LLM, C2S-Scale and per-cluster
# templates are imported from paper/experiments/src/prompts.py, which reads them from the
# benchmark's prompt files (models/openrouter_LLM_*/prompt.txt, models/c2s_scale_*/prompt.txt)
# and adds the {context} slot; with {context} empty they are the benchmark's templates, so one
# block serves both. InstructCell's prompt and
# LangCell's text for a candidate are built by the benchmark wrappers' own functions. prompts.py
# also holds the templates of a condition without genes, which the study does not report.
# (heading, where the template was used, prompts.py constant). The ablation's conditions without
# the list are M6, M1 and M3, those with it M2 and M4.
SHARED_BLOCKS = [
    ("Per-cell template, unconstrained",
     "the benchmark by the general-purpose LLMs and in the ablation by GPT-OSS-120B and "
     "Gemma-2-2B per cell, in the conditions without the list", "INSTRUCTION_NO_TYPES"),
    ("Per-cell template, constrained",
     "the benchmark by the general-purpose LLMs and in the ablation by GPT-OSS-120B and "
     "Gemma-2-2B per cell, in the conditions with the list", "INSTRUCTION_WITH_TYPES"),
    ("C2S-Scale's template, unconstrained",
     "the benchmark by C2S-Scale 2B and 27B and in the ablation by C2S-Scale 2B, in the "
     "conditions without the list", "C2S_NO_TYPES"),
    ("C2S-Scale's template, constrained",
     "the benchmark by C2S-Scale 2B and 27B and in the ablation by C2S-Scale 2B, in the "
     "conditions with the list", "C2S_WITH_TYPES"),
]
CLUSTER_BLOCKS = [
    ("Per-cluster template, unconstrained", "the ablation by GPT-OSS-120B and Gemini 3.1 Pro "
     "per cluster, in the conditions without the list", "MARKER_NO_TYPES"),
    ("Per-cluster template, constrained", "the ablation by GPT-OSS-120B and Gemini 3.1 Pro per "
     "cluster, in the conditions with the list", "MARKER_WITH_TYPES"),
]


def _wrapper_function(relpath, name, **scope):
    """Function ``name`` of a benchmark wrapper, compiled on its own. The wrappers import their
    model stacks at the top, which this environment lacks, and the prompt code needs none of it."""
    path = paths.MODELS / relpath
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
    unconstrained = _wrapper_function("instructcell_unconstrained/src/predict.py", "build_prompt",
                                      **scope)(meta)
    constrained = _wrapper_function("instructcell_constrained/src/predict.py", "build_prompt",
                                    **scope)(meta, ["{unique_cell_types}"])
    text = _wrapper_function("langcell_unconstrained/src/predict.py", "gettextfromname")
    with contextlib.redirect_stdout(io.StringIO()):  # the wrapper prints every text it builds
        defined = text("{name}", {"{name}": "id"}, {}, {"id": {"name": "{name}",
                                                              "def": "{definition}"}})
        bare = text("{candidate}", {}, {}, {})
    return unconstrained, constrained, defined.rstrip(), bare


def _dataset_configs():
    return {ds: yaml.safe_load((paths.DATASET_CONFIGS / f"{ds}.yml").read_text())
            for ds in DATASET_ORDER}


def _context_phrases(P, cfg):
    """The species-and-tissue phrase each dataset filled the ablation's {context} with, built by
    the study's own function from the configuration it read."""
    phrases = []
    for ds, c in cfg.items():
        phrase = P.context_phrase({"tissue_species": True}, "instruction", tissue=c.get("tissue"),
                                  species=c.get("species") or c.get("organism"),
                                  override=c.get("context_phrase")).strip()
        phrases.append(f"\\texttt{{{phrase}}} for {dataset_label(ds)}")
    return join(phrases)


def _instructcell_slots(cfg):
    """The species and tissue InstructCell received, from the dataset configurations."""
    species = sorted({c["species"] for c in cfg.values()})
    return " or ".join(species), join([f"\\texttt{{{tex(c['tissue'])}}} for "
                                          f"{dataset_label(ds)}" for ds, c in cfg.items()])


def write_prompts_tex(out_path):
    """Every prompt the benchmark and the two studies sent, verbatim, with the slots explained."""
    sys.path.insert(0, str(paths.EXPERIMENT_SOURCES))
    import prompts as P

    cfg = _dataset_configs()
    phrases = _context_phrases(P, cfg)
    lines = [
        "% GENERATED by paper/scripts/ablation_tables.py -- do not edit by hand.",
        "% The templates are imported from paper/experiments/src/prompts.py or built by the",
        "% benchmark wrappers' own functions, so they cannot drift from what was sent.",
        "",
        # The note's own counter, defined in the supplement, so that \ref to the label gives
        # the note's number.
        "\\refstepcounter{suppnote}",
        "\\paragraph{Supplementary Note \\thesuppnote: Prompt templates.}\\label{suppnote:prompts}",
        "",
        "The prompt templates of the benchmark, the prompt-context ablation and the "
        "imperfect-list analysis are reproduced below, with the substitution slots in braces. "
        "\\texttt{num\\_genes} caps the gene list at 200 per cell and 25 per cluster. "
        "\\texttt{cell\\_sentence} and \\texttt{marker\\_genes} carry the gene names, and "
        "\\texttt{organism} names the dataset's organism, Homo sapiens or Mus musculus. "
        "\\texttt{unique\\_cell\\_types} is the list of expected cell types. The per-cell "
        "templates receive it as a Python list, permuted independently for each cell, the "
        "per-cluster templates separated by commas and permuted "
        "for each cluster, and the templates of C2S-Scale and InstructCell separated by commas and "
        "sorted. \\texttt{context} is empty in the benchmark. In the conditions with the tissue "
        f"and species, the prompt-context ablation fills it with them, {phrases}, and with the "
        "tissue alone in C2S-Scale's template, which already names the organism (Supplementary "
        "Table~\\ref{supptab:ablation_design}). The benchmark sent each prompt to the "
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
                   "InstructCell replaces \\texttt{input} with embeddings of the cell that its "
                   "expression encoder computes from the raw counts. The "
                   "template is the example prompt of the InstructCell repository without its "
                   "sequencing method, which the benchmark did not pass.")
    lines += block("InstructCell's template, constrained", "the benchmark", instruct_c,
                   " It adds the expected cell types in the benchmark's own wording, not in one "
                   "of the option phrasings InstructCell was trained with.")
    lines += block("Candidate texts of the embedding-based scLMs",
                   "the benchmark and the imperfect-list analysis", langcell_text,
                   " These models compare each cell with a text for every candidate cell type, "
                   "all 3,169 Cell Ontology names when unconstrained, the expected cell types when "
                   "constrained and the drawn list in the imperfect-list analysis. CellWhisperer "
                   "and mmContext take the candidate names as "
                   "written. LangCell takes the text below, with the name and definition from the "
                   "Cell Ontology file its authors provide, and "
                   f"\\texttt{{{bare}}} for a candidate that file lacks.")
    for heading, used_in, const in CLUSTER_BLOCKS:
        lines += block(heading, used_in, getattr(P, const))
    lines += [
        "\\paragraph{Imperfect-list analysis.} The analysis used the constrained templates, the "
        "per-cell template for GPT-OSS-120B, C2S-Scale's own for C2S-Scale 2B and the "
        "per-cluster template for Gemini 3.1 Pro, with the drawn list in place of the dataset's "
        "own. With the exact list + uncertain, A3 and A1 + A2 + A3, the templates add the rule "
        "below, as rule 5 of the per-cell and per-cluster templates and directly after the list "
        "in C2S-Scale's template.",
        "\\begin{quote}\\footnotesize\\ttfamily\\raggedright",
        _verbatimish(P.UNCERTAIN_RULE),
        "\\end{quote}",
        "",
    ]
    out_path.write_text("\n".join(lines))
    return out_path


# Characters a prompt block shows as typed. \symbol takes the glyph from the typewriter font
# itself, which \{ and \_ do not in the OT1 encoding. The typewriter fonts have no em dash (OT1
# gives "|", and "---" shows three hyphens), so it comes from the roman font.
_TT_CHARS = {"\\": "\\symbol{92}", "{": "\\symbol{123}", "}": "\\symbol{125}",
             "_": "\\symbol{95}", "^": "\\symbol{94}", "~": "\\symbol{126}", "&": "\\&",
             "%": "\\%", "#": "\\#", "$": "\\$", "—": "{\\rmfamily\\textemdash}"}


def _verbatimish(text):
    """Prompt text as escaped lines with forced breaks for a monospaced ``quote`` block, which a
    verbatim environment cannot sit in and whose em dash it would pass through unprocessed."""
    return "\n".join("".join(_TT_CHARS.get(c, c) for c in line) + " \\\\"
                     for line in text.split("\n"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(paths.ABLATION_RESULTS))
    p.add_argument("--lists", default=str(paths.LIST_RUNS))
    p.add_argument("--outdir", default=str(paths.TABLES))
    args = p.parse_args(argv)

    raw = pd.read_csv(args.input)
    outdir = Path(args.outdir)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "tables" / "supp_ablation_results.csv"
    per_dataset_frame(raw).to_csv(csv_path, index=False)
    print(f"[csv]  {csv_path}")

    runs = pd.read_csv(args.lists)
    conditions = yaml.safe_load(paths.EXPERIMENT_CONFIG.read_text())["lists"]["conditions"]
    list_path = outdir / "tables" / "supp_list_results.csv"
    list_frame(runs).to_csv(list_path, index=False)
    print(f"[csv]  {list_path}")

    tex_path = write_tex(raw, outdir / "supplementary_ablation_tables_generated.tex")
    print(f"[tex]  {tex_path}")
    # The list study's table in a file of its own, so that the supplement can place it apart.
    list_tex = outdir / "supplementary_list_tables_generated.tex"
    list_tex.write_text("\n".join(["% GENERATED by paper/scripts/ablation_tables.py -- do not "
                                   "edit by hand.",
                                   "% The imperfect-list analysis. Requires "
                                   "\\usepackage{longtable} and booktabs.", ""]
                                  + write_list_tex(runs, conditions)))
    print(f"[tex]  {list_tex}")

    prompts_path = write_prompts_tex(outdir / "supplementary_prompts_generated.tex")
    print(f"[tex]  {prompts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
