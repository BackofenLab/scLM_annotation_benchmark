#!/usr/bin/env python
"""Generate the supplementary tables for the prompt-context ablation (Results R3).

Three typeset tables plus one reusable CSV:

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

Nothing is hand-written, so these cannot drift from Figure 3 (style Rule 6).

Usage:
    python bin/ablation_tables.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manuscript_figures import DEFAULT_ABLATION_CSV  # noqa: E402

# Per-arm facts that are NOT in the results CSV: the checkpoint or endpoint, where it ran, and
# whose prompt template it used. Everything else in the design table is derived from the data.
# Sourced from context_ablation/config.yml; if that changes, change this.
ARM_META = {
    "A1_llm_percell": ("GPT-OSS-120B", "openai/gpt-oss-120b", "hosted API", "cell",
                       "benchmark instruction"),
    "A2_base_percell": ("Gemma-2-2B", "google/gemma-2-2b", "local GPU", "cell",
                        "benchmark instruction"),
    "A3_scllm_percell": ("C2S-Scale-2B", "vandijklab/C2S-Scale-Gemma-2-2B", "local GPU", "cell",
                         "C2S-Scale's own"),
    "A4_llm_percluster": ("GPT-OSS-120B", "openai/gpt-oss-120b", "hosted API", "cluster",
                          "cluster marker genes"),
    "A8_gemini_percluster": ("Gemini-3.1-Pro", "google/gemini-3.1-pro-preview", "hosted API",
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
# PLAIN ACCURACY DROPPED 2026-08-15, matching what bin/supplementary_tables.py already does for the
# per-dataset benchmark tables. Over the 36 rows of this table it correlates with balanced accuracy
# at Spearman 0.985 and Pearson 0.995, so it reorders almost nothing, while Lin correlates at 0.953
# and does carry independent information. It remains in supp_ablation_results.csv, which the
# caption points to, so nothing is lost.
TEX_METRICS = [
    ("balanced_accuracy", "Bal.\\ acc."),
    ("lin_avg_similarity", "Lin"),
    ("non_cl_ratio", "non-CL"),
    ("frac_empty", "Empty"),
    ("frac_over_8_words", "$>$8 words"),
]


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
                     ", ".join(conditions), "; ".join(variants),
                     int(sub["seed"].nunique()), int(sub["dataset"].nunique())))
    return rows


def write_tex(raw, out_path, csv_dir_name="tables"):
    lines = [
        "% GENERATED by bin/ablation_tables.py -- do not edit by hand.",
        "% Regenerate after any change to context_ablation/results/context_ablation.csv so these",
        "% tables cannot drift from Figure 3. Requires \\usepackage{longtable} and booktabs.",
        "",
    ]

    # ---- design ---------------------------------------------------------
    lines += [
        "\\begin{table}[!ht]", "\\centering", "\\scriptsize",
        "\\begin{tabular}{lllllll}", "\\toprule",
        "Model & Identifier & Run on & Unit & Prompt template & Conditions & Gene variants \\\\",
        "\\midrule",
    ]
    for name, ident, host, gran, template, conds, variants, _ns, _nd in design_rows(raw):
        lines.append(" & ".join([_tex(name), f"\\texttt{{{_tex(ident)}}}", host, gran,
                                 _tex(template), _tex(conds), _tex(variants)]) + " \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}",
        "\\caption{\\textbf{Arms of the prompt-context ablation.}\\\\ "
        "Each row is one arm. \\emph{Unit} is what one model call annotated, a single cell or a "
        "whole cluster. \\emph{Prompt template} names whose wording the arm used: the benchmark's "
        "own template for the general-purpose models, C2S-Scale's for the model it was released "
        "with, and a marker-gene template for the cluster-level arms. \\emph{Conditions} are the "
        "prompt conditions of Supplementary Table~\\ref{supptab:ablation_conditions}. "
        "\\emph{Gene variants} are the gene vocabularies for the per-cell arms and the clustering "
        "resolutions for the per-cluster arms. Every combination ran three times under a different "
        "seed on each of the four datasets, at 1,000 cells per dataset. The table is generated "
        "from the scored results, so it lists what was run rather than what was configured.}",
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
        "\\caption{\\textbf{What each prompt condition supplies.}\\\\ "
        "Conditions are ordered by how much they supply, from least to most, which is the order "
        "they appear on the x axis of the prompt-context figure. "
        "A dash marks information the condition withholds. \\emph{Gene order} is whether the gene "
        "list keeps its descending-expression ranking or is permuted while holding the same genes, "
        "the same count and the same surrounding prompt text. \\emph{Benchmark regime} names the "
        "setting of the main benchmark that a condition reproduces. The prompt templates of those "
        "two conditions are copied from the benchmark's own model configurations rather than "
        "rewritten, which is what licenses comparing the values here with the rest of the paper.}",
        "\\label{supptab:ablation_conditions}", "\\end{table}", "",
    ]

    # ---- results --------------------------------------------------------
    overall = overall_frame(raw)
    order = {a: i for i, a in enumerate(ARM_ORDER)}
    overall["_a"] = overall["arm"].map(order)
    overall["_c"] = overall["mode"].map(_cond_key)
    overall = overall.sort_values(["_a", "variant", "_c"])

    header = ["Model", "Unit", "Genes", "Cond."] + [h for _c, h in TEX_METRICS]
    lines += [
        "\\scriptsize",
        "\\begin{longtable}{llll" + "r" * len(TEX_METRICS) + "}",
        "\\caption{\\textbf{Complete results of the prompt-context ablation.} "
        "One row per arm, gene variant and prompt condition, with conditions in the order of "
        "increasing supplied context. Each value is the mean over the four "
        "datasets of that dataset's mean over three seeds, at 1,000 cells per dataset, so $n = 4$ "
        "datasets and 12 runs per row. Bold marks the best value within each arm, so it identifies "
        "the condition that won for that model rather than making a comparison across models, "
        "which differ in size, annotation unit and prompt template. Best means the highest "
        "balanced accuracy and Lin similarity and the lowest non-CL fraction. "
        "\\emph{non-CL} is the fraction of predictions matching no "
        "Cell Ontology term, \\emph{Empty} the fraction returning no answer, and \\emph{$>$8 "
        "words} the fraction running past eight words, both diagnostics of answer format rather "
        "than of accuracy. Conditions are defined in Supplementary "
        "Table~\\ref{supptab:ablation_conditions} and arms in Supplementary "
        "Table~\\ref{supptab:ablation_design}. Per-dataset values, the between-seed and "
        "between-dataset spread, and every remaining metric are in "
        f"\\texttt{{{csv_dir_name}/supp\\_ablation\\_results.csv}}.}}"
        "\\label{supptab:ablation_results}\\\\",
        "\\toprule", " & ".join(header) + " \\\\", "\\midrule", "\\endfirsthead",
        "\\multicolumn{%d}{l}{\\emph{Ablation results, continued}}\\\\" % len(header),
        "\\toprule", " & ".join(header) + " \\\\", "\\midrule", "\\endhead",
        "\\midrule",
        "\\multicolumn{%d}{r}{\\emph{continued on the next page}}\\\\" % len(header),
        "\\endfoot", "\\bottomrule", "\\endlastfoot",
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
    # bin/supplementary_tables.py excludes non-CL from bolding, and that exclusion is right for the
    # reason it gives -- marking the LARGEST fraction would flag the method most dependent on our
    # manual mapping as though it had won. Marking the SMALLEST is the opposite and is meaningful:
    # it is the run whose raw output was already closest to the ontology, which is the mechanism
    # R3.3 argues from. So the direction, not the column, is what made the earlier exclusion right.
    #
    # Empty and >8 words stay unbolded. They are at or near zero for every arm except the base
    # model, so a bold there would mark a tie between runs that never had the failure mode.
    BOLD_MAX = {"balanced_accuracy", "lin_avg_similarity"}
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


# --------------------------------------------------------------------------- #
# Prompt templates. IMPORTED from the study's own module, never transcribed -- R3 claims that one
# prompt beats another, so a reader must be able to see both, and a hand-copied version could
# silently stop matching what was actually sent. context_ablation/src/prompts.py has no
# dependencies beyond the standard library, so importing it here is free.
#
# M5's two templates exist in that module and are deliberately NOT listed: the mode was removed
# from the study on 2026-08-13 together with its prompts, predictions and scored runs.
# --------------------------------------------------------------------------- #
# Condition lists use the manuscript's C-labels (see CONDITION_LABEL): M1->C2, M2->C4, M3->C3,
# M4->C5, M6->C1. Written out rather than mapped, because these are editorial groupings of
# templates and not a column of the data.
PROMPT_BLOCKS = [
    ("instruction", "Per-cell template, expected cell types withheld",
     "INSTRUCTION_NO_TYPES", "C1, C2, C3"),
    ("instruction", "Per-cell template, expected cell types supplied",
     "INSTRUCTION_WITH_TYPES", "C4, C5"),
    ("c2s", "C2S-Scale's own per-cell template, expected cell types withheld",
     "C2S_NO_TYPES", "C1, C2, C3"),
    ("c2s", "C2S-Scale's own per-cell template, expected cell types supplied",
     "C2S_WITH_TYPES", "C4, C5"),
    ("marker_genes", "Per-cluster template, expected cell types withheld",
     "MARKER_NO_TYPES", "C2, C3"),
    ("marker_genes", "Per-cluster template, expected cell types supplied",
     "MARKER_WITH_TYPES", "C4, C5"),
]


def write_prompts_tex(out_path):
    """Every prompt template the ablation sent, verbatim, with the slots explained."""
    src = Path(__file__).resolve().parents[1] / "context_ablation" / "src"
    sys.path.insert(0, str(src))
    import prompts as P  # noqa: E402

    lines = [
        "% GENERATED by bin/ablation_tables.py -- do not edit by hand.",
        "% The templates are IMPORTED from context_ablation/src/prompts.py, so they cannot drift",
        "% from what was actually sent to the models.",
        "",
        # A \label inside an unnumbered \subsection* attaches to whatever counter was stepped last
        # and resolves to an EMPTY string, so "Supplementary Note~\ref{...}" renders with nothing
        # after it and hyperref warns about an empty link. A dedicated counter, stepped with
        # \refstepcounter so \ref picks it up, is what makes the cross-reference work. \newcounter
        # is safe here because this file is input exactly once.
        "\\newcounter{suppnote}",
        "\\refstepcounter{suppnote}",
        "\\subsection*{Supplementary Note \\thesuppnote: prompt templates of the context "
        "ablation}\\label{suppnote:prompts}",
        "",
        "Every template used by the prompt-context ablation is reproduced below as it was sent, "
        "with the substitution slots left in braces. "
        "\\texttt{num\\_genes} is the cap on the gene list, 200 for the per-cell arms and 25 for "
        "the per-cluster arms. "
        "\\texttt{cell\\_sentence} and \\texttt{marker\\_genes} carry the gene names. "
        "\\texttt{unique\\_cell\\_types} is the set of cell types present in the dataset, permuted "
        "independently for each cell or cluster. "
        "\\texttt{organism} is filled by the C2S-Scale template only. "
        "\\texttt{context} is the tissue-and-species phrase, empty in the conditions that withhold "
        "it and rendered inline otherwise, for example \\texttt{ mouse nervous system}. "
        "One asymmetry follows from that and is not an oversight: the C2S-Scale template already "
        "names the organism, so adding tissue and species supplies only the tissue for that arm. "
        "The conditions are defined in Supplementary "
        "Table~\\ref{supptab:ablation_conditions} and the arms in Supplementary "
        "Table~\\ref{supptab:ablation_design}.",
        "",
        "The two conditions that reproduce the benchmark's own regimes carry its wording "
        "unchanged, copied from the model configurations the pipeline ships rather than rewritten. "
        "That is what licenses comparing any value in these results with the rest of the paper.",
        "",
    ]
    for _template, heading, const, modes in PROMPT_BLOCKS:
        body = getattr(P, const).rstrip("\n")
        lines += [
            f"\\paragraph{{{heading}.}} Used in conditions {modes}.",
            "\\begin{quote}\\footnotesize\\ttfamily\\raggedright",
            _verbatimish(body),
            "\\end{quote}",
            "",
        ]
    out_path.write_text("\n".join(lines))
    return out_path


def _verbatimish(text):
    """Typeset prompt text in a monospaced block without a verbatim environment.

    ``verbatim`` cannot be used: the templates contain braces, which the reader must SEE because
    they mark the substitution slots, and they contain an em dash that a verbatim environment
    passes through unprocessed. Escaping by hand and forcing the line breaks keeps both visible
    and keeps the block inside a ``quote``, which ``verbatim`` would not allow.
    """
    out = []
    for line in text.split("\n"):
        line = (line.replace("\\", "\\textbackslash{}")
                    .replace("{", "\\{").replace("}", "\\}")
                    .replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")
                    .replace("#", "\\#").replace("$", "\\$")
                    .replace("—", "---"))
        out.append(line + " \\\\")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(DEFAULT_ABLATION_CSV))
    p.add_argument("--outdir", default="scLLMs_benchmark_paper/bioRxiv_template")
    args = p.parse_args(argv)

    raw = load(args.input)
    outdir = Path(args.outdir)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "tables" / "supp_ablation_results.csv"
    per_dataset_frame(raw).to_csv(csv_path, index=False)
    print(f"[csv]  {csv_path}")

    tex_path = write_tex(raw, outdir / "supplementary_ablation_tables_generated.tex")
    print(f"[tex]  {tex_path}")

    prompts_path = write_prompts_tex(outdir / "supplementary_prompts_generated.tex")
    print(f"[tex]  {prompts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
