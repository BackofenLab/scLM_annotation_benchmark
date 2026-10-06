#!/usr/bin/env python
"""Generate the per-dataset supplementary result tables from the aggregate snapshot.

Two outputs, deliberately different in scope:

* ``tables/supp_results_<dataset>.csv`` -- EVERY metric column the snapshot carries. This is the
  artifact a reader reuses, and the one converted to the Word/Excel file BiB asks for at
  submission.
* ``supplementary_tables_generated.tex`` -- a READABLE subset of those columns, one longtable per
  dataset, plus a table recording which models were run and then excluded from the analysis.

The split is the point: a 20-column table is unreadable typeset and a 10-column one is not
transparent, so each format carries what it is good at and the caption points from one to the
other. Nothing here is hand-written, so the tables cannot drift from the figures (style Rule 6).

Usage:
    python bin/supplementary_tables.py --input cache/result/aggregated_by_dataset_model_<v>.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manuscript_figures import DATASET_ORDER, _display  # noqa: E402
from plot_model_comparison import (  # noqa: E402
    EXCLUDED_BASES, categorize_model, derive_mode, model_base, unsupported_pairs)

# Columns shown in the typeset table, in order, as (snapshot column, header).
TEX_COLUMNS = [
    # Two classification metrics, then the two ontology-graph similarities, then the label-space
    # diagnostics. Plain accuracy is dropped from the typeset table because balanced accuracy is
    # the paper's primary metric and the two move together; it remains in the CSV.
    ("balanced_accuracy", "Bal.\\ acc."),
    ("agreement_score", "Agree."),
    ("macro_f1", "Macro F1"),
    ("rada_avg_similarity", "Rada"),
    ("lin_avg_similarity", "Lin"),
    ("non_cl_ratio", "non-CL"),
    ("n_unique_pred_raw", "Pred.\\ labels"),
]
FAMILY = {"LLM": "general-purpose LLM", "scLLM": "scLLM",
          "classical": "baseline", "simple": "chance floor"}
# Dataset names for anything a READER sees. The manuscript uses the pipeline's keys only in
# Methods, where they let someone move between the paper and the repository; everywhere else, and
# in every caption, the readable name is used. Only human_pancreas differs -- the other three keys
# are already the names.
DATASET_DISPLAY = {"human_pancreas": "human pancreas"}
# Short forms for the typeset table only. The descriptive names above overflow the text block
# by ~75pt in the Family column; the CSVs keep the long form, where width costs nothing.
# Columns where a higher value is better, so the column maximum is bolded. non_cl_ratio and
# n_unique_pred_raw are deliberately excluded: they are diagnostics, not scores. Bolding the
# largest non-CL fraction would mark the method MOST dependent on our manual mapping, and the
# largest label count is not a best of anything.
BOLD_MAX = {"balanced_accuracy", "agreement_score", "macro_f1", "rada_avg_similarity",
            "lin_avg_similarity"}
FAMILY_TEX = {"general-purpose LLM": "LLM", "scLLM": "scLLM",
              "baseline": "baseline", "chance floor": "chance"}
REGIME = {"blind": "unconstrained", "unblind": "constrained", "baseline": "--"}

# Models that were run and then left out of the analysis. Editorial, not derivable from the
# snapshot (they are absent from it by construction), so the reasons live here and are the
# single place the exclusion rationale is recorded for the manuscript.
EXCLUDED = [
    ("Cell2Sentence (C2S-Pythia-410M)", "The only scLLM with no constrained counterpart. Its "
     "prompt formatter exposes no slot for a candidate label list, so one member of every paired "
     "comparison would be missing."),
    ("UCell", "Scored by taking the argmax over marker-set scores, which its authors do not "
     "prescribe. Their own tool refuses that step and returns an ambiguous call instead."),
    ("SingleR (per-tier variants)", "Superseded by a single SingleR configuration that selects "
     "the reference per dataset. The per-tier runs are retained in the repository."),
]


def _tex(text):
    """Escape the LaTeX specials that occur in model and dataset names."""
    return str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&")


def _fmt(value, digits=3):
    if pd.isna(value):
        return "--"
    return f"{value:.0f}" if digits == 0 else f"{value:.{digits}f}"


def build_frame(df):
    """Annotate the snapshot with the family / setting labels the tables are organized by.

    Models the paper excludes (EXCLUDED_BASES, as in the figures) are dropped here; the
    excluded-models table lists them with the reason.
    """
    out = df[~df["model"].map(model_base).isin(EXCLUDED_BASES)].copy()
    out["family"] = out["model"].map(lambda m: FAMILY.get(categorize_model(m), "other"))
    out["setting"] = out["model"].map(lambda m: REGIME.get(derive_mode(m), "--"))
    out["method"] = out["model"].map(lambda m: _display(model_base(m)))
    sort_cols = ["subset_size", "balanced_accuracy"]
    return out.sort_values(sort_cols, ascending=[True, False], na_position="last")


def write_csvs(frame, outdir):
    """One complete-metric CSV per dataset. Every snapshot column is kept."""
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    lead = ["dataset", "family", "method", "model", "setting", "subset_size"]
    for dataset in [d for d in DATASET_ORDER if d in set(frame["dataset"])]:
        sub = frame[frame["dataset"] == dataset]
        cols = lead + [c for c in sub.columns if c not in lead + ["mode"]]
        path = outdir / f"supp_results_{dataset}.csv"
        sub[cols].to_csv(path, index=False)
        written.append(path)
    return written


def write_tex(frame, out_path, csv_dir_name="tables", subset_size=1000):
    """One longtable per dataset plus the excluded-model table.

    Typeset at ONE subset size (the size every cross-family comparison in the main text uses),
    which also drops a constant column. The CSVs keep every size, so nothing is lost -- the
    caption points there.
    """
    frame = frame[frame["subset_size"] == subset_size]
    lines = [
        "% GENERATED by bin/supplementary_tables.py -- do not edit by hand.",
        "% Regenerate after any change to the aggregate snapshot so the tables cannot drift",
        "% from the figures. Requires \\usepackage{longtable} and \\usepackage{booktabs}.",
        "",
    ]
    header = ["Family", "Method", "Setting"] + [h for _c, h in TEX_COLUMNS]
    align = "lll" + "r" * len(TEX_COLUMNS)

    for dataset in [d for d in DATASET_ORDER if d in set(frame["dataset"])]:
        sub = frame[frame["dataset"] == dataset]
        n_methods = sub["model"].nunique()
        # A model that cannot process this dataset's organism never ran on it: say so.
        absent = sorted(_display(base) for base, ds in unsupported_pairs() if ds == dataset)
        absent_note = (f"{' and '.join(absent)} {'is' if len(absent) == 1 else 'are'} not listed: "
                       f"it handles human data only. " if absent else "")
        # NOTE: no "\\\\" inside a longtable caption -- the caption is itself a table row,
        # so a line break there ends the row and the table collapses with "Extra }".
        caption = (
            f"\\textbf{{Complete results for {_tex(DATASET_DISPLAY.get(dataset, dataset))}.}} "
            f"One row per method, {n_methods} in total, sorted by balanced accuracy. Every row is "
            f"a 1,000-cell subset, the size at which all methods were compared. {absent_note}"
            f"The best value in each score column is in bold. \\emph{{Setting}} is whether the cell "
            f"types expected in the dataset were supplied to the model. The marker- and "
            f"reference-based baselines and the chance floors carry a single score, so that "
            f"column is not applicable to them. \\emph{{non-CL}} is the fraction of cells whose "
            f"raw predicted label is not a Cell Ontology term. "
            f"\\emph{{Pred.\\ labels}} is how many distinct cell type labels the method produced "
            f"for this dataset, counted before ontology mapping. \\emph{{Agree.}} is the "
            f"agreement score (\\hyperref[s:methods]{{Methods}}): 1 for the true type, 0.5 for its "
            f"direct parent or child in the Cell Ontology, 0 otherwise. Rada and Lin are Cell "
            f"Ontology graph similarities. Macro F1 averages over the true cell types. "
            f"A dash marks a metric the coverage gate left unscored. The complete metric set for "
            f"this dataset, including plain accuracy, the text-encoder scores and the 10,000-cell "
            f"runs, is in "
            f"\\texttt{{{csv_dir_name}/supp\\_results\\_{_tex(dataset)}.csv}}."
        )
        lines += [
            "\\footnotesize",
            f"\\begin{{longtable}}{{{align}}}",
            f"\\caption{{{caption}}}\\label{{supptab:results_{dataset.lower()}}}\\\\",
            "\\toprule",
            " & ".join(header) + " \\\\", "\\midrule", "\\endfirsthead",
            "\\multicolumn{%d}{l}{\\emph{%s, continued}}\\\\"
            % (len(header), _tex(DATASET_DISPLAY.get(dataset, dataset))),
            "\\toprule", " & ".join(header) + " \\\\", "\\midrule", "\\endhead",
            "\\midrule", "\\multicolumn{%d}{r}{\\emph{continued on the next page}}\\\\" % len(header),
            "\\endfoot", "\\bottomrule", "\\endlastfoot",
        ]
        # Best value per score column, compared on the PRINTED string rather than the stored
        # float. Two methods whose scores differ in the fifth decimal print identically, and
        # bolding only one of them would read as a typesetting error.
        best = {}
        for col, _h in TEX_COLUMNS:
            if col not in BOLD_MAX:
                continue
            values = pd.to_numeric(sub[col], errors="coerce")
            if values.notna().any():
                best[col] = _fmt(values.max(), 3)

        for _, r in sub.iterrows():
            cells = [_tex(FAMILY_TEX.get(r["family"], r["family"])),
                     _tex(r["method"]), _tex(r["setting"])]
            for col, _h in TEX_COLUMNS:
                text = _fmt(r.get(col), 0 if col == "n_unique_pred_raw" else 3)
                if text != "--" and best.get(col) == text:
                    text = f"\\textbf{{{text}}}"
                cells.append(text)
            lines.append(" & ".join(cells) + " \\\\")
        lines += ["\\end{longtable}", ""]

    lines += [
        "\\begin{table}[!ht]", "\\centering", "\\footnotesize",
        "\\begin{tabular}{p{4.2cm}p{10.5cm}}", "\\toprule",
        "Method & Reason for exclusion \\\\", "\\midrule",
    ]
    for name, reason in EXCLUDED:
        lines.append(f"{_tex(name)} & {reason} \\\\[2pt]")
    lines += [
        "\\bottomrule", "\\end{tabular}",
        "\\caption{\\textbf{Methods that were run and then excluded from the analysis.}\\\\ "
        "Each was executed by the pipeline and its predictions are retained in the repository, "
        "but it is not included in the figures or in the tables above. Listing them here records "
        "the exclusions rather than leaving them to be inferred from the model roster.}",
        "\\label{supptab:excluded}", "\\end{table}", "",
    ]
    out_path.write_text("\n".join(lines))
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="Results/aggregated_by_dataset_model.csv")
    p.add_argument("--outdir", default="scLLMs_benchmark_paper/bioRxiv_template")
    p.add_argument("--subset-size", type=int, default=1000,
                   help="cells per method in the TYPESET tables; the CSVs keep every size")
    args = p.parse_args(argv)

    frame = build_frame(pd.read_csv(args.input))
    outdir = Path(args.outdir)
    csvs = write_csvs(frame, outdir / "tables")
    tex = write_tex(frame, outdir / "supplementary_tables_generated.tex",
                    subset_size=args.subset_size)
    print(f"{len(frame)} rows, {frame['dataset'].nunique()} datasets, "
          f"{frame['model'].nunique()} methods")
    for path in csvs:
        print(f"  csv -> {path}")
    print(f"  tex -> {tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
