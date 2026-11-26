#!/usr/bin/env python
"""Generate the per-dataset result tables of the supplement from the benchmark metrics.

Writes to --outdir:

* ``tables/supp_results_<dataset>.csv``: every column of the benchmark metrics at every subset
  size, with the columns of DERIVED_AFTER;
* ``supplementary_tables_generated.tex``: one landscape longtable per dataset with one block per
  subset size, holding the metrics of METRIC_GROUPS in their groups and the diagnostics of the
  curated mapping and of the distinct labels, with the caption below the table;
* ``supplementary_unlisted_table_generated.tex``: the share of answers that name a cell type not
  on the list, for the constrained run of every generative method, per dataset.

    python paper/scripts/supplementary_tables.py [--input CSV] [--outdir DIR]
"""
from __future__ import annotations

import argparse
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

import paths
from methods import (DATASET_ORDER, METRIC_GROUPS, REGIME, REGIME_SHORT, _benchmark_pairs,
                     _display, _mapping_scores, _needs_mapping, _unlisted, categorize_model,
                     dataset_label, derive_mode, load_confusion_pairs, model_base, model_tier)

# Columns computed from the confusion pairs of each subset size: balanced accuracy without the
# curated mapping, the share of cells whose answer needs it, and for a constrained run the share
# of answers outside the list. In the CSVs each follows the column it qualifies.
NO_MAPPING_COLUMN = "balanced_accuracy_no_curated_mapping"
NEEDS_MAPPING_COLUMN = "share_needing_curated_mapping"
UNLISTED_COLUMN = "share_unlisted"
DERIVED_AFTER = {NO_MAPPING_COLUMN: "balanced_accuracy", NEEDS_MAPPING_COLUMN: "non_cl_ratio",
                 UNLISTED_COLUMN: "n_unique_pred_mapped"}
# The unlisted-answers table shows the generative models, which alone can answer outside the
# list, at the size every method ran at.
UNLISTED_SIZE = 1000
# Column headers of the typeset table: the metrics in their groups, then the diagnostics. Every
# metric is higher-is-better and its best value is bold; the diagnostics describe the output.
METRIC_HEADERS = {
    "balanced_accuracy": "Bal.\\ acc.", "accuracy": "Acc.", "macro_f1": "Macro F1",
    "weighted_f1": "Wtd.\\ F1", "agreement_score": "Agree.", "rada_avg_similarity": "Rada",
    "lin_avg_similarity": "Lin", "text_emb_norm_balanced_mapped": "Mapped",
    "text_emb_norm_balanced_raw": "Raw"}
TEX_GROUPS = [(group.capitalize(), [(key, METRIC_HEADERS[key]) for key, _name in metrics])
              for group, metrics in METRIC_GROUPS] + [
    ("Curated mapping", [(NEEDS_MAPPING_COLUMN, "Needed"),
                         (NO_MAPPING_COLUMN, "\\shortstack[r]{Bal.\\ acc.\\\\without}")]),
    ("Distinct labels", [("n_unique_pred_mapped", "Mapped"), ("n_unique_pred_raw", "Raw")])]
BOLD_MAX = {key for _group, metrics in METRIC_GROUPS for key, _name in metrics}
COUNT_COLUMNS = {"n_unique_pred_mapped", "n_unique_pred_raw"}
FAMILY = {"LLM": "general-purpose LLM", "scLLM": "scLM",
          "classical": "baseline", "chance": "chance floor"}
FAMILY_PLURAL = {"general-purpose LLM": "general-purpose LLMs", "scLM": "scLMs",
                 "baseline": "marker- and reference-based baselines"}
# Short forms for the typeset table; the CSVs keep the long ones.
FAMILY_TEX = {"general-purpose LLM": "LLM", "scLM": "scLM",
              "baseline": "baseline", "chance floor": "chance"}
# The typeset table writes the setting as U or C; the CSVs spell it out.
SETTING_TEX = {REGIME[mode]: short for mode, short in REGIME_SHORT.items()}


def _tex(text):
    """Escape the LaTeX specials that occur in model and dataset names."""
    return str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&")


def _fmt(value, digits=3):
    """``value`` rounded half up as the decimal number it is stored as. The benchmark metrics keep
    four decimals, and binary rounding would print a stored 0.3495 as 0.349."""
    if pd.isna(value):
        return "--"
    return str(Decimal(str(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def _cells(size):
    return f"{size:,} cells"


def _join(names):
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _derived(out):
    """DERIVED_AFTER's columns per run, dataset and subset size, from the confusion pairs of that
    size; a size without them is left empty, and so is the unlisted share of a run without list."""
    frames = []
    for size in sorted(out["subset_size"].unique()):
        pairs = _benchmark_pairs(out[out["subset_size"] == size],
                                 load_confusion_pairs(subset_size=size), chance=True)
        constrained = pairs[pairs["model"].map(derive_mode) == "unblind"]
        frames.append(pd.concat([_mapping_scores(pairs)["without"].rename(NO_MAPPING_COLUMN),
                                 _needs_mapping(pairs).rename(NEEDS_MAPPING_COLUMN),
                                 _unlisted(constrained).rename(UNLISTED_COLUMN)], axis=1)
                      .reset_index().assign(subset_size=size))
    # Four decimals, as the benchmark metrics are stored, so that a run whose answers need no
    # curated mapping prints the same balanced accuracy in both columns.
    return pd.concat(frames, ignore_index=True).round(dict.fromkeys(DERIVED_AFTER, 4))


def build_frame(df):
    """The benchmark metrics with the family, setting and name of every run's method and the
    columns of DERIVED_AFTER, sorted by size and balanced accuracy. The chance model has the
    setting "--"."""
    out = df.copy()
    out["family"] = out["model"].map(lambda m: FAMILY[categorize_model(m)])
    out["setting"] = out["model"].map(lambda m: REGIME.get(derive_mode(m), "--"))
    out["method"] = out["model"].map(lambda m: _display(model_base(m)))
    out = out.merge(_derived(out), on=["model", "dataset", "subset_size"], how="left")
    sort_cols = ["subset_size", "balanced_accuracy"]
    return out.sort_values(sort_cols, ascending=[True, False], na_position="last")


def write_csvs(frame, outdir):
    """One CSV per dataset with every column of the benchmark metrics."""
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    lead = ["dataset", "family", "method", "model", "setting", "subset_size"]
    for dataset in DATASET_ORDER:
        sub = frame[frame["dataset"] == dataset]
        cols = [c for c in sub.columns if c not in lead + ["mode", *DERIVED_AFTER]]
        for column, after in DERIVED_AFTER.items():
            cols.insert(cols.index(after) + 1, column)
        path = outdir / f"supp_results_{dataset}.csv"
        sub[lead + cols].to_csv(path, index=False)
        written.append(path)
    return written


def _only_smallest(sub, sizes):
    """Name the runs at the smallest size that the larger sizes lack, a whole family by name."""
    small = sub[sub["subset_size"] == sizes[0]]
    missing = small[~small["model"].isin(sub.loc[sub["subset_size"] != sizes[0], "model"])]
    names = []
    for family, group in missing.groupby("family", sort=False):
        if len(group) == (small["family"] == family).sum():
            names.append(f"the {FAMILY_PLURAL.get(family, family)}")
        else:
            names += [f"{_tex(r.method)} ({SETTING_TEX.get(r.setting, r.setting)})"
                      for r in group.itertuples()]
    if not names:
        return ""
    text = _join(names)
    return f"{text[0].upper()}{text[1:]} ran at {_cells(sizes[0])} only. "


def _caption(dataset, sub, sizes, first):
    """The caption of the first table, of dataset ``first``, defines every column; the others
    refer to it."""
    name = _tex(dataset_label(dataset))
    coverage = sub["reviewed_fraction"].min()
    coverage_note = ("Every answer was a Cell Ontology term or had a reviewed mapping."
                     if coverage == 1 else
                     f"At least {coverage:.3f} of the answers of every run were Cell Ontology "
                     f"terms or had a reviewed mapping, and the others count as wrong.")
    counts = _join([f"{sub.loc[sub['subset_size'] == size, 'model'].nunique()} at {_cells(size)}"
                    for size in sizes])
    head = (f"\\textbf{{Benchmark results for {name}.}} One row per method and setting, {counts}, "
            f"sorted by balanced accuracy within each size. {_only_smallest(sub, sizes)}")
    if dataset != first:
        return (head + f"Columns, abbreviations and bold type as in Supplementary "
                f"Table~\\ref{{supptab:results_{first.lower()}}}. {coverage_note}")
    # The columns from left to right.
    return head + (
        "\\emph{Family} and \\emph{Method} name the method of a run, and \\emph{Setting} is C for "
        "a run constrained by the list of expected cell types and U for a run without it. The "
        "random model, which assigns cell types in proportion to their frequency, runs in one "
        "setting. The best value of each metric at each size is highlighted in bold. "
        "\\emph{Bal.\\ acc.} is balanced accuracy, \\emph{Acc.} accuracy, \\emph{Wtd.\\ F1} "
        "weighted F1 and \\emph{Agree.} the agreement score (Supplementary "
        "Table~\\ref{supptab:metrics}). Text similarity uses the answer after the curated mapping "
        "(\\emph{Mapped}) or as written (\\emph{Raw}). \\emph{Needed} is the share of cells whose "
        "answer needs the curated mapping, and \\emph{Bal.\\ acc.\\ without} is the balanced "
        "accuracy when only the answers that need no curated mapping count as correct. "
        "\\emph{Distinct labels} counts the cell types named after the curated mapping "
        "(\\emph{Mapped}) and the different answers as written (\\emph{Raw}). "
        f"{coverage_note}")


def write_tex(frame, out_path):
    """One landscape longtable per dataset, one block per subset size, the caption below."""
    lines = [
        "% GENERATED by bin/supplementary_tables.py -- do not edit by hand.",
        "% Regenerate after any change to the aggregate snapshot so the tables cannot drift",
        "% from the figures. Requires \\usepackage{longtable}, \\usepackage{booktabs} and",
        "% \\usepackage{pdflscape}.",
        "",
    ]
    columns = [column for _group, group in TEX_GROUPS for column, _header in group]
    groups, rules, start = ["", "", ""], [], 4
    for group, members in TEX_GROUPS:
        groups.append(f"\\multicolumn{{{len(members)}}}{{c}}{{{group}}}")
        rules.append(f"\\cmidrule(lr){{{start}-{start + len(members) - 1}}}")
        start += len(members)
    header = ["\\toprule", " & ".join(groups) + " \\\\", "".join(rules),
              " & ".join(["Family", "Method", "Setting"]
                         + [h for _group, group in TEX_GROUPS for _c, h in group]) + " \\\\",
              "\\midrule"]
    n_cols = 3 + len(columns)

    for dataset in DATASET_ORDER:
        sub = frame[frame["dataset"] == dataset]
        sizes = sorted(sub["subset_size"].unique())
        caption = _caption(dataset, sub, sizes, DATASET_ORDER[0])
        # A longtable caption is itself a table row, so it holds no line break.
        lines += [
            "\\begin{landscape}",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\captionsetup{position=below}",
            f"\\begin{{longtable}}{{lll{'r' * len(columns)}}}",
            *header, "\\endfirsthead",
            "\\multicolumn{%d}{l}{\\emph{%s, continued}}\\\\"
            % (n_cols, _tex(dataset_label(dataset))),
            *header, "\\endhead",
            "\\midrule", "\\multicolumn{%d}{r}{\\emph{continued on the next page}}\\\\" % n_cols,
            "\\endfoot", "\\bottomrule",
            f"\\caption{{{caption}}}\\label{{supptab:results_{dataset.lower()}}}\\\\",
            "\\endlastfoot",
        ]
        for i, size in enumerate(sizes):
            block = sub[sub["subset_size"] == size]
            # Each size starts a page, so that no block is split over two pages.
            lines += ([] if i == 0 else ["\\pagebreak"]) + [
                "\\multicolumn{%d}{l}{\\textbf{%s}}\\\\" % (n_cols, _cells(size))]
            # The best value per metric and size, compared as printed, so that every run that
            # prints it is bold.
            best = {}
            for col in BOLD_MAX:
                values = pd.to_numeric(block[col], errors="coerce")
                if values.notna().any():
                    best[col] = _fmt(values.max(), 3)
            for _, r in block.iterrows():
                cells = [_tex(FAMILY_TEX[r["family"]]),
                         _tex(r["method"]), _tex(SETTING_TEX.get(r["setting"], r["setting"]))]
                for col in columns:
                    text = _fmt(r.get(col), 0 if col in COUNT_COLUMNS else 3)
                    if text != "--" and best.get(col) == text:
                        text = f"\\textbf{{{text}}}"
                    cells.append(text)
                lines.append(" & ".join(cells) + " \\\\")
        lines += ["\\end{longtable}", "\\end{landscape}", ""]

    out_path.write_text("\n".join(lines))
    return out_path


def write_unlisted_tex(frame, out_path):
    """The unlisted share of every generative model's constrained run at UNLISTED_SIZE: one row
    per method, one column per dataset and their mean, sorted by the mean."""
    runs = frame[(frame["subset_size"] == UNLISTED_SIZE) & (frame["setting"] == REGIME["unblind"])
                 & frame["model"].map(lambda m: categorize_model(m) == "LLM"
                                      or model_tier(m) == "generative")]
    shares = runs.pivot_table(index=["family", "method"], columns="dataset",
                              values=UNLISTED_COLUMN)
    shares = (shares[DATASET_ORDER].assign(mean=shares[DATASET_ORDER].mean(axis=1))
              .sort_values("mean", ascending=False, kind="stable"))
    header = " & ".join(["Family", "Method", *[_tex(dataset_label(d)) for d in DATASET_ORDER],
                         "Mean"]) + " \\\\"
    rows = [" & ".join([_tex(FAMILY_TEX[family]), _tex(method), *map(_fmt, values)]) + " \\\\"
            for (family, method), values in zip(shares.index, shares.to_numpy())]
    caption = (
        "\\textbf{Answers outside the list of expected cell types.} One row per generative model, "
        "the general-purpose LLMs and the generative scLMs, in its constrained run at "
        f"{_cells(UNLISTED_SIZE)}, sorted by the mean. \\emph{{Family}} and \\emph{{Method}} name "
        "the method. Each dataset column gives the share of cells whose answer named a cell type "
        "that is not on the list, and \\emph{Mean} averages the four datasets. Answers are "
        "compared with the listed types after the curated mapping, so another name for a listed "
        "type counts as listed, and an answer that names no cell type does not count as outside "
        "the list. Shares are averaged over cell types, as balanced accuracy is.")
    lines = ["% GENERATED by bin/supplementary_tables.py -- do not edit by hand.", "",
             "\\begin{table}[!ht]", "\\centering", "\\footnotesize",
             f"\\begin{{tabular}}{{ll{'r' * (len(DATASET_ORDER) + 1)}}}", "\\toprule", header,
             "\\midrule", *rows, "\\bottomrule", "\\end{tabular}",
             "\\captionsetup{position=below}", f"\\caption{{{caption}}}",
             "\\label{supptab:unlisted}", "\\end{table}", ""]
    out_path.write_text("\n".join(lines))
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(paths.BENCHMARK_METRICS))
    p.add_argument("--outdir", default=str(paths.TABLES))
    args = p.parse_args(argv)

    frame = build_frame(pd.read_csv(args.input))
    outdir = Path(args.outdir)
    csvs = write_csvs(frame, outdir / "tables")
    texs = [write_tex(frame, outdir / "supplementary_tables_generated.tex"),
            write_unlisted_tex(frame, outdir / "supplementary_unlisted_table_generated.tex")]
    print(f"{len(frame)} rows, {frame['dataset'].nunique()} datasets, "
          f"{frame['model'].nunique()} methods")
    for path in csvs:
        print(f"  csv -> {path}")
    for path in texs:
        print(f"  tex -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
