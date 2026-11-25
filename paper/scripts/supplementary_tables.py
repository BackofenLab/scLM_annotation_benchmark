#!/usr/bin/env python
"""Generate the per-dataset supplementary result tables from the aggregate snapshot.

Three outputs, deliberately different in scope:

* ``tables/supp_results_<dataset>.csv`` -- EVERY metric column the snapshot carries, at every
  subset size. This is the artifact a reader reuses, and the one converted to the Word/Excel file
  BiB asks for at submission.
* ``supplementary_tables_generated.tex`` -- one landscape longtable per dataset, with a block per
  subset size: every metric of Supplementary Figure S3, in its groups, and the diagnostics the
  paper reads (how much of the output needs the curated mapping, the score without it, and how
  many distinct labels a method gives). The caption sits below the table.
* ``supplementary_unlisted_table_generated.tex`` -- for every constrained run of a generative
  model, the share of answers that name a cell type not on the list, per dataset.

Nothing here is hand-written, so the tables cannot drift from the figures (style Rule 6).

Usage:
    python bin/supplementary_tables.py --input cache/result/aggregated_by_dataset_model_<v>.csv
"""
from __future__ import annotations

import argparse
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from manuscript_figures import (  # noqa: E402
    DATASET_ORDER, METRIC_GROUPS, _benchmark_pairs, _display, _mapping_scores, _needs_mapping,
    _unlisted)
from plot_model_comparison import (  # noqa: E402
    EXCLUDED_BASES, REGIME, REGIME_SHORT, categorize_model, derive_mode,
    load_confusion_pairs, model_base, model_tier, unsupported_pairs)

# The one chance model the paper describes: Figure 2's dashed line. The pipeline's other chance
# floors (uniform random, always "cell") are described nowhere, so they are left out.
CHANCE_MODELS = {"random_stratified"}
# Figure 2G and Supplementary Figure S1 per run, and the share of answers outside the list of a
# constrained run; computed here from the confusion pairs of each subset size
# (bin/per_class_scores.py). In the CSVs each sits next to the snapshot column it qualifies.
NO_MAPPING_COLUMN = "balanced_accuracy_no_curated_mapping"
NEEDS_MAPPING_COLUMN = "share_needing_curated_mapping"
UNLISTED_COLUMN = "share_unlisted"
DERIVED_AFTER = {NO_MAPPING_COLUMN: "balanced_accuracy", NEEDS_MAPPING_COLUMN: "non_cl_ratio",
                 UNLISTED_COLUMN: "n_unique_pred_mapped"}
# The unlisted-answers table shows the generative models, which alone can answer outside the
# list, at the size every method ran at.
UNLISTED_SIZE = 1000
# Column headers of the typeset table: Supplementary Figure S3's metrics in its groups, then the
# diagnostics. Every metric is higher-is-better, so its best value is bolded; the diagnostics
# describe the output rather than rank it.
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
FAMILY = {"LLM": "general-purpose LLM", "scLLM": "scLLM",
          "classical": "baseline", "simple": "chance floor"}
FAMILY_PLURAL = {"general-purpose LLM": "general-purpose LLMs", "scLLM": "scLLMs",
                 "baseline": "marker- and reference-based baselines"}
# Dataset names for anything a READER sees. The manuscript uses the pipeline's keys only in
# Methods, where they let someone move between the paper and the repository; everywhere else, and
# in every caption, the readable name is used. Only human_pancreas differs -- the other three keys
# are already the names.
DATASET_DISPLAY = {"human_pancreas": "human pancreas"}
# Short forms for the typeset table only; the CSVs keep the long form, where width costs nothing.
FAMILY_TEX = {"general-purpose LLM": "LLM", "scLLM": "scLLM",
              "baseline": "baseline", "chance floor": "chance"}
# The typeset table writes the setting as the paper's U and C; the CSVs spell it out.
SETTING_TEX = {REGIME[mode]: short for mode, short in REGIME_SHORT.items()}


def _tex(text):
    """Escape the LaTeX specials that occur in model and dataset names."""
    return str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&")


def _fmt(value, digits=3):
    """``value`` rounded half up as the decimal number it is stored as. The snapshot keeps four
    decimals, and binary rounding would print a stored 0.3495 as 0.349."""
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
    # Four decimals, as the snapshot stores every metric, so a run whose answers need no curated
    # mapping prints the same balanced accuracy in both columns.
    return pd.concat(frames, ignore_index=True).round(dict.fromkeys(DERIVED_AFTER, 4))


def build_frame(df):
    """Annotate the snapshot with the family / setting labels the tables are organized by.

    Models the paper excludes (EXCLUDED_BASES, as in the figures) are dropped here and are not
    mentioned anywhere, and so are the chance floors other than CHANCE_MODELS. The chance model
    runs in one setting, "--". Adds the two columns of DERIVED_AFTER.
    """
    out = df[~df["model"].map(model_base).isin(EXCLUDED_BASES)].copy()
    out = out[(out["model"].map(categorize_model) != "simple") | out["model"].isin(CHANCE_MODELS)]
    out["family"] = out["model"].map(lambda m: FAMILY.get(categorize_model(m), "other"))
    out["setting"] = out["model"].map(lambda m: REGIME.get(derive_mode(m), "--"))
    out["method"] = out["model"].map(lambda m: _display(model_base(m)))
    out = out.merge(_derived(out), on=["model", "dataset", "subset_size"], how="left")
    sort_cols = ["subset_size", "balanced_accuracy"]
    return out.sort_values(sort_cols, ascending=[True, False], na_position="last")


def write_csvs(frame, outdir):
    """One complete-metric CSV per dataset. Every snapshot column is kept."""
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    lead = ["dataset", "family", "method", "model", "setting", "subset_size"]
    for dataset in [d for d in DATASET_ORDER if d in set(frame["dataset"])]:
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
    name = _tex(DATASET_DISPLAY.get(dataset, dataset))
    # A model that cannot process this dataset's organism never ran on it: say so.
    absent = sorted(_display(base) for base, ds in unsupported_pairs() if ds == dataset)
    absent_note = (f"{' and '.join(absent)} {'is' if len(absent) == 1 else 'are'} not listed "
                   f"because the pipeline does not run "
                   f"{'it' if len(absent) == 1 else 'them'} on this organism. " if absent else "")
    coverage = sub["reviewed_fraction"].min()
    coverage_note = ("Every answer was a Cell Ontology term or had a reviewed mapping."
                     if coverage == 1 else
                     f"At least {coverage:.3f} of the answers of every run were Cell Ontology "
                     f"terms or had a reviewed mapping, and the others count as wrong.")
    counts = _join([f"{sub.loc[sub['subset_size'] == size, 'model'].nunique()} at {_cells(size)}"
                    for size in sizes])
    head = (f"\\textbf{{Benchmark results for {name}.}} One row per method and setting, {counts}, "
            f"sorted by balanced accuracy within each size. {_only_smallest(sub, sizes)}"
            f"{absent_note}")
    if dataset != first:
        return (head + f"Columns, abbreviations and bold type as in Supplementary "
                f"Table~\\ref{{supptab:results_{first.lower()}}}. {coverage_note}")
    # The columns from left to right; Supplementary Table S3 defines the metrics.
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

    datasets = [d for d in DATASET_ORDER if d in set(frame["dataset"])]
    for dataset in datasets:
        sub = frame[frame["dataset"] == dataset]
        sizes = sorted(sub["subset_size"].unique())
        caption = _caption(dataset, sub, sizes, datasets[0])
        # NOTE: no "\\\\" inside a longtable caption -- the caption is itself a table row,
        # so a line break there ends the row and the table collapses with "Extra }".
        lines += [
            "\\begin{landscape}",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\captionsetup{position=below}",
            f"\\begin{{longtable}}{{lll{'r' * len(columns)}}}",
            *header, "\\endfirsthead",
            "\\multicolumn{%d}{l}{\\emph{%s, continued}}\\\\"
            % (n_cols, _tex(DATASET_DISPLAY.get(dataset, dataset))),
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
            # Best value per metric and size, compared on the PRINTED string rather than the
            # stored float. Two methods whose scores differ in the fifth decimal print
            # identically, and bolding only one of them would read as a typesetting error.
            best = {}
            for col in BOLD_MAX:
                values = pd.to_numeric(block[col], errors="coerce")
                if values.notna().any():
                    best[col] = _fmt(values.max(), 3)
            for _, r in block.iterrows():
                cells = [_tex(FAMILY_TEX.get(r["family"], r["family"])),
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
    datasets = [d for d in DATASET_ORDER if d in shares.columns]
    shares = (shares[datasets].assign(mean=shares[datasets].mean(axis=1))
              .sort_values("mean", ascending=False, kind="stable"))
    header = " & ".join(["Family", "Method", *[_tex(DATASET_DISPLAY.get(d, d)) for d in datasets],
                         "Mean"]) + " \\\\"
    rows = [" & ".join([_tex(FAMILY_TEX[family]), _tex(method), *map(_fmt, values)]) + " \\\\"
            for (family, method), values in zip(shares.index, shares.to_numpy())]
    caption = (
        "\\textbf{Answers outside the list of expected cell types.} One row per generative model, "
        "the general-purpose LLMs and the generative scLLMs, in its constrained run at "
        f"{_cells(UNLISTED_SIZE)}, sorted by the mean. \\emph{{Family}} and \\emph{{Method}} name "
        "the method. Each dataset column gives the share of cells whose answer named a cell type "
        "that is not on the list, and \\emph{Mean} averages the four datasets. Answers are "
        "compared with the listed types after the curated mapping, so another name for a listed "
        "type counts as listed, and an answer that names no cell type does not count as outside "
        "the list. Shares are averaged over cell types, as balanced accuracy is.")
    lines = ["% GENERATED by bin/supplementary_tables.py -- do not edit by hand.", "",
             "\\begin{table}[!ht]", "\\centering", "\\footnotesize",
             f"\\begin{{tabular}}{{ll{'r' * (len(datasets) + 1)}}}", "\\toprule", header,
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
