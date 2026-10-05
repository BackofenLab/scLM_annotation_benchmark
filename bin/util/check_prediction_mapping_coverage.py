#!/usr/bin/env python
"""
Test/utility script that checks whether all predicted cell type labels found in
the prediction CSVs can be mapped to an ontology term.

For every unique predicted label collected from
``cache/models/predictions/**/*.csv`` the script applies the *same* mapping logic
used in evaluation (see :mod:`util.ontology_helpers` and
:mod:`util.ontology_mapper`):

1. If the label maps directly to a Cell Ontology term
   (``map_string_to_term``), it is considered mapped.
2. Otherwise the label is looked up in
   ``cache/cell_type_ontology_mappings.csv``. If a non-empty ``chosen_match`` is
   present and that ``chosen_match`` itself maps to an ontology term, the label
   is considered mapped (this mirrors
   ``ontology_helpers.apply_ontology_mappings_to_cell_types``).
3. Otherwise the label is reported as *unmapped*.

The script prints the number of labels that cannot be mapped and, optionally, the
full list together with the affected prediction files.

Usage (run from the ``bin`` directory or anywhere - the script fixes up the path
itself)::

    python bin/util/check_prediction_mapping_coverage.py
    python bin/util/check_prediction_mapping_coverage.py --print-unmapped
    python bin/util/check_prediction_mapping_coverage.py \
        --predictions-dir cache/models/predictions \
        --cache-dir cache \
        --pred-cols predicted_labels true_labels

Exit code is 0 if everything maps, 1 if any label cannot be mapped (useful for
CI / pre-commit checks).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

# When run directly (``python bin/util/check_prediction_mapping_coverage.py``)
# Python puts this file's directory (``bin/util``) at the front of sys.path,
# which shadows stdlib modules such as ``http`` with the local ``http.py``.
# Drop it so imports resolve correctly.
_THIS_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", ".", _THIS_DIR)]

# Make ``util`` importable as a package regardless of the current working
# directory (the file lives in ``bin/util/``; its parent is ``bin``).
_BIN_DIR = Path(__file__).resolve().parent.parent
if str(_BIN_DIR) not in sys.path:
    sys.path.insert(0, str(_BIN_DIR))

from util.cell_ontology_similarity import (  # noqa: E402
    build_label_index,
    download_cl_ontology,
    map_string_to_term,
)
from util.ontology_helpers import check_ontology_mapping_coverage  # noqa: E402
from util.ontology_mapper import load_ontology_mappings  # noqa: E402
from util.label_normalization import (  # noqa: E402
    normalize_predicted_labels,
    normalize_true_labels,
)

# Repo root is two levels above ``bin`` (``bin/util`` -> ``bin`` -> repo root).
_REPO_ROOT = _BIN_DIR.parent
_DEFAULT_PREDICTIONS_DIR = _REPO_ROOT / "cache" / "models" / "predictions"
_DEFAULT_CACHE_DIR = _REPO_ROOT / "cache"


def collect_labels_from_predictions(
    predictions_dir: Path,
    pred_cols: List[str],
    exclude: List[str] | None = None,
    run: str | None = None,
    verbose: bool = True,
) -> Dict[str, Set[Path]]:
    """
    Collect all unique labels from the prediction CSV files.

    Parameters
    ----------
    predictions_dir : Path
        Directory containing prediction CSVs (searched recursively).
    pred_cols : List[str]
        Candidate column names that hold predicted labels. The first matching
        column present in a given CSV is used.
    exclude : List[str], optional
        If any of these substrings appears in a CSV file's path (relative to
        ``predictions_dir``), that file is skipped. Matching is case-insensitive.
    run : str, optional
        If given, only include files whose path has this run subfolder as a path
        component (e.g. ``1000_stratified`` in
        ``model/dataset/1000_stratified/predictions.csv``).
    verbose : bool
        If True, print progress information.

    Returns
    -------
    Dict[str, Set[Path]]
        Mapping of unique label -> set of files in which it appears.
    """
    exclude = exclude or []
    exclude_lower = [e.lower() for e in exclude]

    all_csv_files = sorted(predictions_dir.rglob("*.csv"))

    csv_files = []
    skipped_exclude = 0
    skipped_run = 0
    for f in all_csv_files:
        rel_path = f.relative_to(predictions_dir)
        rel = str(rel_path).lower()
        if run is not None and run not in rel_path.parts:
            skipped_run += 1
            continue
        if any(e in rel for e in exclude_lower):
            skipped_exclude += 1
            continue
        csv_files.append(f)

    if verbose:
        print(f"Found {len(all_csv_files)} CSV files under {predictions_dir}")
        if run is not None:
            print(f"Skipped {skipped_run} files not in run subfolder '{run}'")
        if exclude:
            print(f"Excluded {skipped_exclude} files matching {exclude}")
        print(f"Checking {len(csv_files)} files")

    label_to_files: Dict[str, Set[Path]] = defaultdict(set)

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:  # pragma: no cover - defensive
            print(f"  WARNING: could not read {csv_file}: {e}")
            continue

        cols_present = [c for c in pred_cols if c in df.columns]
        if not cols_present:
            if verbose:
                print(
                    f"  WARNING: none of {pred_cols} found in {csv_file} "
                    f"(columns: {list(df.columns)})"
                )
            continue

        for col in cols_present:
            # Apply the exact same normalization the evaluation workflow uses
            # before ontology lookup, so coverage reflects real behavior.
            # True-label columns get the lighter strip+lower treatment;
            # everything else is treated as a predicted-label column.
            if "true" in col.lower():
                normalized = normalize_true_labels(df[col])
            else:
                normalized = normalize_predicted_labels(df[col])

            for label in normalized.dropna().unique():
                label = str(label)
                if label:
                    label_to_files[label].add(csv_file)

    return label_to_files


def check_labels_mappable(
    labels: List[str],
    cache_dir: Path,
    verbose: bool = True,
):
    """
    Determine which labels can be mapped to an ontology term.

    Reuses the exact mapping logic from the evaluation pipeline:
    direct ontology lookup first, then the ``chosen_match`` from the ontology
    mappings CSV (which must itself map to an ontology term).

    Parameters
    ----------
    labels : List[str]
        Unique labels to check.
    cache_dir : Path
        Cache directory holding the ontology file(s) and the
        ``cell_type_ontology_mappings.csv`` file.
    verbose : bool
        If True, print progress information.

    Returns
    -------
    Tuple[Set[str], Dict[str, str]]
        - Set of labels that cannot be mapped.
        - Dict (label -> how it was mapped) for the labels that *can* be mapped.
    """
    # Ensure the ontology is available locally; reuse the shared downloader.
    cl_file_path = download_cl_ontology(cache_dir)

    # Step 1: direct ontology coverage (reuses shared helper which builds the
    # label index and runs map_string_to_term for every label).
    mapped_directly, unmapped_after_direct, label_index = check_ontology_mapping_coverage(
        cell_types=labels,
        ontology_path=str(cl_file_path),
    )

    if verbose:
        print(f"  Directly mappable to ontology: {len(mapped_directly)}")
        print(f"  Not directly mappable: {len(unmapped_after_direct)}")

    # Step 2: try to resolve the remaining labels via the mappings CSV.
    mappings = load_ontology_mappings(cache_dir)
    if mappings is None:
        if verbose:
            print(
                f"  WARNING: no ontology mappings CSV found in {cache_dir}; "
                f"only direct ontology matches count as mapped."
            )
        mappings = {}

    how_mapped: Dict[str, str] = {label: "ontology (direct)" for label in mapped_directly}
    unmapped: Set[str] = set()

    for label in unmapped_after_direct:
        chosen_match = mappings.get(label)
        # pandas may yield NaN for empty cells -> treat as missing.
        if isinstance(chosen_match, float) and pd.isna(chosen_match):
            chosen_match = None
        if chosen_match is not None:
            chosen_match = str(chosen_match).strip()

        if chosen_match and map_string_to_term(chosen_match, label_index) is not None:
            how_mapped[label] = f"mappings.csv -> '{chosen_match}'"
        else:
            unmapped.add(label)

    return unmapped, how_mapped


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check that all predicted labels in the prediction CSVs can be "
            "mapped to an ontology term (directly or via "
            "cell_type_ontology_mappings.csv)."
        )
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=_DEFAULT_PREDICTIONS_DIR,
        help=f"Directory with prediction CSVs (default: {_DEFAULT_PREDICTIONS_DIR})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_DEFAULT_CACHE_DIR,
        help=f"Cache dir with ontology + mappings CSV (default: {_DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--pred-cols",
        nargs="+",
        default=["predicted_labels"],
        help=(
            "Column name(s) holding predicted labels to check "
            "(default: predicted_labels). Pass e.g. 'predicted_labels "
            "true_labels' to also check the ground-truth labels."
        ),
    )
    parser.add_argument(
        "--run",
        default="1000_stratified",
        help=(
            "Only include prediction files whose path contains this run "
            "subfolder (default: 1000_stratified). Pass --run '' to include all "
            "runs."
        ),
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=["llama-3.1-8b"],
        help=(
            "Skip prediction files whose path contains any of these substrings "
            "(case-insensitive). Default: llama-3.1-8b. Pass --exclude '' to "
            "disable exclusion."
        ),
    )
    parser.add_argument(
        "--print-unmapped",
        action="store_true",
        help="Print the full list of labels that cannot be mapped.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Write the unmapped labels to this CSV file (columns: "
            "predicted_label, n_files, example_file)."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce progress output.",
    )
    args = parser.parse_args(argv)

    verbose = not args.quiet

    if not args.predictions_dir.exists():
        print(f"ERROR: predictions dir does not exist: {args.predictions_dir}")
        return 2

    exclude = [e for e in (args.exclude or []) if e]
    run = args.run or None
    label_to_files = collect_labels_from_predictions(
        predictions_dir=args.predictions_dir,
        pred_cols=args.pred_cols,
        exclude=exclude,
        run=run,
        verbose=verbose,
    )
    unique_labels = sorted(label_to_files.keys())

    if verbose:
        print(f"Collected {len(unique_labels)} unique labels from predictions")

    if not unique_labels:
        print("No labels found - nothing to check.")
        return 0

    unmapped, _how_mapped = check_labels_mappable(
        labels=unique_labels,
        cache_dir=args.cache_dir,
        verbose=verbose,
    )

    print("\n" + "=" * 60)
    print("ONTOLOGY MAPPING COVERAGE SUMMARY")
    print("=" * 60)
    print(f"Total unique predicted labels: {len(unique_labels)}")
    print(f"Mappable:                      {len(unique_labels) - len(unmapped)}")
    print(f"NOT mappable:                  {len(unmapped)}")

    if unmapped:
        if args.print_unmapped:
            print("\nLabels that cannot be mapped:")
            for label in sorted(unmapped):
                files = sorted(label_to_files[label])
                example = files[0].relative_to(_REPO_ROOT) if files else "?"
                more = f" (+{len(files) - 1} more files)" if len(files) > 1 else ""
                print(f"  - {label!r}  [e.g. {example}{more}]")
        elif not args.output:
            print("\nRe-run with --print-unmapped to list them.")

        if args.output:
            rows = []
            for label in sorted(unmapped):
                files = sorted(label_to_files[label])
                example = str(files[0].relative_to(_REPO_ROOT)) if files else ""
                rows.append(
                    {
                        "predicted_label": label,
                        "n_files": len(files),
                        "example_file": example,
                    }
                )
            out_df = pd.DataFrame(rows, columns=["predicted_label", "n_files", "example_file"])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(args.output, index=False)
            print(f"\nWrote {len(rows)} unmapped labels to {args.output}")
        return 1

    print("\nAll predicted labels can be mapped to the ontology. ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
