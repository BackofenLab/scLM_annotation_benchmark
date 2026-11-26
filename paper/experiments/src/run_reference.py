#!/usr/bin/env python
"""Run the baselines of the imperfect-list study with the drawn candidate lists.

A baseline can name only the cell types of its reference or marker database, so each list becomes
a label constraint (``util.label_constraint``, with this study's mapping table) and the method's
pipeline wrapper (``models/<method>_unconstrained``) runs with ``LABEL_CONSTRAINT``: SingleR again
on the restricted reference, since it selects marker genes among the labels it gets; scType again
on the kept gene sets, since its marker weights depend on the set; CellTypist and SCSA with their
best match among the kept labels. scType and SCSA cluster the dataset's context first, as in the
pipeline. The wrapper records per cell whether a label it can give counts for the true type
(``reachable``).

Writes ``results_lists/predictions/<arm>/<dataset>/<condition>/default/seed<seed>/predictions.csv``
for ``score.py --study lists``. Each distinct list runs once; a condition whose list equals E's
(EU) is skipped, since these methods cannot answer "uncertain".

Run in the evaluation environment, with ENV the conda environment of the method's wrapper
(``envs/models/<method>_unconstrained.yml``). Leave OMP_NUM_THREADS unset, as in the pipeline,
since the number of threads can change SCSA's labels.
    python src/run_reference.py singler --env ENV [--datasets ...]
    python src/run_reference.py celltypist --env ENV [--datasets ...]
    python src/run_reference.py sctype --env ENV [--datasets ...]
    python src/run_reference.py scsa --env ENV [--datasets ...]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lists import MAPPINGS, REPO_ROOT, ROOT, eval_subset, ontology, read_list  # noqa: E402
from util.label_constraint import closure, write  # noqa: E402

ARMS = {"singler": "B1_singler", "celltypist": "B2_celltypist", "sctype": "B3_sctype",
        "scsa": "B4_scsa"}
CONDA = os.environ.get("CONDA_EXE", "conda")


def staged_inputs(method, dataset):
    """The extra inputs the pipeline passes to a method as environment variables: SingleR's
    CELLxGENE reference, or the clustering context of scType and SCSA."""
    cached = REPO_ROOT / "cache" / "datasets"
    ds_cfg = yaml.safe_load((REPO_ROOT / "datasets" / f"{dataset}.yml").read_text())
    if method == "singler" and ds_cfg.get("singler_reference_source") == "cellxgene":
        return {"SINGLER_REF_H5AD": cached / "reference" / f"{dataset}_singler_ref.h5ad"}
    if method in ("sctype", "scsa"):
        return {"CONTEXT_H5AD": cached / "context" / f"{dataset}_context.h5ad"}
    return {}


def run_wrapper(method, env, subset, dataset, out, **variables):
    """A method's pipeline wrapper on the eval subset, with extra environment variables."""
    subprocess.run([CONDA, "run", "--no-capture-output", "-n", env, "bash",
                    str(REPO_ROOT / "models" / f"{method}_unconstrained" / "src" / "run.sh"),
                    str(subset), str(out), str(REPO_ROOT / "datasets" / f"{dataset}.yml"),
                    str(REPO_ROOT / "cache" / "models")],
                   check=True, env={**os.environ, **{k: str(v) for k, v in variables.items()}})
    return pd.read_csv(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("method", choices=sorted(ARMS))
    p.add_argument("--env", required=True, help="the conda environment of the method's wrapper")
    p.add_argument("--config", default=str(ROOT / "config.yml"))
    p.add_argument("--datasets", nargs="+")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    from util.ontology_mapping import load_review_status
    cache_dir = ROOT / cfg["scoring"]["cache_dir"]
    label_index, taxonomy, _addable = ontology(cache_dir)
    _reviewed, review_map = load_review_status(cache_dir / MAPPINGS)
    tree, arm = ROOT / cfg["studies"]["lists"], ARMS[args.method]

    for dataset in args.datasets or cfg["datasets"]:
        subset, extra = eval_subset(dataset, cfg), staged_inputs(args.method, dataset)
        done = {}
        for seed in cfg["repeats"]["seeds"]:
            exact = read_list(tree, dataset, "E", seed)
            for condition in cfg["lists"]["conditions"]:
                listed = read_list(tree, dataset, condition, seed)
                if condition != "E" and listed == exact:
                    continue
                if tuple(listed) not in done:
                    with tempfile.TemporaryDirectory() as tmp:
                        table = Path(tmp) / f"{dataset}__{condition}__seed{seed}.constraint.tsv"
                        write(closure(listed, label_index, taxonomy, review_map), table)
                        done[tuple(listed)] = run_wrapper(args.method, args.env, subset, dataset,
                                                          Path(tmp) / "raw.csv",
                                                          LABEL_CONSTRAINT=table, **extra)
                out = (tree / "predictions" / arm / dataset / condition / "default" / f"seed{seed}"
                       / "predictions.csv")
                out.parent.mkdir(parents=True, exist_ok=True)
                done[tuple(listed)][["cell_id", "true_labels", "predicted_labels",
                                     "reachable"]].to_csv(out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
