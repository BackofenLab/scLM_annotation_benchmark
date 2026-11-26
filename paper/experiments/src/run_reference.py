#!/usr/bin/env python
"""The baselines given the candidate lists: part B of the imperfect-list study.

A baseline can name only the cell types of its reference or marker database, so a list restricts
it rather than informs it. Each list becomes a label constraint (``util.label_constraint``, built
with this study's mapping table), and the method's pipeline wrapper runs in its constrained mode
(``LABEL_CONSTRAINT``): SingleR again on the restricted reference, because it chooses its marker
genes among the labels it is given; scType again on the kept gene sets, because its marker weights
depend on the set; CellTypist and SCSA with their best match among the kept labels, since each
label is scored on its own. scType and SCSA first cluster their dataset's context, as in the
pipeline. The wrapper answers with listed names and records per cell whether its true type was
reachable (``reachable``), since a listed type that no label of the method reaches cannot be
predicted.

Writes ``predictions/<arm>/<dataset>/<condition>/default/seed<seed>/predictions.csv`` in the list
study's tree, for ``score.py --study lists``. A list runs once however many conditions share it,
and a condition whose list equals E's (EU) is skipped: these methods cannot answer "uncertain".

Usage, in the evaluation environment (each method's wrapper runs in its own environment), without
OMP_NUM_THREADS so the wrappers thread as in the pipeline (single-threaded, SCSA labels one pancreas
cell differently):
    python src/run_reference.py singler --env singler [--datasets ...]
    python src/run_reference.py celltypist --env celltypist_blind [--datasets ...]
    python src/run_reference.py sctype --env sctype [--datasets ...]
    python src/run_reference.py scsa --env scsa_blind [--datasets ...]
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
from lists import REPO_ROOT, ROOT, eval_subset, ontology, read_list  # noqa: E402
from util.label_constraint import closure, write  # noqa: E402

ARMS = {"singler": "B1_singler", "celltypist": "B2_celltypist", "sctype": "B3_sctype",
        "scsa": "B4_scsa"}
CONDA = os.environ.get("CONDA_EXE", "conda")


def staged_inputs(method, dataset):
    """The files main.nf stages for a method: SingleR's CELLxGENE reference, and the clustering
    context of scType and SCSA."""
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
                    str(REPO_ROOT / "models" / f"{method}_blind" / "src" / "run.sh"), str(subset),
                    str(out), str(REPO_ROOT / "datasets" / f"{dataset}.yml"),
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
    _reviewed, review_map = load_review_status(cache_dir / "cell_type_ontology_mappings.csv")
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
