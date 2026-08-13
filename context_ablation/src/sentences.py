#!/usr/bin/env python
"""Build cell sentences, optionally restricted to a highly-variable gene pool.

Both model paths in the benchmark hand an AnnData to ``cell2sentence.CSData.adata_to_arrow`` and
let the library rank genes by expression, so restricting that AnnData to the HVG pool BEFORE the
call changes what every arm reads, with one intervention rather than two implementations.

The gene-selection ablation must change WHICH genes are eligible, not HOW MANY appear. It cannot
fully: at a 5,000-gene pool, 49% of HFRA cells, 27% of HIHA, 22% of Zeisel and 2% of
human_pancreas have fewer than 200 detected pool genes, so their sentences fall short of the cap.
``build`` therefore records the realized length of every sentence, so the confound is measurable in
the analysis instead of invisible.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ENSEMBL_RE = r"^ENS[A-Z]*G\d"


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def resolve_symbols(adata):
    """Set ``var_names`` to gene symbols, uppercased and made unique.

    Mirrors ``models/openrouter_LLM_blind/src/predict.py`` so sentences built here are identical to
    the benchmark's. The column is chosen by the share of entries that are real symbols -- neither
    Ensembl IDs nor null placeholders -- because a fixed preference order picks the wrong column on
    two of the four datasets (HIHA's ``gene_names`` holds Ensembl IDs; its ``gene_symbols`` is
    almost entirely the string ``nan``).
    """
    def usable(values):
        missing = values.isin(("nan", "None", "", "NA", "<NA>"))
        return float((~values.str.match(ENSEMBL_RE) & ~missing).mean())

    candidates = [("var_names", pd.Index(adata.var_names.astype(str)))]
    candidates += [(c, pd.Index(adata.var[c].astype(str)))
                   for c in ("gene_names", "feature_name", "gene_symbols")
                   if c in adata.var.columns]
    col, values = max(candidates, key=lambda kv: usable(kv[1]))
    if col != "var_names":
        adata.var_names = values
    adata.var_names = adata.var_names.str.upper()
    adata.var_names_make_unique()
    return adata, col


def hvg_path(dataset, cfg):
    hvg_cfg = cfg["hvg"]
    suffix = "_batch" if hvg_cfg.get("use_batch_key") else ""
    return Path(hvg_cfg["out_dir"]) / (
        f"{dataset}_hvg_{hvg_cfg['flavor']}_{hvg_cfg['n_top_genes']}{suffix}.csv")


def restrict_to_hvg(adata, dataset, cfg):
    """Subset to the cached HVG pool, keyed on ``var_id`` (Ensembl), before symbols are resolved.

    Keyed on the Ensembl id rather than the symbol because that is what the h5ad is indexed by at
    this point and it is unambiguous; the symbol column is exactly the thing that varies between
    these datasets.
    """
    path = hvg_path(dataset, cfg)
    if not path.exists():
        raise SystemExit(f"{dataset}: no HVG artifact at {path}; run src/hvg.py first.")
    pool = set(pd.read_csv(path)["var_id"].astype(str))
    keep = adata.var_names.astype(str).isin(pool)
    if not keep.any():
        raise SystemExit(
            f"{dataset}: the HVG pool shares no gene ids with the subset "
            f"(pool e.g. {sorted(pool)[:3]}, subset e.g. {list(adata.var_names[:3])}).")
    return adata[:, keep].copy(), int(keep.sum())


def build(dataset, cfg, use_hvg, seed=1234):
    """Return ``(arrow_dataset, vocabulary, info)`` for one dataset.

    ``info`` carries the realized sentence lengths, which the truncation caveat depends on.
    """
    import cell2sentence as cs  # imported late so --help works without the heavy stack

    eval_cfg = cfg["eval"]
    src = (Path(eval_cfg["subset_dir"]) /
           f"{dataset}_sub{eval_cfg['size']}_{eval_cfg['strategy']}.h5ad")
    if not src.exists():
        raise SystemExit(f"{dataset}: no eval subset at {src}")
    adata = ad.read_h5ad(src)

    n_pool = None
    if use_hvg:
        adata, n_pool = restrict_to_hvg(adata, dataset, cfg)
    adata, symbol_col = resolve_symbols(adata)

    arrow_ds, vocab = cs.CSData.adata_to_arrow(
        adata=adata,
        random_state=seed,
        sentence_delimiter=" ",
        label_col_names=[],
    )

    cap = cfg["genes"]["cap"]
    lengths = [len(str(arrow_ds[i]["cell_sentence"]).split(" ")) for i in range(arrow_ds.num_rows)]
    lengths = pd.Series(lengths)
    info = {
        "dataset": dataset,
        "variant": "hvg" if use_hvg else "expression_ranked",
        "n_cells": int(arrow_ds.num_rows),
        "n_genes_available": int(adata.n_vars),
        "n_hvg_pool_in_subset": n_pool,
        "symbol_col": symbol_col,
        "median_sentence_len": int(lengths.median()),
        "frac_below_cap": float((lengths < cap).mean()),
        "cap": cap,
    }
    return arrow_ds, vocab, info


def check_equivalence(dataset, cfg, n_check=50, seed=1234):
    """Regression test: with HVG off, sentences must match the benchmark's construction.

    This is what licenses comparing any number from this study to the main benchmark table. It
    rebuilds the sentences the way ``models/openrouter_LLM_blind/src/predict.py`` does -- same
    subset, same symbol resolution, same ``adata_to_arrow`` call -- and compares the first
    ``n_check`` cells token for token.
    """
    import cell2sentence as cs
    from cell2sentence.prompt_formatter import get_cell_sentence_str

    arrow_ds, _vocab, info = build(dataset, cfg, use_hvg=False, seed=seed)

    eval_cfg = cfg["eval"]
    src = (Path(eval_cfg["subset_dir"]) /
           f"{dataset}_sub{eval_cfg['size']}_{eval_cfg['strategy']}.h5ad")
    reference, _col = resolve_symbols(ad.read_h5ad(src))
    ref_ds, _v = cs.CSData.adata_to_arrow(
        adata=reference, random_state=seed, sentence_delimiter=" ", label_col_names=[])

    cap = cfg["genes"]["cap"]
    mismatches = []
    for i in range(min(n_check, arrow_ds.num_rows)):
        ours, _n = get_cell_sentence_str(arrow_ds[i], num_genes=cap)
        theirs, _n = get_cell_sentence_str(ref_ds[i], num_genes=cap)
        if ours != theirs:
            mismatches.append(i)
    return mismatches, info


def materialize(dataset, cfg, out_dir, arms=None):
    """Write every per-cell prompt this study needs, as files the runners just read.

    The split runs the OpenRouter arms on one machine and the local-weight arms on another. Both
    consume THESE files rather than building sentences themselves, because ``cell2sentence``
    defines the sentence construction: if the two machines resolved different versions they would
    silently read different inputs and the arms would stop being comparable. Building prompts once
    makes that impossible rather than unlikely, and it means the local-arm machine needs neither
    cell2sentence nor scanpy nor the 1 GB context h5ads.

    One file per (dataset, arm, mode, gene variant, seed). The seed matters: the candidate list is
    shuffled per cell, so a different seed is a genuinely different prompt set, and repeats that
    reused one seed would measure nothing.
    """
    import numpy as np
    from prompts import build_prompt, candidate_list

    eval_cfg = cfg["eval"]
    src = (Path(eval_cfg["subset_dir"]) /
           f"{dataset}_sub{eval_cfg['size']}_{eval_cfg['strategy']}.h5ad")
    truth_key = _annotation_key(dataset)
    obs = ad.read_h5ad(src, backed="r").obs
    labels = sorted(obs[truth_key].astype(str).unique())

    ds_cfg = _dataset_cfg(dataset)
    tissue, species = ds_cfg.get("tissue"), (ds_cfg.get("species") or ds_cfg.get("organism"))
    organism = ds_cfg.get("organism") or ds_cfg.get("species") or "Homo sapiens"
    override = ds_cfg.get("context_phrase")

    arms = arms or cfg["arms"]
    hvg_modes = cfg["genes"].get("hvg_modes")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    built = {}          # (variant) -> (cell_ids, sentences), so each variant is built once
    written = []
    for arm_name, arm in arms.items():
        if arm["granularity"] != "cell":
            continue    # the cluster arm's prompts come from cluster.py, not from cell sentences
        for mode_name, mode in cfg["modes"].items():
            for variant, spec in cfg["genes"]["variants"].items():
                if spec["hvg"] and hvg_modes and mode_name not in hvg_modes:
                    continue
                if spec["hvg"] and not mode.get("genes"):
                    continue    # no gene list to reselect
                if variant not in built and mode.get("genes"):
                    arrow_ds, _v, info = build(dataset, cfg, use_hvg=spec["hvg"])
                    cells = list(ad.read_h5ad(src, backed="r").obs_names.astype(str))
                    from cell2sentence.prompt_formatter import get_cell_sentence_str
                    sents = [get_cell_sentence_str(arrow_ds[i], num_genes=cfg["genes"]["cap"])[0]
                             for i in range(arrow_ds.num_rows)]
                    built[variant] = (cells, sents, info)

                for seed in cfg["repeats"]["seeds"]:
                    rng = np.random.default_rng(seed)
                    if mode.get("genes"):
                        cells, sents, _info = built[variant]
                    else:
                        cells = list(obs.index.astype(str))
                        sents = [None] * len(cells)

                    rows = []
                    for cell_id, sentence in zip(cells, sents):
                        if mode.get("shuffle_genes") and sentence:
                            # Same genes, same count, same format -- only the descending-expression
                            # ORDER is destroyed. That single bit is the whole intervention: it asks
                            # whether the model reads a cell sentence as a ranked profile or as an
                            # unordered bag of markers, which is the premise cell2sentence rests on.
                            #
                            # Shuffled from the SEEDED rng, so the three repeats see three different
                            # orders and the permutation is reproducible from the seed alone.
                            #
                            # The prompt still says "ordered by descending expression level" and is
                            # deliberately NOT reworded: changing the instruction as well would make
                            # this two interventions, and the comparison to M1 would no longer
                            # isolate the ordering.
                            genes = sentence.split(" ")
                            sentence = " ".join(rng.permutation(genes).tolist())
                        types = candidate_list(labels, rng) if mode.get("cell_types") else None
                        rows.append({
                            "cell_id": cell_id,
                            "true_labels": str(obs.loc[cell_id, truth_key]),
                            "prompt": build_prompt(
                                arm["template"], mode, num_genes=str(cfg["genes"]["cap"]),
                                cell_sentence=sentence, unique_cell_types=types,
                                organism=organism, tissue=tissue, species=species,
                                context_override=override),
                        })
                    # A prompt is the experiment's input, so a rendering artefact in it is a silent
                    # change to the intervention. This one is not hypothetical: `list()` instead of
                    # `.tolist()` on a numpy permutation put `[np.str_('retinal rod cell'), ...]`
                    # into every M2/M4/M5 prompt of the instruction arms -- 38,373 prompts -- and
                    # nothing failed. It was found by eye, after the numbers had been reported.
                    # Refuse to write rather than warn: an unwritten file is obvious, a subtly
                    # wrong one is not.
                    frame = pd.DataFrame(rows)
                    for artefact in ("np.str_", "np.float", "np.int", "dtype=", "array(", "nan"):
                        hits = frame["prompt"].str.contains(artefact, regex=False).sum()
                        if hits:
                            raise SystemExit(
                                f"{dataset}/{arm_name}/{mode_name}/{variant}/seed{seed}: "
                                f"{hits} prompt(s) contain '{artefact}'. This is a rendering leak, "
                                f"not valid prompt text -- fix the formatting before writing.")
                    name = f"{dataset}__{arm_name}__{mode_name}__{variant}__seed{seed}.csv"
                    frame.to_csv(out_dir / name, index=False)
                    written.append(name)
    return written


def _dataset_cfg(dataset):
    with open(REPO_ROOT / "datasets" / f"{dataset}.yml") as fh:
        return yaml.safe_load(fh) or {}


def _annotation_key(dataset):
    return _dataset_cfg(dataset).get("annotation_key", "cell_type")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yml"))
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--check-equivalence", action="store_true",
                   help="verify HVG-off sentences match the benchmark, then exit")
    p.add_argument("--materialize", action="store_true",
                   help="write every per-cell prompt to results/prompts/ for the runners")
    p.add_argument("--n-check", type=int, default=50)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    datasets = args.datasets or cfg["datasets"]

    if args.check_equivalence:
        failed = False
        for dataset in datasets:
            mismatches, info = check_equivalence(dataset, cfg, args.n_check)
            status = "OK" if not mismatches else f"MISMATCH on {len(mismatches)} of {args.n_check}"
            print(f"[{status:12}] {dataset}: symbol_col={info['symbol_col']}, "
                  f"median len={info['median_sentence_len']}")
            failed |= bool(mismatches)
        if failed:
            print("\nSentences differ from the benchmark's construction. Do NOT compare any "
                  "number from this study to the main table until this passes.", file=sys.stderr)
            return 1
        return 0

    if args.materialize:
        out_dir = Path(__file__).resolve().parents[1] / cfg["execution"]["prompts_dir"]
        for dataset in datasets:
            written = materialize(dataset, cfg, out_dir)
            print(f"[done] {dataset}: {len(written)} prompt files -> {out_dir}")
        return 0

    rows = []
    for dataset in datasets:
        for variant, spec in cfg["genes"]["variants"].items():
            _ds, _vocab, info = build(dataset, cfg, use_hvg=spec["hvg"])
            rows.append(info)
            print(f"[{variant:17}] {dataset:16} median len={info['median_sentence_len']:4d}  "
                  f"below cap={info['frac_below_cap']:.0%}  genes={info['n_genes_available']:,}")
    pd.DataFrame(rows).to_csv(Path(__file__).resolve().parents[1]
                              / "results" / "sentence_stats.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
