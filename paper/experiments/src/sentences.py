#!/usr/bin/env python
"""Build cell sentences of the eval subsets and write the prompt files of the per-cell arms.

Sentences come from ``cell2sentence.CSData.adata_to_arrow``, as in the benchmark's wrappers. The
``hvg`` gene variant first restricts the subset to the gene pool of ``hvg.py``; a cell with fewer
pool genes than ``genes.cap`` then gets a shorter sentence, so the sentence lengths are recorded.

Run from paper/experiments, since the data paths in config.yml are relative:
    python src/sentences.py                          # writes results/sentence_stats.csv
    python src/sentences.py --materialize            # writes <study tree>/prompts/
    python src/sentences.py --materialize --study lists
    python src/sentences.py --check-equivalence
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
ENSEMBL_RE = r"^ENS[A-Z]*G\d"


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def resolve_symbols(adata):
    """Set ``var_names`` to upper-case gene symbols, made unique.

    The symbols come from whichever of ``var_names``, ``gene_names``, ``feature_name`` and
    ``gene_symbols`` has the largest share of entries that are neither Ensembl ids nor null
    placeholders (HIHA's ``gene_symbols`` is mostly ``nan``). Returns ``(adata, column)``.
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
    """The subset restricted to the HVG pool, matched on the Ensembl ids (``var_id``) before the
    symbols are resolved. Returns ``(adata, number of pool genes in the subset)``."""
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
    """``(arrow_dataset, vocabulary, info)`` for one dataset; ``info`` holds the median sentence
    length and the share of sentences shorter than the cap."""
    import cell2sentence as cs  # imported here so --help works without cell2sentence

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
    """Compare the first ``n_check`` sentences of ``build`` without the HVG pool with the sentences
    that the benchmark's per-cell LLM wrapper
    (``models/openrouter_LLM_unconstrained/src/predict.py``) builds from the same test set with
    its own settings. Returns the indices of differing cells and the info."""
    import importlib.util
    from cell2sentence.prompt_formatter import get_cell_sentence_str

    arrow_ds, _vocab, info = build(dataset, cfg, use_hvg=False, seed=seed)

    wrapper_dir = REPO_ROOT / "models" / "openrouter_LLM_unconstrained" / "src"
    spec = importlib.util.spec_from_file_location("benchmark_wrapper", wrapper_dir / "predict.py")
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    settings = load_config(wrapper_dir / "openrouter.yml")
    eval_cfg = cfg["eval"]
    src = (Path(eval_cfg["subset_dir"]) /
           f"{dataset}_sub{eval_cfg['size']}_{eval_cfg['strategy']}.h5ad")
    label_key = _annotation_key(dataset)
    reference = wrapper.build_prompts(wrapper.load_test_set(src, label_key), "{cell_sentence}",
                                      label_key, settings["seed"], settings["top_k_genes"])

    cap = cfg["genes"]["cap"]
    return [i for i in range(min(n_check, arrow_ds.num_rows))
            if get_cell_sentence_str(arrow_ds[i], num_genes=cap)[0] != reference[i]], info


def materialize(dataset, cfg, out_dir, arms=None, modes=None, candidates=None):
    """Write the prompt files of the per-cell arms for one dataset.

    One file per arm, mode, gene variant and seed; the runners read these files and build no
    sentences. The seed draws the order of the candidate list (``prompts.ordered_candidates``) and,
    in a mode with ``shuffle_genes``, the order of the genes. ``modes`` and
    ``candidates(mode, seed)`` come from ``lists.prompt_args`` for the imperfect-list study;
    without them every mode lists the dataset's true types. Returns the file names.
    """
    import numpy as np
    from prompts import build_prompt, ordered_candidates

    eval_cfg = cfg["eval"]
    src = (Path(eval_cfg["subset_dir"]) /
           f"{dataset}_sub{eval_cfg['size']}_{eval_cfg['strategy']}.h5ad")
    truth_key = _annotation_key(dataset)
    obs = ad.read_h5ad(src, backed="r").obs
    labels = sorted(obs[truth_key].astype(str).unique())

    ds_cfg = _dataset_cfg(dataset)
    tissue, species = ds_cfg.get("tissue"), (ds_cfg.get("species") or ds_cfg.get("organism"))
    # The organism of the C2S template, as in the C2S-Scale wrapper: the subset's organism column,
    # else "Homo sapiens".
    organism = (obs["organism"].astype(str) if "organism" in obs.columns
                else pd.Series("Homo sapiens", index=obs.index))
    override = ds_cfg.get("context_phrase")

    arms = arms or cfg["arms"]
    hvg_modes = cfg["genes"].get("hvg_modes")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    built = {}          # gene variant: (cell ids, sentences, info), built once per variant
    written = []
    for arm_name, arm in arms.items():
        if arm["granularity"] != "cell":
            continue    # cluster arms get their prompts from cluster.py
        for mode_name, mode in (modes or cfg["modes"]).items():
            for variant, spec in cfg["genes"]["variants"].items():
                if spec["hvg"] and hvg_modes and mode_name not in hvg_modes:
                    continue
                if spec["hvg"] and not mode.get("genes"):
                    continue    # a mode without genes has no HVG variant
                if variant not in built and mode.get("genes"):
                    arrow_ds, _v, info = build(dataset, cfg, use_hvg=spec["hvg"])
                    cells = list(ad.read_h5ad(src, backed="r").obs_names.astype(str))
                    from cell2sentence.prompt_formatter import get_cell_sentence_str
                    sents = [get_cell_sentence_str(arrow_ds[i], num_genes=cfg["genes"]["cap"])[0]
                             for i in range(arrow_ds.num_rows)]
                    built[variant] = (cells, sents, info)

                for seed in cfg["repeats"]["seeds"]:
                    rng = np.random.default_rng(seed)
                    listed = candidates(mode_name, seed) if candidates else labels
                    if mode.get("genes"):
                        cells, sents, _info = built[variant]
                    else:
                        cells = list(obs.index.astype(str))
                        sents = [None] * len(cells)

                    rows = []
                    for cell_id, sentence in zip(cells, sents):
                        if mode.get("shuffle_genes") and sentence:
                            # The same genes in an order drawn from the seed; the prompt text,
                            # including "ordered by descending expression level", is unchanged.
                            genes = sentence.split(" ")
                            sentence = " ".join(rng.permutation(genes).tolist())
                        types = (ordered_candidates(arm["template"], listed, rng)
                                 if mode.get("cell_types") else None)
                        rows.append({
                            "cell_id": cell_id,
                            "true_labels": str(obs.loc[cell_id, truth_key]),
                            "prompt": build_prompt(
                                arm["template"], mode, num_genes=str(cfg["genes"]["cap"]),
                                cell_sentence=sentence, unique_cell_types=types,
                                organism=organism[cell_id], tissue=tissue, species=species,
                                context_override=override),
                        })
                    # No file is written if a prompt holds the text of a numpy object or "nan".
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
    p.add_argument("--study", default="ablation", help="the study tree (see `studies` in config)")
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
            print("\nSentences differ from those built from the eval subset directly; the "
                  "prompts would then differ from the benchmark's.", file=sys.stderr)
            return 1
        return 0

    if args.materialize:
        from lists import prompt_args
        tree = Path(__file__).resolve().parents[1] / cfg["studies"][args.study]
        for dataset in datasets:
            study = prompt_args(cfg, tree, dataset) if args.study == "lists" else {}
            written = materialize(dataset, cfg, tree / "prompts", **study)
            print(f"[done] {dataset}: {len(written)} prompt files -> {tree / 'prompts'}")
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
