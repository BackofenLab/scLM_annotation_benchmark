#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Zero-shot cell type annotation with LangCell, with the dataset's labels as candidates (unblind).

The candidates are the labels in obs[--annotation_key]. Each candidate is described by its Cell
Ontology name and definition from data/obo.json. A cell's score for a candidate combines the
cell-text embedding similarity and the cell-text matching (CTM) head,
alpha * sim + (1 - alpha) * ctm, each softmax-normalized over the candidates. Labels are
lowercased.

Input: preprocessed h5ad with raw counts in layers["counts"] and Ensembl IDs in
var["ensembl_id"], and the model files that prepare.sh puts into data/. Output: CSV with
cell_id, predicted_idx, true_idx, predicted_labels and true_labels. Called by run.sh.
"""

from __future__ import annotations

import os
import json
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import anndata as ad
from mygene import MyGeneInfo
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
    silhouette_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns  # used for heatmap

# Model, collator and tokenizer classes from the LangCell repository
from utils import BertModel as MedBertModel
from utils import (
    LangCellDataCollatorForCellClassification as DataCollatorForCellClassification,
)
from utils import LangCellTranscriptomeTokenizer

logger = logging.getLogger(__name__)
curr_path = os.getcwd()
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Pooler(nn.Module):
    """Project the CLS token to an L2-normalized proj_dim embedding with pretrained weights."""

    def __init__(self, config, pretrained_proj: str, proj_dim: int):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, proj_dim)
        self.proj.load_state_dict(torch.load(pretrained_proj, map_location="cpu"))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled_output = hidden_states[:, 0]
        pooled_output = F.normalize(self.proj(pooled_output), dim=-1)
        return pooled_output


def load_model_assets(base_dir: Path, device: torch.device):
    """Load the cell encoder, text tokenizer, text encoder and CTM head.

    Reads from base_dir/data/ckpt: cell_bert/ and text_bert/ (Hugging Face model folders),
    cell_proj.bin and text_proj.bin (projection weights) and ctm_head.bin (CTM head).
    """
    ckpt_dir = base_dir / "data" / "ckpt"
    model = BertModel.from_pretrained(str(ckpt_dir / "cell_bert"))
    model.pooler = Pooler(
        model.config,
        pretrained_proj=str(ckpt_dir / "cell_proj.bin"),
        proj_dim=256,
    )
    model = model.to(device)
    text_pretrained_model = (
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    )
    tokenizer = BertTokenizer.from_pretrained(text_pretrained_model)
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    text_encoder = MedBertModel.from_pretrained(
        str(ckpt_dir / "text_bert"),
        add_pooling_layer=True,
    )
    text_encoder.pooler = Pooler(
        text_encoder.config,
        pretrained_proj=str(ckpt_dir / "text_proj.bin"),
        proj_dim=256,
    )
    text_encoder = text_encoder.to(device)
    # The CTM head maps the text CLS token to two logits; softmax(...)[..., 1] is the match probability.
    ctm_head = nn.Linear(text_encoder.config.hidden_size, 2)
    ctm_head.load_state_dict(
        torch.load(str(ckpt_dir / "ctm_head.bin"), map_location="cpu")
    )
    ctm_head = ctm_head.to(device)
    return model, tokenizer, text_encoder, ctm_head


def calculate_mean_max_min(adata, feature_name):
    vals = adata.obs[feature_name]
    return np.mean(vals), np.median(vals), np.max(vals), np.min(vals)


def setup_dataframe(dataframe, annotation_key):
    """Add the lowercased labels and the Ensembl IDs (version suffix removed) that the tokenizer needs."""
    # Without var["ensembl_id"], gene symbols are mapped to Ensembl IDs with mygene (network access).
    if "ensembl_id" in dataframe.var.columns:
        gene_names = dataframe.var["ensembl_id"].fillna("").astype(str)
    else:
        gene_names = dataframe.var_names.astype(str)
    if isinstance(gene_names, pd.Index):
        gene_names = gene_names.to_list()
        print("Converting gene names to Ensembl IDs")
        mg = MyGeneInfo()
        res = mg.querymany(gene_names, scopes="symbol,alias,gene", fields="ensembl.gene", species="human")
        symbol2ensembl = {}
        for entry in res:
            symbol = entry["query"]
            ens = None
            if isinstance(entry.get("ensembl"), dict):
                ens = entry["ensembl"].get("gene")
            elif isinstance(entry.get("ensembl"), list):
                ens = entry["ensembl"][0].get("gene")
            symbol2ensembl[symbol] = ens
        ensembl_ids = dataframe.var_names.map(symbol2ensembl)
        print(f"Final Ensembl IDs: {len(ensembl_ids)}")
    else:
        ensembl_ids = gene_names.replace(r"\.\d+$", "", regex=True)
    celltypes = dataframe.obs[annotation_key].astype(str).str.strip().str.lower()
    dataframe.obs["str_labels"] = celltypes
    dataframe.obs["celltype"] = celltypes
    dataframe.var["feature_id"] = ensembl_ids
    dataframe.var["ensembl_id"] = ensembl_ids
    # Stop if too few human genes remain, e.g. mouse data that were not mapped to human
    # orthologs (bin/orthologize.py). The model would otherwise see no genes.
    min_human = int(os.environ.get("SCLLM_MIN_HUMAN_GENES", "5000"))
    n_human = int(pd.Series(ensembl_ids).astype(str).str.startswith("ENSG").sum())
    print(f"Human Ensembl gene IDs in the data: {n_human}")
    if n_human < min_human:
        raise SystemExit(f"Only {n_human} human Ensembl gene IDs (minimum {min_human}); "
                         "map mouse genes to human orthologs first with bin/orthologize.py")
    return dataframe


def load_dataset(tokenizer, text_encoder, annotation_key, args):
    """Tokenize the cells, encode the texts of the candidate cell types and build the data loader."""
    dataset_sub = sc.read(args.input)

    # Geneformer's tokenizer expects raw counts: it normalizes them by per-cell totals and gene
    # medians and encodes each cell as a gene ranking.
    if "counts" in dataset_sub.layers:
        dataset_sub.X = dataset_sub.layers["counts"]
    else:
        raise ValueError("Counts layer not found in adata.layers")

    n_genes = (dataset_sub.X > 0).sum(axis=1)
    dataset_sub.obs["n_genes"] = np.asarray(n_genes, dtype=np.int32)
    dataset_sub.obs["n_counts"] = dataset_sub.X.sum(axis=1)

    # The tokenizer reads obs["n_counts"] and obs["filter_pass"].
    dataset_sub.obs["filter_pass"] = (dataset_sub.obs["n_genes"] > 0).astype(bool)

    dataset_sub = setup_dataframe(dataset_sub, annotation_key)
    cell_indices = dataset_sub.obs.index.tolist()
    tk = LangCellTranscriptomeTokenizer(
        dict([(k, k) for k in dataset_sub.obs.keys()]), nproc=4
    )
    tokenized_cells, cell_metadata = tk.tokenize_anndata(dataset_sub)
    tokenized_dataset = tk.create_dataset(tokenized_cells, cell_metadata)

    for label_name in ["celltype", "cell_type", "str_labels", "labels"]:
        if label_name in tokenized_dataset.column_names:
            break
    if label_name != "celltype":
        tokenized_dataset = tokenized_dataset.rename_column(label_name, "celltype")

    # Candidates: the dataset's own labels
    types = list(set(tokenized_dataset["celltype"]))
    obo_path = args.model_path.parent / "data" / "obo.json"
    ontology = json.load(open(obo_path))
    name2id = {ontology[id]["name"]: id for id in ontology}
    synonym2id = {
        synonym: oid
        for oid, entry in ontology.items()
        for synonym in ([entry["synonym"]] if isinstance(entry["synonym"], str) else entry["synonym"])
    }
    texts = [gettextfromname(typename, name2id, synonym2id, ontology) for typename in types]
    print(f"Cell types: {types}")
    print(f"Cell types with definitions: {texts}")
    type2num = dict([(type, i) for i, type in enumerate(types)])
    with torch.no_grad():
        text_embs = torch.cat(
            [text_encode(text, tokenizer, text_encoder) for text in texts], 0
        ).T.to(device)  # [256, n_types]
    text_embs.requires_grad = False

    def classes_to_ids(example):
        example["label"] = type2num[example["celltype"]]
        return example

    testdataset = tokenized_dataset.map(classes_to_ids, num_proc=16)
    remove_columns = testdataset.column_names
    remove_columns.remove("input_ids")
    remove_columns.remove("label")
    testdataset = testdataset.remove_columns(remove_columns)
    collator = DataCollatorForCellClassification()
    dataloader = DataLoader(
        testdataset, batch_size=args.batchsize, collate_fn=collator, shuffle=False
    )
    return (
        tokenized_dataset,
        testdataset,
        texts,
        text_embs,
        dataloader,
        types,
        cell_indices
    )


def gettextfromname(name, name2id, synonym2id, ontology, description=True):
    """Describe a cell type by its ontology name and definition, or by the name alone if not found."""
    ontology_name = name.lower()
    s = "cell type: "
    id = None
    if ontology_name in name2id:
        id = name2id[ontology_name]
        s += ontology[id]["name"] + ". "
    elif ontology_name in synonym2id:
        id = synonym2id[ontology_name]
        s += ontology[id]["synonym"] + ". "
    if description and id is not None:
        if ontology[id]["def"] != []:
            s += ontology[id]["def"] + "; "
    else:
        print(f"{ontology_name} not found in ontology.")
        s += ontology_name
    print(f"Cell definition: {s}")
    return s


def text_encode(text, tokenizer, text_encoder):
    text = tokenizer(
        text, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)
    text = text_encoder(**text).pooler_output
    return text


def cell_encode(model, cell_input_ids, cell_atts):
    cell = model(cell_input_ids.to(device), cell_atts.to(device))
    cell_last_h = cell.last_hidden_state
    cell_pooler = cell.pooler_output
    return cell_last_h, cell_pooler


def ctm(tokenizer, text_encoder, ctm_head, text, cell_emb, cell_atts):
    """Probability from the cell-text matching (CTM) head that each cell matches its text."""
    text = tokenizer(
        text, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)
    output = text_encoder(
        **text,
        encoder_hidden_states=cell_emb.to(device),
        encoder_attention_mask=cell_atts.to(device),
        return_dict=True,
        mode="multimodal",
    )
    logits = ctm_head(output.last_hidden_state[:, 0, :])
    logits = F.softmax(logits, dim=-1)[..., 1]  # [n]
    return logits


def predict(
    model,
    tokenizer,
    text_encoder,
    ctm_head,
    dataset_sub,
    testdataset,
    texts,
    text_embs,
    dataloader,
    types,
    cell_indices,
    args,
):
    """Score every cell against every candidate, print diagnostics and return the results table."""
    batchsize = args.batchsize
    cell_embs = torch.zeros(len(dataset_sub), 256)
    model.eval()
    text_encoder.eval()
    preds = torch.zeros(len(dataset_sub))
    sim_logits = torch.zeros(len(dataset_sub), text_embs.shape[-1])
    ctm_logits = torch.zeros(len(dataset_sub), text_embs.shape[-1])
    logits = torch.zeros(len(dataset_sub), text_embs.shape[-1])
    labels = torch.tensor(testdataset["label"])
    with torch.no_grad():
        for i, d in tqdm(enumerate(dataloader)):
            cell_last_h, cellemb = cell_encode(
                model, d["input_ids"], d["attention_mask"]
            )  # [batch, 256]
            sim = (
                cellemb @ text_embs
            ) / 0.05  # [batch, n_types], temperature 0.05
            sim_logit = F.softmax(sim, dim=-1)
            ctm_logit = torch.zeros_like(sim_logit)
            for text_idx, text in enumerate(texts):
                text_list = [text] * sim_logit.shape[0]
                ctm_logit[:, text_idx] = ctm(
                    tokenizer,
                    text_encoder,
                    ctm_head,
                    text_list,
                    cell_last_h,
                    d["attention_mask"],
                )
            ctm_logit = F.softmax(ctm_logit, dim=-1)
            logit = (
                sim_logit + ctm_logit
            ) / 2
            pred = logit.argmax(dim=-1)
            sim_logits[i * batchsize : (i + 1) * batchsize] = sim_logit.cpu()
            ctm_logits[i * batchsize : (i + 1) * batchsize] = ctm_logit.cpu()
            logits[i * batchsize : (i + 1) * batchsize] = logit.cpu()
            cell_embs[i * batchsize : (i + 1) * batchsize] = cellemb.cpu()
            preds[i * batchsize : (i + 1) * batchsize] = pred.cpu()
    alpha = args.alpha
    preds = (alpha * sim_logits + (1 - alpha) * ctm_logits).argmax(dim=-1)
    nmi = normalized_mutual_info_score(labels, preds)
    ari = adjusted_rand_score(labels, preds)
    print(f"Normalized Mutual Information (NMI) Score: {nmi}")
    print(f"Adjusted Rand Index (ARI) Score: {ari}")
    for row in confusion_matrix(labels, preds):
        print("\t".join([str(x) for x in row]))
    print(classification_report(labels, preds, digits=4))
    pred_df = create_results_dataframe(cell_indices, preds, labels, types)
    return pred_df


def create_results_dataframe(cell_indices, preds, labels, types):
    label_keys = ["predicted_idx", "true_idx", "predicted_labels", "true_labels"]
    pred_df = pd.DataFrame(
        {
            "cell_id": cell_indices,
            label_keys[0]: preds.numpy(),
            label_keys[1]: labels.numpy(),
            label_keys[2]: [types[i] for i in preds],
            label_keys[3]: [types[i] for i in labels],
        }
    )
    return pred_df


def plot_umap(cell_embs, dataset_sub, preds, types, file_name, mode, o_path, nmi, ari):
    plt.tight_layout()
    cell_embs_ad = ad.AnnData(cell_embs.numpy())
    cell_embs_ad.obs["celltype"] = dataset_sub["celltype"]
    cell_embs_ad.obs["predictions"] = [types[i] for i in preds]
    sc.pp.neighbors(cell_embs_ad, use_rep="X", n_neighbors=80)
    sc.tl.umap(cell_embs_ad)
    sc.pl.umap(
        cell_embs_ad,
        color=["celltype", "predictions"],
        legend_fontsize="xx-small",
        size=5,
        legend_fontweight="light",
    )
    save_path = f"{o_path}/umap_plot_{file_name}_{mode}.png"
    plt.title(f"UMAP plot of {file_name} ({mode}): NMI={nmi:.2f}, ARI={ari:.2f}")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"UMAP plot saved to {save_path}")


def plot_confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    classes: Sequence[str],
    out_dir: Path,
    title: str,
    file_n: str,
    mode: str,
    normalize: bool = True,
) -> None:
    """Save a labeled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=False, cmap=plt.cm.Blues)
    plt.title(
        title or ("Normalized confusion matrix" if normalize else "Confusion matrix")
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    out_path = f"{out_dir}/confusion_matrix_{file_n}_{mode}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix: {out_path}")


def parse_args() -> argparse.Namespace:
    curr_path = Path.cwd()
    default_input = curr_path.parents[3] / "util" / "test_adata.h5ad"
    default_output = curr_path / "langcell_predictions.csv"

    p = argparse.ArgumentParser(
        description="LangCell few/zero-shot prediction for single-cell data"
    )
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=default_input,
        help=f"Input h5ad file path (default: {default_input})",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_output,
        help=f"Output CSV file path (default: {default_output})",
    )
    p.add_argument(
        "-m",
        "--model_path",
        type=Path,
        default=default_output,
        help=f"Model files path (default: {default_output})",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--batchsize", type=int, default=32, help="Test batch size (default: 32)"
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Fusion weight for SIM logits (default: 0.1)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device, e.g. 'cuda:0' or 'cpu' (default: auto)",
    )
    p.add_argument(
        "--annotation_key",
        default="cell_type", 
        help="Key in adata.obs containing true labels",
    )
    return p.parse_args()


def main():
    args = parse_args()
    logger.info("Args: %s", args)

    base_dir = Path.cwd()
    model_path = args.model_path.parent
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    model, tokenizer, text_encoder, ctm_head = load_model_assets(model_path, device)
    logger.info("Model assets loaded.")

    annotation_key = args.annotation_key
    print(f"Running prediction for annotation key: {annotation_key}")
    (
        dataset_sub,
        testdataset,
        texts,
        text_embs,
        dataloader,
        types,
        cell_indices
    ) = load_dataset(tokenizer, text_encoder, annotation_key, args)
    print("Loading dataset finished.")
    pred_df = predict(
        model,
        tokenizer,
        text_encoder,
        ctm_head,
        dataset_sub,
        testdataset,
        texts,
        text_embs,
        dataloader,
        types,
        cell_indices,
        args,
    )
    pred_df.to_csv(args.output, index=True)
    logger.info("Langcell prediction pipeline completed successfully")
    logger.info(f"Results saved to: {args.output}")
    
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
