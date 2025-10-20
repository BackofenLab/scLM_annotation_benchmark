#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LangCell zero-shot cell-type classification with CTM fusion.
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

# Project-local utils
from utils import BertModel as MedBertModel
from utils import (
    LangCellDataCollatorForCellClassification as DataCollatorForCellClassification,
)
from utils import LangCellTranscriptomeTokenizer

logger = logging.getLogger(__name__)
curr_path = os.getcwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Set RNG seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Pooler(nn.Module):
    """
    Projects CLS token to a lower-dimensional, L2-normalized embedding.

    Parameters
    ----------
    config : transformers.BertConfig
        Config with `hidden_size` used to init linear layer.
    pretrained_proj : str
        Path to a `.bin` state_dict for the projection layer.
    proj_dim : int
        Target dimensionality of the pooled embedding.
    """

    def __init__(self, config, pretrained_proj: str, proj_dim: int):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, proj_dim)
        self.proj.load_state_dict(torch.load(pretrained_proj, map_location="cpu"))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Use CLS token (position 0), then project + normalize
        pooled_output = hidden_states[:, 0]
        pooled_output = F.normalize(self.proj(pooled_output), dim=-1)
        return pooled_output


def load_model_assets(base_dir: Path, device: torch.device):
    """
    Load transcriptome encoder, text encoder, tokenizer, and CTM head.

    Assumes the following files exist under `base_dir / 'data' / 'ckpt'`:
    - cell_bert            (HF folder)
    - cell_proj.bin        (projection weights)
    - text_bert            (HF folder)
    - text_proj.bin        (projection weights)
    - ctm_head.bin         (linear classification head)
    """
    ckpt_dir = base_dir / "data" / "ckpt"
    # Transcriptome side (BERT-like)
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
    # CTM head maps text CLS to binary logits; we will use softmax(...)[...,1]
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
    """
    Add necessary attributes to the dataframe.
    """
    if sp.issparse(dataframe.X):
        n_counts = np.asarray(dataframe.X.sum(axis=1)).ravel()
        n_genes = dataframe.X.getnnz(axis=1)
    else:
        n_counts = dataframe.X.sum(axis=1)
        n_genes = (dataframe.X > 0).sum(axis=1)

    dataframe.obs["batch"] = dataframe.obs["_scvi_batch"]
    dataframe.obs["str_labels"] = dataframe.obs[annotation_key]
    dataframe.obs["celltype"] = dataframe.obs[annotation_key]
    dataframe.var["ensembl_id"] = dataframe.var["ensembl_id"].fillna("").astype(str)
    f_ensembl_ids = dataframe.var["ensembl_id"].str.replace(r"\.\d+$", "", regex=True)
    dataframe.var["feature_id"] = f_ensembl_ids
    dataframe.var["ensembl_id"] = f_ensembl_ids
    dataframe.obs["n_genes"] = np.asarray(n_genes, dtype=np.int32)
    dataframe.obs["n_counts"] = np.asarray(n_counts, dtype=np.int32)
    return dataframe


def load_dataset(tokenizer, text_encoder, annotation_key, args):
    """
    Read h5ad, filter/prepare, tokenize cells, map labels, stratify zero-shot test.
    """
    dataset_sub = sc.read(args.input)

    # find highly variable genes
    #sc.pp.highly_variable_genes(dataset_sub, n_top_genes=args.n_top_genes)
    #dataset_sub = dataset_sub[:, dataset_sub.var.highly_variable]

    if "n_genes_by_counts" in dataset_sub.obs.columns:
        ngenes_col = "n_genes_by_counts"
    elif "n_genes" in dataset_sub.obs.columns:
        ngenes_col = "n_genes"
    else:
        raise KeyError("Neither 'n_genes_by_counts' nor 'n_genes' found in .obs")

    # add key features
    dataset_sub.obs["filter_pass"] = (dataset_sub.obs[ngenes_col] >= 1).astype(bool)

    dataset_sub = setup_dataframe(dataset_sub, annotation_key)
    cell_indices = dataset_sub.obs.index.tolist()
    tk = LangCellTranscriptomeTokenizer(
        dict([(k, k) for k in dataset_sub.obs.keys()]), nproc=4
    )
    # create tokenized dataset
    tokenized_cells, cell_metadata = tk.tokenize_anndata(dataset_sub)
    tokenized_dataset = tk.create_dataset(tokenized_cells, cell_metadata)

    for label_name in ["celltype", "cell_type", "str_labels", "labels"]:
        if label_name in tokenized_dataset.column_names:
            break
    if label_name != "celltype":
        tokenized_dataset = tokenized_dataset.rename_column(label_name, "celltype")

    # collect textual features
    types = list(set(tokenized_dataset["celltype"]))
    ontology = json.load(open(f"{curr_path}/data/obo.json"))
    name2id = {ontology[id]["name"]: id for id in ontology}
    texts = [gettextfromname(typename, name2id, ontology) for typename in types]
    type2num = dict([(type, i) for i, type in enumerate(types)])
    with torch.no_grad():
        text_embs = torch.cat(
            [text_encode(text, tokenizer, text_encoder) for text in texts], 0
        ).T.cuda()  # 256 * N
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


def gettextfromname(name, name2id, ontology, description=True):
    """
    Collect description for cell types from the ontology.
    """
    ontology_name = name.lower()
    id = name2id[ontology_name.lower()]
    s = "cell type: "
    s += ontology[id]["name"] + ". "
    if description:
        if ontology[id]["def"] != []:
            s += ontology[id]["def"] + "; "
    return s


def text_encode(text, tokenizer, text_encoder):
    text = tokenizer(
        text, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to("cuda")
    text = text_encoder(**text).pooler_output
    return text


def cell_encode(model, cell_input_ids, cell_atts):
    cell = model(cell_input_ids.to("cuda"), cell_atts.to("cuda"))
    cell_last_h = cell.last_hidden_state
    cell_pooler = cell.pooler_output
    return cell_last_h, cell_pooler


def ctm(tokenizer, text_encoder, ctm_head, text, cell_emb, cell_atts):
    text = tokenizer(
        text, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to("cuda")
    output = text_encoder(
        **text,
        encoder_hidden_states=cell_emb.to("cuda"),
        encoder_attention_mask=cell_atts.to("cuda"),
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
            )  # batchsize * 256
            sim = (
                cellemb @ text_embs
            ) / 0.05  # batchsize * 161 # uncomment to combine cell emb and text
            sim_logit = F.softmax(sim, dim=-1)
            # ctm
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
            ) / 2  # Use the following line when combinging cell emb and text
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
    # evaluation
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
    """
    Plot UMAP
    """
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
    """Parse CLI arguments."""
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
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--batchsize", type=int, default=32, help="Test batch size (default: 32)"
    )
    #p.add_argument("--n_top_genes", type=int, default=3000, help="Number of HVGs (default: 3000)")
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
        "--annotation_keys",
        nargs='+',
        default=["cell_type"], 
        help="Keys in adata.obs containing true labels (can specify multiple)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    logger.info("Args: %s", args)

    base_dir = Path.cwd()
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_stem = output_path.stem
    output_suffix = output_path.suffix
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seed
    set_seed(args.seed)

    # Load models/tokenizer/head
    model, tokenizer, text_encoder, ctm_head = load_model_assets(base_dir, device)
    logger.info("Model assets loaded.")

    for annotation_key in args.annotation_keys:
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
        # use the following line to combine cell emb and text
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
        output_csv = output_dir / f"{output_stem}_{annotation_key}{output_suffix}"
        pred_df.to_csv(output_csv, index=True)
        logger.info("Langcell prediction pipeline completed successfully")
        logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
