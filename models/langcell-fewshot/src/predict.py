#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LangCell few/zero-shot cell-type classification with CTM fusion.
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
    normalized_mutual_info_score,
    adjusted_rand_score,
)
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

    # Text side (PubMedBERT + custom Pooler)
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
    Read h5ad, filter/prepare, tokenize cells, map labels, stratify few-shot train/test.
    """
    dataset_sub = sc.read(args.input)
    file_name = os.path.splitext(os.path.basename(args.input))[0]

    # find highly variable genes
    #sc.pp.highly_variable_genes(dataset_sub, n_top_genes=args.n_top_genes)
    #dataset_sub = dataset_sub[:, dataset_sub.var.highly_variable]

    dataset_sub.obs["filter_pass"] = (
        dataset_sub.obs["n_genes_by_counts"] >= 200
    ).astype(bool)

    dataset_sub = setup_dataframe(dataset_sub, annotation_key)

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
    tokenized_dataset = tokenized_dataset.filter(
        lambda example: example["celltype"] != "Other"
    )

    types = list(set(tokenized_dataset["celltype"]))
    ontology = json.load(open(f"{curr_path}/data/obo.json"))
    name2id = {ontology[id]["name"]: id for id in ontology}

    texts = [gettextfromname(typename, name2id, ontology) for typename in types]
    type2num = dict([(type, i) for i, type in enumerate(types)])

    def classes_to_ids(example):
        example["label"] = type2num[example["celltype"]]
        return example

    dataset = tokenized_dataset.map(classes_to_ids, num_proc=16)
    dataset = dataset.remove_columns(["length"])
    dataset = dataset.shuffle(args.seed)
    label_num = len(type2num.keys())
    type2trainlist = {}
    for i in range(label_num):
        type2trainlist[i] = []
    if args.nshot >= 1:
        for i, l in enumerate(dataset["label"]):
            if len(type2trainlist[l]) < args.nshot:
                type2trainlist[l].append(i)
                br = True
                for k in type2trainlist.keys():
                    if len(type2trainlist[k]) < args.nshot:
                        br = False
                        break
                if br:
                    break
    train_idx = []
    for k in type2trainlist.keys():
        train_idx += type2trainlist[k]
    test_idx = list(set(range(len(dataset))) - set(train_idx))
    # to keep: ['input_ids', 'n_counts', 'batch', 'labels', 'n_genes', 'filter_pass', 'label']
    features_to_keep = [
        "input_ids",
        "n_counts",
        "labels",
        "n_genes",
        "filter_pass",
        "label",
        "celltype",
    ]
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in features_to_keep]
    )

    traindataset = dataset.select(train_idx).shuffle(args.seed)
    testdataset = dataset.select(test_idx).shuffle(args.seed)
    traindataset = traindataset.remove_columns(["celltype"])
    testdataset = testdataset.remove_columns(["celltype"])

    train_loader = DataLoader(
        traindataset,
        batch_size=args.train_batchsize,
        collate_fn=DataCollatorForCellClassification(),
        shuffle=False,
    )
    test_loader = DataLoader(
        testdataset,
        batch_size=args.test_batchsize,
        collate_fn=DataCollatorForCellClassification(),
        shuffle=False,
    )

    return (
        train_loader,
        test_loader,
        texts,
        testdataset,
        tokenizer,
        text_encoder,
        types,
        file_name
    )


def gettextfromname(name, name2id, ontology, discription=True):
    """
    Collect description for cell types from the ontology.
    """
    ontology_name = name.lower()
    id = name2id[ontology_name.lower()]
    s = "cell type: "
    s += ontology[id]["name"] + ". "
    if discription:
        if ontology[id]["def"] != []:
            s += ontology[id]["def"] + "; "
    return s


def text_encode(text):
    text = tokenizer(
        text, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to("cuda")
    text = text_encoder(**text).pooler_output
    return text


def cell_encode(cell_input_ids, cell_atts):
    cell = model(cell_input_ids.to("cuda"), cell_atts.to("cuda"))
    cell_last_h = cell.last_hidden_state
    cell_pooler = cell.pooler_output
    return cell_last_h, cell_pooler


def ctm(text, cell_emb, cell_atts):
    # n texts, n cells -> n scores
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


def fine_tune_model(
    model: BertModel,
    text_encoder: MedBertModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    texts: List[str],
    testdataset: dict,
    types: List[str],
    epochs: int,
    alpha: float = 0.1,
) -> pd.DataFrame:
    model.train()
    text_encoder.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer2 = torch.optim.Adam(text_encoder.parameters(), lr=1e-5)

    for epoch in range(epochs):
        print(f"Training epoch {epoch+1}/{epochs}")
        for i, d in tqdm(enumerate(train_loader)):
            model.train()
            text_encoder.train()
            text_embs = torch.cat([text_encode(text) for text in texts], 0).T.cuda()
            cell_last_h, cellemb = cell_encode(
                d["input_ids"], d["attention_mask"]
            )  # batchsize * 256
            # text_embs: 256 * class_num
            sim = (cellemb @ text_embs) / 0.05  # batchsize * class_num
            loss_sim = loss_fn(sim, d["labels"].cuda())
            ctm_logit = torch.zeros_like(sim)
            for text_idx, text in enumerate(texts):
                text_list = [text] * sim.shape[0]
                ctm_logit[:, text_idx] = ctm(
                    text_list, cell_last_h, d["attention_mask"]
                )
            loss_ctm = loss_fn(ctm_logit, d["labels"].cuda())
            loss = loss_sim + loss_ctm
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
    ## Make inferences
    cell_embs = torch.zeros(len(testdataset), 256)
    model.eval()
    text_encoder.eval()
    preds = torch.zeros(len(testdataset))
    sim_logits = torch.zeros(len(testdataset), text_embs.shape[-1])
    ctm_logits = torch.zeros(len(testdataset), text_embs.shape[-1])
    logits = torch.zeros(len(testdataset), text_embs.shape[-1])
    labels = torch.tensor(testdataset["label"])
    text_embs = torch.cat([text_encode(text) for text in texts], 0).T.cuda()
    with torch.no_grad():
        for i, d in tqdm(enumerate(test_loader)):
            cell_last_h, cellemb = cell_encode(
                d["input_ids"], d["attention_mask"]
            )  # batchsize * 256
            sim = (cellemb @ text_embs) / 0.05  # batchsize * class_num
            sim_logit = F.softmax(sim, dim=-1)
            # ctm
            ctm_logit = torch.zeros_like(sim_logit)
            for text_idx, text in enumerate(texts):
                text_list = [text] * sim_logit.shape[0]
                ctm_logit[:, text_idx] = ctm(
                    text_list, cell_last_h, d["attention_mask"]
                )
            ctm_logit = F.softmax(ctm_logit, dim=-1)

            sim_logits[i * args.test_batchsize : (i + 1) * args.test_batchsize] = (
                sim_logit.cpu()
            )
            ctm_logits[i * args.test_batchsize : (i + 1) * args.test_batchsize] = (
                ctm_logit.cpu()
            )
            logit = (sim_logit + ctm_logit) / 2
            pred = logit.argmax(dim=-1)
            logits[i * args.test_batchsize : (i + 1) * args.test_batchsize] = (
                logit.cpu()
            )
            cell_embs[i * args.test_batchsize : (i + 1) * args.test_batchsize] = (
                cellemb.cpu()
            )
            preds[i * args.test_batchsize : (i + 1) * args.test_batchsize] = pred.cpu()

    for k in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        preds_k = (k * sim_logits + (1 - k) * ctm_logits).argmax(dim=-1)
        print(
            k,
            "\n",
            accuracy_score(labels, preds_k),
            f1_score(labels, preds_k, average="macro"),
        )
    preds = (alpha * sim_logits + (1 - alpha) * ctm_logits).argmax(dim=-1)
    nmi = normalized_mutual_info_score(labels, preds)
    ari = adjusted_rand_score(labels, preds)
    print(f"Normalized Mutual Information (NMI) Score: {nmi}")
    print(f"Adjusted Rand Index (ARI) Score: {ari}")
    for row in confusion_matrix(labels, preds):
        print("\t".join([str(x) for x in row]))
    print(classification_report(labels, preds, digits=4, zero_division=0))
    pred_df = create_results_dataframe(preds, labels, types)
    return pred_df


def create_results_dataframe(preds, labels, types):
    label_keys = ["predicted_idx", "true_idx", "predicted_labels", "true_labels"]
    pred_df = pd.DataFrame(
        {
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
    p.add_argument(
        "--epochs", type=int, default=2, help="Fine-tuning epochs (default: 1)"
    )
    p.add_argument(
        "--nshot",
        type=int,
        default=1,
        help="Number of examples per class for training (default: 1)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--train_batchsize", type=int, default=4, help="Train batch size (default: 4)"
    )
    p.add_argument(
        "--test_batchsize", type=int, default=32, help="Test batch size (default: 32)"
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


if __name__ == "__main__":
    args = parse_args()
    logger.info("Args: %s", args)

    base_dir = Path.cwd()
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_stem = output_path.stem
    output_suffix = output_path.suffix
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device & seed
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    set_seed(args.seed)

    # Load models/tokenizer/head
    model, tokenizer, text_encoder, ctm_head = load_model_assets(base_dir, device)
    logger.info("Model assets loaded.")

    output_path = f"{curr_path}/data/output/ctm_" + str(args.nshot) + "-shot/"
    subprocess.call(f"mkdir {output_path}", shell=True)

    for annotation_key in args.annotation_keys:
        tr_loader, te_lodader, texts, t_data, tokenizer, t_enc, types, name_stub = (
            load_dataset(tokenizer, text_encoder, annotation_key, args)
        )
        print("Loading dataset finished.")

        logger.info("Dataset prepared. Classes: %d", len(types))
        o_dir_name = os.path.dirname(args.output)
        pred_df = fine_tune_model(
            model=model,
            text_encoder=t_enc,
            train_loader=tr_loader,
            test_loader=te_lodader,
            texts=texts,
            testdataset=t_data,
            types=types,
            epochs=args.epochs,
            alpha=args.alpha,
        )
        output_csv = output_dir / f"{output_stem}_{annotation_key}{output_suffix}"
        pred_df.to_csv(output_csv, index=True)
        logger.info("Langcell prediction pipeline completed successfully")
        logger.info(f"Results saved to: {args.output}")
