#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LangCell-CE few-shot cell-type classification using only cell encoder.
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
    adjusted_rand_score
)
import matplotlib.pyplot as plt
import seaborn as sns  # used for heatmap

# Project-local utils
from utils import BertModel as MedBertModel
from utils import LangCellDataCollatorForCellClassification as DataCollatorForCellClassification
from utils import LangCellTranscriptomeTokenizer

from collections import Counter
import pickle
import subprocess

from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertForSequenceClassification
from transformers import Trainer
from transformers.training_args import TrainingArguments
import argparse
from utils import LangCellDataCollatorForCellClassification as DataCollatorForCellClassification
from geneformer import TranscriptomeTokenizer
import os

logger = logging.getLogger(__name__)
curr_path = os.getcwd()


def set_seed(seed: int) -> None:
    """Set RNG seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy and macro f1 using sklearn's function
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    return {
      'accuracy': acc,
      'macro_f1': macro_f1
    }


def run_model(base_dir, output_path, dataset, args):
    model_path = f'{base_dir}/data/ckpt/cell_bert'
    trainset_organ_shuffled = dataset.shuffle(seed=1)
    for label_name in ["celltype", "cell_type", "str_labels", "labels"]:
        if label_name in trainset_organ_shuffled.column_names:
            break
    trainset_organ_shuffled = trainset_organ_shuffled.rename_column(label_name,"label")
    target_names = list(Counter(trainset_organ_shuffled["label"]).keys())
    target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))

    # change labels to numerical ids
    def classes_to_ids(example):
        example["label"] = target_name_id_dict[example["label"]]
        return example
    labeled_trainset = trainset_organ_shuffled.map(classes_to_ids, num_proc=16)

    # n-shot trainset
    label_num = len(target_name_id_dict.keys())
    type2trainlist = {}
    for i in range(label_num):
        type2trainlist[i] = []
    if args.nshot >= 1:
        for i, l in enumerate(labeled_trainset["label"]):
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
    test_idx = list(set(range(len(labeled_trainset))) - set(train_idx))

    labeled_train_split = labeled_trainset.select(train_idx).shuffle(42)
    labeled_eval_split = labeled_trainset.select(test_idx).shuffle(42)
    labeled_eval_split_subset = labeled_eval_split 

    train_set = labeled_train_split
    eval_set = labeled_eval_split_subset
    label_name_id = target_name_id_dict

    max_lr = 5e-3
    geneformer_batch_size = 16
    epochs = args.epochs

    # %%
    import torch.nn as nn
    organ_trainset = train_set
    organ_evalset = eval_set
    organ_label_dict = label_name_id

    # set logging steps
    steps_per_epoch = round(len(organ_trainset)/geneformer_batch_size)
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    logging_steps = steps_per_epoch * 5

    model = BertForSequenceClassification.from_pretrained(model_path,
                                                    num_labels=len(organ_label_dict.keys()),
                                                    output_attentions = False,
                                                    output_hidden_states = False).cuda()
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    
    tk = TranscriptomeTokenizer()
    config = model.bert.config
    if config.vocab_size == len(tk.gene_token_dict) - 1:
        embedding_layer = nn.Embedding(config.vocab_size + 1, config.hidden_size, padding_idx=config.pad_token_id)
        for param, param_pretrain in zip(embedding_layer.parameters(), model.bert.embeddings.word_embeddings.parameters()):
            param.data[:-1] = param_pretrain.data
        model.bert.embeddings.word_embeddings = embedding_layer
    elif config.vocab_size != len(tk.gene_token_dict):
        raise Exception("Vocab size does not match.")

    # define output directory path
    output_dir = output_path

    # ensure not overwriting previously saved model
    saved_model_test = os.path.join(output_dir, f"pytorch_model.bin")
    if os.path.isfile(saved_model_test) == True:
        raise Exception("Model already saved to this directory.")

    # make output directory
    subprocess.call(f'mkdir {output_dir}', shell=True)

    # set training arguments
    training_args = {
        "learning_rate": max_lr,
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": "steps",
        "eval_steps": logging_steps,
        "save_total_limit": 1,
        "logging_steps": logging_steps,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        # "lr_scheduler_type": lr_schedule_fn,
        # "warmup_steps": warmup_steps,
        "weight_decay": 0.001,
        "per_device_train_batch_size": geneformer_batch_size,
        "per_device_eval_batch_size": geneformer_batch_size,
        "num_train_epochs": epochs,
        "load_best_model_at_end": False,
        "output_dir": output_dir,
    }

    training_args_init = TrainingArguments(**training_args)

    # create the trainer
    trainer = Trainer(
        model=model,
        args=training_args_init,
        data_collator=DataCollatorForCellClassification(),
        train_dataset=organ_trainset,
        eval_dataset=organ_evalset,
        compute_metrics=compute_metrics
    )
    # train the cell type classifier

    # %%
    trainer.train()
    predictions = trainer.predict(organ_evalset)
    with open(f"{output_dir}predictions.pickle", "wb") as fp:
        pickle.dump(predictions, fp)
    trainer.save_metrics("eval",predictions.metrics)
    print(f"Prediction metrics: {predictions.metrics}")
    trainer.save_model(output_dir)


def calculate_mean_max_min(adata, feature_name):
    vals = adata.obs[feature_name]
    return np.mean(vals), np.median(vals), np.max(vals), np.min(vals)


def load_dataset(args):
    """
    Read h5ad, filter/prepare, tokenize cells, map labels, stratify few-shot train/test.
    """
    dataset_sub = sc.read(args.input)
    file_name = os.path.splitext(os.path.basename(args.input))[0]

    # find highly variable genes
    sc.pp.highly_variable_genes(dataset_sub, n_top_genes=args.n_top_genes)
    dataset_sub = dataset_sub[:, dataset_sub.var.highly_variable]

    print(f"n_genes_by_counts (mean, med, max, min): {calculate_mean_max_min(dataset_sub, 'n_genes_by_counts')}")
    print(f"total_counts (mean, med, max, min): {calculate_mean_max_min(dataset_sub, 'total_counts')}")
    print(f"pct_counts_mt (mean, med, max, min): {calculate_mean_max_min(dataset_sub, 'pct_counts_mt')}")

    dataset_sub.obs["filter_pass"] = (
        (dataset_sub.obs["n_genes_by_counts"] >= 200)
    ).astype(bool)

    if sp.issparse(dataset_sub.X):
        n_counts = np.asarray(dataset_sub.X.sum(axis=1)).ravel()
        n_genes  = dataset_sub.X.getnnz(axis=1)
    else:
        n_counts = dataset_sub.X.sum(axis=1)
        n_genes  = (dataset_sub.X > 0).sum(axis=1)

    dataset_sub.obs['batch'] = dataset_sub.obs['_scvi_batch']
    dataset_sub.obs['str_labels'] = dataset_sub.obs['cell_ontology_class']
    dataset_sub.obs['celltype'] = dataset_sub.obs['cell_ontology_class']
    dataset_sub.obs['n_counts'] = n_counts
    f_ensembl_ids = dataset_sub.var['ensembl_id'].str.replace(r'\.\d+$','', regex=True)
    dataset_sub.var['feature_id'] = f_ensembl_ids 
    dataset_sub.var['ensembl_id'] = f_ensembl_ids
    dataset_sub.obs["n_genes"]  = np.asarray(n_genes, dtype=np.int32)

    tk = LangCellTranscriptomeTokenizer(dict([(k, k) for k in dataset_sub.obs.keys()]), nproc=4)
    tokenized_cells, cell_metadata = tk.tokenize_anndata(dataset_sub)
    tokenized_dataset = tk.create_dataset(tokenized_cells, cell_metadata)

    return tokenized_dataset


def plot_umap(cell_embs, dataset_sub, preds, types, file_name, mode, o_path, nmi, ari):
    """
    Plot UMAP
    """
    plt.tight_layout()
    cell_embs_ad = ad.AnnData(cell_embs.numpy())
    cell_embs_ad.obs['celltype'] = dataset_sub['celltype']
    cell_embs_ad.obs['predictions'] = [types[i] for i in preds]
    sc.pp.neighbors(cell_embs_ad, use_rep='X', n_neighbors=80)
    sc.tl.umap(cell_embs_ad)
    sc.pl.umap(cell_embs_ad, color=['celltype', 'predictions'], legend_fontsize ='xx-small', size=5, legend_fontweight='light')
    save_path = f'{o_path}/umap_plot_{file_name}_{mode}.png'
    plt.title(f'UMAP plot of {file_name} ({mode}): NMI={nmi:.2f}, ARI={ari:.2f}')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'UMAP plot saved to {save_path}')


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
    plt.title(title or ("Normalized confusion matrix" if normalize else "Confusion matrix"))
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    out_path = f'{out_dir}/confusion_matrix_{file_n}_{mode}.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f'Saved confusion matrix: {out_path}')


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    curr_path = Path.cwd()
    default_input = curr_path.parents[3] / "util" / "test_adata.h5ad"
    default_output = curr_path / "langcell_predictions.csv"

    p = argparse.ArgumentParser(description="LangCell few/zero-shot prediction for single-cell data")
    p.add_argument("-i", "--input", type=Path, default=default_input, help=f"Input h5ad file path (default: {default_input})")
    p.add_argument("-o", "--output", type=Path, default=default_output, help=f"Output CSV file path (default: {default_output})")
    p.add_argument("--epochs", type=int, default=2, help="Fine-tuning epochs (default: 1)")
    p.add_argument("--nshot", type=int, default=2, help="Number of examples per class for training (default: 1)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--train_batchsize", type=int, default=4, help="Train batch size (default: 4)")
    p.add_argument("--test_batchsize", type=int, default=32, help="Test batch size (default: 32)")
    p.add_argument("--n_top_genes", type=int, default=3000, help="Number of HVGs (default: 3000)")
    p.add_argument("--alpha", type=float, default=0.1, help="Fusion weight for SIM logits (default: 0.1)")
    p.add_argument("--device", type=str, default=None, help="torch device, e.g. 'cuda:0' or 'cpu' (default: auto)")
    return p.parse_args()


if __name__ == "__main__":

    args = parse_args()
    logger.info("Args: %s", args)

    base_dir = Path.cwd()
    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device & seed
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # Load models/tokenizer/head
    #run_model(base_dir, device)

    #output_path = f'{curr_path}/data/output/ctm_' + str(args.nshot) + '-shot/'
    #subprocess.call(f'mkdir {output_path}', shell=True)

    tokenised_dataset = load_dataset(args)
    print("Loading dataset finished.")

    o_dir_name = os.path.dirname(args.output)
    pred_df = run_model(base_dir, out_dir, tokenised_dataset, args)
    pred_df.to_csv(args.output, index=True)