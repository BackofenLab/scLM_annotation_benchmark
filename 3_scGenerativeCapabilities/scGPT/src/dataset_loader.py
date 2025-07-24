import os
import shutil
import time
from typing import Dict
import warnings

warnings.filterwarnings("ignore")

from collections import namedtuple
from pathlib import Path

import torch
import numpy as np

from scgpt.tokenizer.gene_tokenizer import GeneVocab

## https://scgpt.readthedocs.io/en/latest/tutorial_annotation.html#

import scanpy as sc
from scipy.sparse import issparse
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

import scgpt as scg
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value


def prepare_anndata(config):

    dataset_name = config["dataset_name"]
    special_tokens = config["special_tokens"]
    pad_token = config["pad_token"]
    pad_value = config["pad_value"]
    n_bins = config["n_bins"]
    include_zero_gene = config["include_zero_gene"]
    input_style = config["input_style"]  # "normed_raw", "log1p", or "binned"
    max_seq_len = config["max_seq_len"]

    save_dir = Path(f"../data/save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    celltype_key = "celltype"  # default cell type key, can be changed later

    if dataset_name == "ms":
        data_dir = Path("../data/ms")
        adata = sc.read(data_dir / "c_data.h5ad")
        adata_test = sc.read(data_dir / "filtered_ms_adata.h5ad")
        # cell types
        adata.obs[celltype_key] = adata.obs[
            "Factor Value[inferred cell type - authors labels]"
        ].astype("category")
        adata_test.obs[celltype_key] = adata_test.obs[
            "Factor Value[inferred cell type - authors labels]"
        ].astype("category")

        adata.obs["batch_id"] = adata.obs["str_batch"] = "0"  # training data
        print("Train data: ", adata.shape)
        adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"  # test data
        print("Test data: ", adata_test.shape)
        adata.var.set_index(adata.var["gene_name"], inplace=True)
        adata_test.var.set_index(adata.var["gene_name"], inplace=True)
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test_raw = adata_test.copy()
        adata = adata.concatenate(adata_test, batch_key="str_batch")

    # make the batch category column
    batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels
    celltype_id_labels = adata.obs[celltype_key].astype("category").cat.codes.values
    celltypes = adata.obs[celltype_key].unique()
    num_types = len(np.unique(celltype_id_labels))
    id2type = dict(enumerate(adata.obs[celltype_key].astype("category").cat.categories))
    adata.obs["celltype_id"] = celltype_id_labels
    adata.var["gene_name"] = adata.var.index.tolist()

    print(adata.obs[celltype_key])
    print(adata.X)

    if config["load_model"] is not None:
        model_dir = Path(config["load_model"])
        vocab_file = model_dir / "vocab.json"
        vocab = GeneVocab.from_file(vocab_file)
        print(vocab)
        shutil.copy(vocab_file, save_dir / "vocab.json")
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        adata = adata[:, adata.var["id_in_vocab"] >= 0]

    ###################################################################3

    # set up the preprocessor, use the args to config the workflow
    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=filter_gene_by_counts,  # step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=1e4,  # step 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=data_is_raw,  # step 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=False,  # step 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=n_bins,  # step 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )

    adata_test = adata[adata.obs["str_batch"] == "1"]
    adata = adata[adata.obs["str_batch"] == "0"]

    preprocessor(adata, batch_key=None)
    preprocessor(adata_test, batch_key=None)
    print("Train data: ", adata.shape)
    print("Test data: ", adata_test.shape)

    input_layer_key = (
        {  # the values of this map coorespond to the keys in preprocessing
            "normed_raw": "X_normed",
            "log1p": "X_normed",
            "binned": "X_binned",
        }[input_style]
    )
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    genes = adata.var["gene_name"].tolist()

    celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    num_batch_types = len(set(batch_ids))
    batch_ids = np.array(batch_ids)

    (
        train_data,
        valid_data,
        train_celltype_labels,
        valid_celltype_labels,
        train_batch_labels,
        valid_batch_labels,
    ) = train_test_split(
        all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
    )

    print(
        train_data.shape,
        train_batch_labels.shape,
        valid_data.shape,
        valid_batch_labels.shape,
    )

    if config["load_model"] is None:
        vocab = Vocab(VocabPybind(genes + special_tokens, None))
    # bidirectional lookup [gene <-> int]
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)
    print(gene_ids, special_tokens)

    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=include_zero_gene,
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=include_zero_gene,
    )
    logger.info(
        f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
    )
    logger.info(
        f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
    )

    PreprocessedData = namedtuple(
        "PreprocessedData",
        [
            "tokenized_train",
            "tokenized_valid",
            "train_celltype_labels",
            "valid_celltype_labels",
            "train_batch_labels",
            "valid_batch_labels",
            "vocab",
            "save_dir",
            "adata",
            "adata_test",
            "adata_test_raw",
            "celltypes",
            "id2type",
            "gene_ids",
            "num_types",
            "num_batch_types",
            "logger",
            "celltype_key"
        ],
    )

    return PreprocessedData(
        tokenized_train,
        tokenized_valid,
        train_celltype_labels,
        valid_celltype_labels,
        train_batch_labels,
        valid_batch_labels,
        vocab,
        str(save_dir),
        adata,
        adata_test,
        adata_test_raw,
        celltypes,
        id2type,
        gene_ids,
        num_types,
        num_batch_types,
        logger,
        celltype_key
    )


def prepare_data(proc_datasets, config, sort_seq_batch=False):
    mask_value = config["mask_value"]
    pad_value = config["pad_value"]
    masked_values_train = random_mask_value(
        proc_datasets.tokenized_train["values"],
        mask_ratio=config["mask_ratio"],
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        proc_datasets.tokenized_valid["values"],
        mask_ratio=config["mask_ratio"],
        mask_value=mask_value,
        pad_value=pad_value,
    )

    input_gene_ids_train, input_gene_ids_valid = (
        proc_datasets.tokenized_train["genes"],
        proc_datasets.tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        proc_datasets.tokenized_train["values"],
        proc_datasets.tokenized_valid["values"],
    )
    tensor_batch_labels_train = torch.from_numpy(
        proc_datasets.train_batch_labels
    ).long()
    tensor_batch_labels_valid = torch.from_numpy(
        proc_datasets.valid_batch_labels
    ).long()
    tensor_celltype_labels_train = torch.from_numpy(
        proc_datasets.train_celltype_labels
    ).long()
    tensor_celltype_labels_valid = torch.from_numpy(
        proc_datasets.valid_celltype_labels
    ).long()

    if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
        train_sort_ids = np.argsort(proc_datasets.train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(proc_datasets.valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "celltype_labels": tensor_celltype_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
        "celltype_labels": tensor_celltype_labels_valid,
    }

    return train_data_pt, valid_data_pt


# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    per_seq_batch_sample: bool = False,
) -> DataLoader:
    if num_workers == 0:
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader
