import os
from typing import Tuple, Dict

import numpy as np
from scgpt.tokenizer import random_mask_value
from scgpt import SubsetsBatchSampler

import torch
from torch.utils.data import Dataset, DataLoader


# input/output representation
input_style = "binned"  # "normed_raw", "log1p", or "binned"
output_style = "binned"  # "normed_raw", "log1p", or "binned"
pad_token = "<pad>"
special_tokens = [
    pad_token,
    "<cls>",
    "<eoc>",
]  # special tokens used for padding and marking the sequence
max_seq_len = 3001
pad_value = -2
per_seq_batch_sample = False
mask_value = -1  # for masked values, now it should always be auto


def prepare_data(
    tr_valid_sets, epoch, config, sort_seq_batch=False
) -> Tuple[Dict[str, torch.Tensor]]:
    mask_ratio = config["mask_ratio"]  # proportion of input tokens to mask for training
    tr_tokenized = tr_valid_sets["tokenized_train"]
    va_tokenized = tr_valid_sets["tokenized_valid"]
    masked_values_train = random_mask_value(
        tr_tokenized["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        va_tokenized["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
    )
    input_gene_ids_train, input_gene_ids_valid = (
        tr_valid_sets["tokenized_train"]["genes"],
        tr_valid_sets["tokenized_valid"]["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tr_valid_sets["tokenized_train"]["values"],
        tr_valid_sets["tokenized_valid"]["values"],
    )

    tensor_batch_labels_train = torch.from_numpy(
        tr_valid_sets["train_batch_labels"]
    ).long()
    tensor_batch_labels_valid = torch.from_numpy(
        tr_valid_sets["valid_batch_labels"]
    ).long()

    tensor_celltype_labels_train = torch.from_numpy(
        tr_valid_sets["train_celltype_labels"]
    ).long()
    tensor_celltype_labels_valid = torch.from_numpy(
        tr_valid_sets["valid_celltype_labels"]
    ).long()

    if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
        train_sort_ids = np.argsort(tr_valid_sets["train_batch_labels"])
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(tr_valid_sets["valid_batch_labels"])
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
