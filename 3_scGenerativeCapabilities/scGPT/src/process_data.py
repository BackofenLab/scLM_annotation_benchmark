## https://scgpt.readthedocs.io/en/latest/tutorial_annotation.html#

from pathlib import Path
import shutil

import time
import scanpy as sc
import numpy as np
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

import scgpt
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor


# input/output representation
input_style = "binned"  # "normed_raw", "log1p", or "binned"
output_style = "binned"  # "normed_raw", "log1p", or "binned"
pad_token = "<pad>"
pad_value = -2
special_tokens = [
    pad_token,
    "<cls>",
    "<eoc>",
]  # special tokens used for padding and marking the sequence
max_seq_len = 3001



def create_datasets(config):

    n_bins = config["n_bins"]  # Number of bins for discretizing gene expression values
    dataset_name = config["dataset_name"]
    save_dir = Path(f"../data/save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    logger = scgpt.logger
    scgpt.utils.add_file_handler(logger, save_dir / "run.log")

    if dataset_name == "ms":
        data_dir = Path(config["dataset_path"])
        adata = sc.read(data_dir / "c_data.h5ad")
        adata_test = sc.read(data_dir / "filtered_ms_adata.h5ad")
        # cell types
        adata.obs["celltype"] = adata.obs[
            "Factor Value[inferred cell type - authors labels]"
        ].astype("category")
        adata_test.obs["celltype"] = adata_test.obs[
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

    celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
    celltypes = adata.obs["celltype"].unique()
    num_types = len(np.unique(celltype_id_labels))
    id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))

    adata.obs["celltype_id"] = celltype_id_labels
    adata.var["gene_name"] = adata.var.index.tolist()
    adata.X = adata.X

    print(adata.X.shape)

    if config["load_model"] is not None:
        model_dir = Path(config["load_model"])
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
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

        # model
        # with open(model_config_file, "r") as f:
        #    model_configs = json.load(f)
        logger.info(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )

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

    vocab = Vocab(VocabPybind(genes + special_tokens, None))
    # bidirectional lookup [gene <-> int]
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)
    include_zero_gene = config["include_zero_gene"]

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

    tr_valid_sets = {
        "tokenized_train": tokenized_train,
        "tokenized_valid": tokenized_valid,
        "train_celltype_labels": train_celltype_labels,
        "valid_celltype_labels": valid_celltype_labels,
        "train_batch_labels": train_batch_labels,
        "valid_batch_labels": valid_batch_labels,
        "vocab": vocab,
        "save_dir": str(save_dir),
        "num_types": num_types,
        "num_batch_types": num_batch_types,
        "adata": adata,
        "adata_test": adata_test,
        "adata_test_raw": adata_test_raw,
        "celltypes": celltypes,
        "id2type": id2type,
        "gene_ids": gene_ids
    }

    return tr_valid_sets