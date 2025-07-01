from pathlib import Path
import time
import shutil
from collections import namedtuple

import numpy as np

import scanpy as sc
import scgpt as scg
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value

from sklearn.model_selection import train_test_split
from scipy.sparse import issparse

logger = scg.logger


def process_tabula_sapiens_data(config):
    print("Loading and processing Tabula Sapiens data...")
    file_path = "../data/tabula_sapiens_v2/Kidney_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad"  # Adjust the path as needed
    adata = sc.read_h5ad(file_path)
    filter_gene_by_counts = True
    data_is_raw = False
    n_bins = 51
    hvg = 3000
    pad_token = "<pad>"
    pad_value=-2
    special_tokens=["<pad>", "<cls>", "<eoc>"]
    max_seq_len = 3001
    include_zero_gene=False


    # Display basic information about the dataset
    print(adata)
    print(adata.X)
    print(adata.X.T)
    print(adata.obs.head())
    print(adata.var.head())

    #print("Is log-normalized" ) #np.allclose( adata.X, adata.layers["log_normalized"]
    print(adata.X.shape)

    print("Cell types")
    celltype_key = "cell_ontology_class"
    print(len(adata.obs[celltype_key]))
    print(adata.obs[celltype_key].unique())

    batch_id_labels = adata.obs["_scvi_batch"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels
    celltype_id_labels = adata.obs[celltype_key].astype("category").cat.codes.values
    celltypes = adata.obs[celltype_key].unique()
    num_types = len(np.unique(celltype_id_labels))
    id2type = dict(enumerate(adata.obs[celltype_key].astype("category").cat.categories))
    adata.obs["celltype_id"] = celltype_id_labels
    adata.var["gene_name"] = adata.var.index.tolist()

    #print(adata.obs["celltype"])
    #print(adata.X)

    model_dir = "../models/scGPT_human/"
    dataset_name = "Kidney"
    save_dir = Path(f"../data/save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    vocab_file = model_dir + "vocab.json"
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
    print("After filtering genes not in vocab:")
    print(adata.X)


    print("Preprocessing ...")
    # set up the preprocessor, use the args to config the workflow
    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=filter_gene_by_counts,  # step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=1e4,  # step 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=data_is_raw,  # step 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=hvg,  # step 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=n_bins,  # step 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )

    preprocessor(adata, batch_key=None)


    genes = adata.var["gene_name"].tolist()

    celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    num_batch_types = len(set(batch_ids))
    batch_ids = np.array(batch_ids)
    all_counts = adata.layers['log_normalized'].A if issparse(adata.layers['log_normalized']) else adata.layers['log_normalized']   

    (
        train_data,
        valid_data,
        train_celltype_labels,
        valid_celltype_labels,
        train_batch_labels,
        valid_batch_labels,
    ) = train_test_split(
        all_counts, celltypes_labels, batch_ids, test_size=0.2, shuffle=True
    )

    print(
        train_data.shape,
        train_batch_labels.shape,
        valid_data.shape,
        valid_batch_labels.shape,
    )

    #if config["load_model"] is None:
    #    vocab = Vocab(VocabPybind(genes + special_tokens, None))
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
        adata,
        adata,
        celltypes,
        id2type,
        gene_ids,
        num_types,
        num_batch_types,
        logger,
        celltype_key
    )


#if __name__ == "__main__":

#   file_path = "../data/tabula_sapiens_v2/Kidney_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad"  # Adjust the path as needed
#   adata = read_tabula_sapiens_data(file_path)
    #print(adata)
    #print(adata.X)
    #print(adata.X.T)
    #print(adata.obs.head())
    #print(adata.var.head())
    