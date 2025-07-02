## https://scgpt.readthedocs.io/en/latest/tutorial_annotation.html#

import copy
import os
import sys
import time
import warnings

from pathlib import Path

warnings.filterwarnings("ignore")

import torch
from torch import nn
import scanpy as sc

import scgpt as scg
from scgpt.loss import masked_mse_loss

from dataset_loader import prepare_anndata, prepare_data, prepare_dataloader
from process_tabula_sapiens import process_tabula_sapiens_data
from model_utils import load_pretrained_model, train_model, evaluate_model
from make_predictions import test
from analyse_prediction import make_confusion_plot, make_umap_plot

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    config = dict(
        seed=0,
        dataset_name="ms",
        do_train=False,
        load_model="../models/scGPT_human",
        mask_ratio=0.0,
        mask_value=-1,  # for masked values, now it should always be auto
        epochs=1,
        n_bins=51,
        MVC=False,  # Masked value prediction for cell embedding
        ecs_thres=0.0,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
        dab_weight=0.0,
        lr=1e-4,
        batch_size=16,
        layer_size=128,
        nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead=4,  # number of heads in nn.MultiheadAttention
        dropout=0.2,  # dropout probability
        schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
        save_eval_interval=5,
        fast_transformer=True,
        pre_norm=False,
        amp=True,  # Automatic Mixed Precision. Combines 16-bit (FP16) and 32-bit (FP32) floating-point calculations.
        # FP16 reduces memory usage and speeds up computations.
        # FP32 is used when precision is critical
        include_zero_gene=False,
        freeze=True,  # freeze
        DSBN=False,  # Domain-spec batchnorm
        pad_token="<pad>",
        special_tokens=["<pad>", "<cls>", "<eoc>"],
        pad_value=-2,
        per_seq_batch_sample=False,
        input_style="binned",
        max_seq_len=3001,
        log_interval=50,
    )

    include_zero_gene = config[
        "include_zero_gene"
    ]  # if True, include zero genes among HGVs in the training
    n_bins = config["n_bins"]  # Number of bins for discretizing gene expression values

    # settings for training
    MLM = False  # whether to use masked language modeling, currently it is always on.
    CLS = True  # celltype classification objective
    ADV = False  # Adversarial training for batch correction
    CCE = False  # Contrastive cell embedding objective
    MVC = config["MVC"]  # Masked value prediction for cell embedding
    ECS = (
        config["ecs_thres"] > 0
    )  # config.ecs_thres > 0  # Elastic cell similarity objective
    DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
    INPUT_BATCH_LABELS = (
        False  # TODO: have these help MLM and MVC, while not to classifier
    )
    input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
    cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
    adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
    adv_D_delay_epochs = 0
    mvc_decoder_style = "inner product"  # https://github.com/bowang-lab/scGPT/blob/7301b51a72f5db321fccebb51bc4dd1380d99023/scgpt/model/model.py
    ecs_threshold = config["ecs_thres"]
    dab_weight = config["dab_weight"]

    # settings for optimizer
    lr = config["lr"]  # TODO: test learning rate ratio between two tasks
    batch_size = config["batch_size"]
    eval_batch_size = config["batch_size"]
    epochs = config["epochs"]
    schedule_interval = 1

    # settings for the model
    fast_transformer = config[
        "fast_transformer"
    ]  # Whether to use a faster Transformer implementation.
    fast_transformer_backend = "flash"  # "linear" or "flash"
    embsize = config["layer_size"]  # embedding dimension
    d_hid = config[
        "layer_size"
    ]  # dimension of the feedforward network in TransformerEncoder
    nlayers = config[
        "nlayers"
    ]  # number of TransformerEncoderLayer in TransformerEncoder
    nhead = config["nhead"]  # number of heads in nn.MultiheadAttention
    dropout = config["dropout"]  # dropout probability

    save_eval_interval = config["save_eval_interval"]  # epochs
    do_eval_scib_metrics = True
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins

    # work with default scGPT datasets
    #proc_datasets = prepare_anndata(config)

    # work with Tabula Sapiens data
    proc_datasets = process_tabula_sapiens_data(config)

    logger = proc_datasets.logger

    model = load_pretrained_model(proc_datasets, config)
    model.to(device)

    best_val_loss = float("inf")
    best_avg_bio = 0.0
    best_model = None

    criterion = masked_mse_loss
    criterion_cls = nn.CrossEntropyLoss()
    criterion_dab = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, eps=1e-4 if config["amp"] else 1e-8
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, schedule_interval, gamma=config["schedule_ratio"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])

    for epoch in range(1, epochs + 1):

        epoch_start_time = time.time()
        train_data_pt, valid_data_pt = prepare_data(
            proc_datasets, config, sort_seq_batch=False
        )
        train_loader = prepare_dataloader(
            train_data_pt,
            batch_size=batch_size,
            shuffle=False,
            intra_domain_shuffle=True,
            drop_last=False,
            per_seq_batch_sample=config["per_seq_batch_sample"],
        )
        valid_loader = prepare_dataloader(
            valid_data_pt,
            batch_size=eval_batch_size,
            shuffle=False,
            intra_domain_shuffle=False,
            drop_last=False,
            per_seq_batch_sample=config["per_seq_batch_sample"],
        )

        if config["do_train"]:
            print("Training scGPT model...")
            model = train_model(
                proc_datasets,
                config,
                model,
                optimizer,
                scheduler,
                scaler,
                criterion_cls,
                epoch,
                loader=train_loader,
            )

        print("Evaluating scGPT model...")
        val_loss, val_err = evaluate_model(
            proc_datasets,
            config,
            model,
            criterion_cls,
            criterion_dab,
            loader=valid_loader,
        )
        elapsed = time.time() - epoch_start_time
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {val_loss:5.4f} | err {val_err:5.4f}"
        )
        logger.info("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            logger.info(f"Best model with score {best_val_loss:5.4f}")

    scheduler.step()

    print("Test the fine-tuned model...")
    predictions, labels, results = test(
        model, proc_datasets, config, criterion_cls, criterion_dab
    )

    # make evaluation plots
    make_umap_plot(predictions, labels, results, proc_datasets)
    make_confusion_plot(predictions, labels, proc_datasets)
