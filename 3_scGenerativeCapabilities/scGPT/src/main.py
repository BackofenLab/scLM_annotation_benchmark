## https://scgpt.readthedocs.io/en/latest/tutorial_annotation.html#

import copy
import os
import time
import warnings

import torch
from torch import nn

import scgpt
from scgpt.loss import (
    masked_mse_loss,
)

from process_data import create_datasets
from dataset_loader import prepare_data, prepare_dataloader
from load_model import load_pretrained_model, train, evaluate

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    hyperparameter_defaults = dict(
        seed=0,
        dataset_name="ms",
        dataset_path="../data/ms",
        do_train=True,
        load_model="../models/scGPT_human",
        mask_ratio=0.0,
        epochs=2,
        n_bins=51,
        MVC=False,  # Masked value prediction for cell embedding
        ecs_thres=0.0,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
        dab_weight=0.0,
        lr=1e-4,
        batch_size=4,
        layer_size=32,  # 128
        nlayers=1,  # 4  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead=1,  # 4  # number of heads in nn.MultiheadAttention
        dropout=0.2,  # dropout probability
        schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
        save_eval_interval=5,
        fast_transformer=True,
        pre_norm=False,
        amp=True,  # Automatic Mixed Precision. Combines 16-bit (FP16) and 32-bit (FP32) floating-point calculations.
        # FP16 reduces memory usage and speeds up computations.
        # FP32 is used when precision is critical
        include_zero_gene=False,
        freeze=False,  # freeze
        DSBN=False,  # Domain-spec batchnorm
    )

    config = hyperparameter_defaults

    # settings for input and preprocessing
    mask_ratio = config["mask_ratio"]  # proportion of input tokens to mask for training

    include_zero_gene = config[
        "include_zero_gene"
    ]  # if True, include zero genes among HGVs in the training

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

    explicit_zero_prob = (
        MLM and include_zero_gene
    )  # whether explicit bernoulli for zeros
    do_sample_in_train = (
        False and explicit_zero_prob
    )  # sample the bernoulli in training

    per_seq_batch_sample = False

    # settings for optimizer
    lr = config["lr"]  # TODO: test learning rate ratio between two tasks
    batch_size = config["batch_size"]
    eval_batch_size = config["batch_size"]
    epochs = config["epochs"]
    schedule_interval = 1
    mask_value = -1
    tr_valid_sets = create_datasets(config)

    print("Vocab size after preprocessing:", len(tr_valid_sets["vocab"]))
    print("Loading model...")
    model, vocab = load_pretrained_model(tr_valid_sets, config)

    best_val_loss = float("inf")
    best_avg_bio = 0.0
    best_model = None
    logger = scgpt.logger

    schedule_interval = 1
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
            tr_valid_sets, epoch, config, sort_seq_batch=per_seq_batch_sample
        )
        train_loader = prepare_dataloader(
            train_data_pt,
            batch_size=batch_size,
            shuffle=False,
            intra_domain_shuffle=True,
            drop_last=False,
        )
        valid_loader = prepare_dataloader(
            valid_data_pt,
            batch_size=eval_batch_size,
            shuffle=False,
            intra_domain_shuffle=False,
            drop_last=False,
        )

        if config["do_train"]:
            train(
                model,
                epoch,
                tr_valid_sets,
                config,
                scaler,
                optimizer,
                scheduler,
                criterion_cls,
                vocab,
                loader=train_loader,
            )
        val_loss, val_err = evaluate(
            model,
            config,
            criterion_cls,
            criterion_dab,
            vocab,
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
