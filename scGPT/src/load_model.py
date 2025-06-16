from pathlib import Path
import shutil
import time
import warnings
import json

import scgpt as scgpt
from scgpt.model import TransformerModel
import numpy as np

from scgpt.tokenizer.gene_tokenizer import GeneVocab

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


logger = scgpt.logger
pad_token = "<pad>"
special_tokens = [
    pad_token,
    "<cls>",
    "<eoc>",
]  # special tokens used for padding and marking the sequence
max_seq_len = 3001
pad_value = -2
mask_value = -1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_interval = 9  # iterations
do_eval_scib_metrics = True
MLM = False  # whether to use masked language modeling, currently it is always on.
CLS = True  # celltype classification objective
ADV = False  # Adversarial training for batch correction
CCE = False  # Contrastive cell embedding objective
DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"  # https://github.com/bowang-lab/scGPT/blob/7301b51a72f5db321fccebb51bc4dd1380d99023/scgpt/model/model.py


def load_pretrained_model(tr_valid_sets, config):

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

    if config["load_model"] is not None:
        model_dir = Path(config["load_model"])
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"

        vocab = GeneVocab.from_file(vocab_file)
        shutil.copy(vocab_file, Path(tr_valid_sets["save_dir"]) / "vocab.json")
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

        adata = tr_valid_sets["adata"]

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
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        logger.info(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        MVC = config["MVC"]  # Masked value prediction for cell embedding
        ecs_threshold = config["ecs_thres"]
        explicit_zero_prob = (
            MLM and config["include_zero_gene"]
        )  # whether explicit bernoulli for zeros

    ntokens = len(vocab)  # len(tr_valid_sets["vocab"])  # size of vocabulary
    print(f"ntokens: {ntokens}", f"n_vocab: {len(vocab)}")
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=3,
        n_cls=tr_valid_sets["num_types"] if CLS else 1,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=MVC,
        do_dab=DAB,
        use_batch_labels=INPUT_BATCH_LABELS,
        num_batch_labels=tr_valid_sets["num_batch_types"],
        domain_spec_batchnorm=config["DSBN"],
        input_emb_style=input_emb_style,
        # This continuous embedding style aligns with the model's ability to handle various types of single-cell data,
        # including gene expression values, which are inherently continuous
        n_input_bins=config["n_bins"],
        cell_emb_style=cell_emb_style,
        mvc_decoder_style=mvc_decoder_style,
        ecs_threshold=ecs_threshold,
        explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=fast_transformer,
        fast_transformer_backend=fast_transformer_backend,
        pre_norm=config["pre_norm"],
    )
    if config["load_model"] is not None:
        try:
            model.load_state_dict(
                torch.load(model_file, map_location=torch.device(device))
            )
            logger.info(f"Loading all model params from {model_file}")
        except:
            # only load params that are in the model and match the size
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file, map_location=torch.device(device))
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                logger.info(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    pre_freeze_param_count = sum(
        dict(
            (p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad
        ).values()
    )

    # Freeze all pre-decoder weights
    for name, para in model.named_parameters():
        print("-" * 20)
        print(f"name: {name}")
        if config["freeze"] and "encoder" in name and "transformer_encoder" not in name:
            print(f"freezing weights for: {name}")
            para.requires_grad = False

    post_freeze_param_count = sum(
        dict(
            (p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad
        ).values()
    )

    logger.info(f"Total Pre freeze Params {(pre_freeze_param_count )}")
    logger.info(f"Total Post freeze Params {(post_freeze_param_count )}")
    return model, vocab


def train(
    model: nn.Module,
    epoch,
    tr_valid_sets,
    config,
    scaler,
    optimizer,
    scheduler,
    criterion_cls,
    vocab,
    loader: DataLoader,
) -> None:
    """
    Train the model for one epoch.
    """

    model.train()
    (
        total_loss,
        total_mse,
        total_cls,
        total_cce,
        total_mvc,
        total_ecs,
        total_dab,
        total_adv_E,
        total_adv_D,
        total_zero_log_prob,
        total_mvc_zero_log_prob,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total_error = 0.0
    start_time = time.time()
    MVC = config["MVC"]  # Masked value prediction for cell embedding
    ECS = (
        config["ecs_thres"] > 0
    )  # config.ecs_thres > 0  # Elastic cell similarity objective
    explicit_zero_prob = (
        MLM and config["include_zero_gene"]
    )  # whether explicit bernoulli for zeros
    do_sample_in_train = (
        False and explicit_zero_prob
    )  # sample the bernoulli in training
    # save_eval_interval = config["save_eval_interval"] # epochs
    vocab = tr_valid_sets["vocab"]

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)
        print(f"Training batch {batch} ...")
        if batch == 10:
            break

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with torch.cuda.amp.autocast(enabled=config["amp"]):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=(
                    batch_labels if INPUT_BATCH_LABELS or config["DSBN"] else None
                ),
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
                do_sample=do_sample_in_train,
            )

            # masked_positions = input_values.eq(mask_value)  # the postions to predict
            loss = 0.0
            metrics_to_log = {}

            if CLS:
                loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                loss = loss + loss_cls
                metrics_to_log.update({"train/cls": loss_cls.item()})

                error_rate = 1 - (
                    (output_dict["cls_output"].argmax(1) == celltype_labels)
                    .sum()
                    .item()
                ) / celltype_labels.size(0)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_cls += loss_cls.item() if CLS else 0.0
        total_zero_log_prob += loss_zero_log_prob.item() if explicit_zero_prob else 0.0
        total_mvc_zero_log_prob += (
            loss_mvc_zero_log_prob.item() if MVC and explicit_zero_prob else 0.0
        )
        total_error += error_rate
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_cls = total_cls / log_interval if CLS else 0.0
            cur_zero_log_prob = (
                total_zero_log_prob / log_interval if explicit_zero_prob else 0.0
            )
            cur_mvc_zero_log_prob = (
                total_mvc_zero_log_prob / log_interval
                if MVC and explicit_zero_prob
                else 0.0
            )
            cur_error = total_error / log_interval
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | "
                + (f"mse {cur_mse:5.2f} | mre {cur_error:5.2f} |" if MLM else "")
                + (f"cls {cur_cls:5.2f} | " if CLS else "")
                + (f"err {cur_error:5.2f} | " if CLS else "")
                + (f"cce {cur_cce:5.2f} |" if CCE else "")
                + (f"mvc {cur_mvc:5.2f} |" if MVC else "")
                + (f"ecs {cur_ecs:5.2f} |" if ECS else "")
                + (f"dab {cur_dab:5.2f} |" if DAB else "")
                + (f"adv_E {cur_adv_E:5.2f} |" if ADV else "")
                + (f"adv_D {cur_adv_D:5.2f} |" if ADV else "")
                + (f"nzlp {cur_zero_log_prob:5.2f} |" if explicit_zero_prob else "")
                + (
                    f"mvc_nzlp {cur_mvc_zero_log_prob:5.2f} |"
                    if MVC and explicit_zero_prob
                    else ""
                )
            )
            total_loss = 0
            total_mse = 0
            total_cls = 0
            total_zero_log_prob = 0
            total_mvc_zero_log_prob = 0
            total_error = 0
            start_time = time.time()


def evaluate(
    model: nn.Module,
    config,
    criterion_cls,
    criterion_dab,
    vocab,
    loader: DataLoader,
    return_raw: bool = False,
) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    predictions = []
    explicit_zero_prob = (
        MLM and config["include_zero_gene"]
    )  # whether explicit bernoulli for zeros
    do_sample_in_train = (
        False and explicit_zero_prob
    )  # sample the bernoulli in training

    with torch.no_grad():
        for batch, batch_data in enumerate(loader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)

            if batch == 10:
                break

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config["amp"]):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=(
                        batch_labels if INPUT_BATCH_LABELS or config["DSBN"] else None
                    ),
                    CLS=CLS,  # evaluation does not need CLS or CCE
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=do_sample_in_train,
                    # generative_training = False,
                )
                output_values = output_dict["cls_output"]
                loss = criterion_cls(output_values, celltype_labels)

                if DAB:
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_loss += loss.item() * len(input_gene_ids)
            accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
            total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
            total_dab += loss_dab.item() * len(input_gene_ids) if DAB else 0.0
            total_num += len(input_gene_ids)
            preds = output_values.argmax(1).cpu().numpy()
            predictions.append(preds)

    if return_raw:
        return np.concatenate(predictions, axis=0)

    return total_loss / total_num, total_error / total_num
