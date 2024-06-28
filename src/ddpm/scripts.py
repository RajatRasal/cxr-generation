import argparse
import os
from itertools import product
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.tuner.tuning import Tuner
from torchmetrics import Accuracy

from classification.mnist import MNISTClassifierLightningModule
from datasets.mnist.datamodules import get_mnist_variant
from ddpm.diffusion.generative_classifier import classify
from ddpm.editing.null_text_inversion import NullTokenOptimisation
from ddpm.training.conditional import GuidedDiffusionLightningModule
from ddpm.training.unconditional import DiffusionLightningModule
from ddpm.utils import get_generator
from semantic_editing.attention import set_attention_processor


def mnist_train_unconditional():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-variant", type=str, choices=["grey", "colour"], default="grey")
    parser.add_argument("--logdir", type=str, default="/vol/biomedic3/rrr2417/cxr-generation/experiments_mnist_diffusers_condition_concat")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--train-timesteps", type=int, default=200)
    parser.add_argument("--dim-mults", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--no-tuner", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--beta-schedule", type=str, choices=["linear", "scaled_linear", "cosine", "sigmoid"], default="linear")
    parser.add_argument("--rescale-betas-zero-snr", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--layers-per-block", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--val-freq", type=int, default=10)
    parser.add_argument("--blocks", nargs="+", choices=["Block", "ResnetBlock", "AttnBlock"], default=["Block", "Block", "Block"])
    args = parser.parse_args()

    seed_everything(args.seed, workers=True)

    channels = {"grey": 1, "colour": 3}[args.mnist_variant]
    # TODO: Save dataset info somewhere - mnist_variant, seed
    dm = get_mnist_variant(args.mnist_variant, args.seed, args.batch_size, args.num_workers)

    model = DiffusionLightningModule(
        dim=args.dim,
        dim_mults=args.dim_mults,
        blocks=args.blocks,
        train_timesteps=args.train_timesteps,
        beta_schedule=args.beta_schedule,
        seed=args.seed,
        channels=channels,
        rescale_betas_zero_snr=args.rescale_betas_zero_snr,
        learning_rate=args.learning_rate,
        layers_per_block=args.layers_per_block,
        dropout=args.dropout,
    )
    tb_logger = TensorBoardLogger(save_dir=args.logdir)
    checkpoint_callback = ModelCheckpoint(save_last=True)
    trainer = Trainer(
        accelerator="gpu",
        devices=args.device_id,
        strategy="auto" if args.device_id != -1 else DDPStrategy(find_unused_parameters=True),
        deterministic=True,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.val_freq,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=1,
    )

    # Tuning batch size is unsupported with distributed strategies in lightning 2.2.4
    if args.device_id != -1 and not args.no_tuner:
        tuner = Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=dm, mode="power")

    trainer.fit(model, datamodule=dm)


def mnist_test_unconditional():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="/vol/biomedic3/rrr2417/cxr-generation/experiments_mnist_diffusers_condition_concat")
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)

    ckpt_path = os.path.join(args.logdir, f"lightning_logs/version_{args.version}/checkpoints/last.ckpt")
    model = DiffusionLightningModule.load_from_checkpoint(ckpt_path)

    seed_everything(model.hparams.seed)

    # TODO: Load colour from a file
    dm = get_mnist_variant("grey", args.seed, args.batch_size, args.num_workers)

    tb_logger = TensorBoardLogger(save_dir=args.logdir, version=args.version)
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        deterministic=True,
        logger=tb_logger,
    )
    trainer.test(model, dm)


def mnist_train_guided():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-variant", type=str, choices=["grey", "colour"], default="grey")
    parser.add_argument("--logdir", type=str, default="/vol/biomedic3/rrr2417/cxr-generation/experiments_mnist_diffusers_condition_concat")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--train-timesteps", type=int, default=200)
    parser.add_argument("--dim-mults", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--no-tuner", action="store_true")
    parser.add_argument("--use-cross-attn", action="store_true")
    parser.add_argument("--cross-attention-dim", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--beta-schedule", type=str, choices=["linear", "scaled_linear", "cosine", "sigmoid"], default="linear")
    parser.add_argument("--rescale-betas-zero-snr", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--layers-per-block", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--val-freq", type=int, default=10)
    parser.add_argument("--blocks", nargs="+", choices=["Block", "ResnetBlock", "AttnBlock", "CrossAttnBlock"], default=["Block", "Block", "Block"])
    parser.add_argument("--uncond-prob", type=float, default=0.2)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-sanity-val-steps", type=int, default=-1)
    parser.add_argument("--norm-num-groups", type=int, default=32)
    args = parser.parse_args()

    seed_everything(args.seed, workers=True)

    channels = {"grey": 1, "colour": 3}[args.mnist_variant]
    # TODO: Save dataset info somewhere - mnist_variant, seed
    dm = get_mnist_variant(args.mnist_variant, args.seed, args.batch_size, args.num_workers)

    model = GuidedDiffusionLightningModule(
        dim=args.dim,
        dim_mults=args.dim_mults,
        blocks=args.blocks,
        train_timesteps=args.train_timesteps,
        beta_schedule=args.beta_schedule,
        seed=args.seed,
        channels=channels,
        rescale_betas_zero_snr=args.rescale_betas_zero_snr,
        learning_rate=args.learning_rate,
        layers_per_block=args.layers_per_block,
        dropout=args.dropout,
        uncond_prob=args.uncond_prob,
        use_cross_attn=args.use_cross_attn,
        cross_attention_dim=args.cross_attention_dim,
        embedding_dim=args.embedding_dim,
        classes=10,
        norm_num_groups=args.norm_num_groups,
    )
    tb_logger = TensorBoardLogger(save_dir=args.logdir)
    checkpoint_callback = ModelCheckpoint(save_last=True)
    trainer = Trainer(
        accelerator="gpu",
        devices=args.device_id,
        strategy=DDPStrategy(find_unused_parameters=True) if args.device_id > 1 else "auto",
        deterministic=True,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.val_freq,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=args.num_sanity_val_steps,
    )

    # Tuning batch size is unsupported with distributed strategies in lightning 2.2.4
    if args.device_id != -1 and not args.no_tuner:
        tuner = Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=dm, mode="power")

    trainer.fit(model, datamodule=dm)
