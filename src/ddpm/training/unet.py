import argparse
import io
import os
import random
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.tuner.tuning import Tuner
from torch.utils.data import DataLoader, random_split
from torchmetrics.aggregation import SumMetric
from torchmetrics.image.fid import FrechetInceptionDistance

from datasets.mnist.datamodules import MNISTDataModule
from ddpm.models.two_dimensions.unet import Unet as Unet2d
from ddpm.utils import get_generator


# TODO: Move into plot utils
def _edit_axis(ax): 
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])


class UnetLightningModule(L.LightningModule):

    def __init__(
        self,
        dim: int = 20,
        dim_mults: List[int] = [1, 2, 4, 8],
        resnet_block_groups: int = 4,
        sinusoidal_pos_emb_theta: int = 10000,
        context_dim: int = 2,
        learning_rate: float = 1e-4,
        seed: int = 0,
        fid_features: Literal[64, 192, 768, 2048] = 192,
        channels: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.unet = Unet2d(
            dim=self.hparams.dim,
            dim_mults=self.hparams.dim_mults,
            channels=self.hparams.channels,
            resnet_block_groups=self.hparams.resnet_block_groups,
            sinusoidal_pos_emb_theta=self.hparams.sinusoidal_pos_emb_theta,
            context_dim=self.hparams.context_dim,
        )
        self.fid = FrechetInceptionDistance(
            feature=self.hparams.fid_features,
            normalize=True,
            compute_on_cpu=True,
        )

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def _fix_dimensions(self, batch):
        data = batch["image"]
        batch_size = data.shape[0]
        conditions = batch["label"].float().reshape(batch_size, self.hparams.context_dim, 1, 1)
        assert len(data.shape) == 4 and \
            len(conditions.shape) == 4 and \
            data.shape[1] == self.hparams.channels and \
            conditions.shape[1] == self.hparams.context_dim
        return data, conditions, batch_size

    def training_step(self, batch, batch_idx):
        data, conditions, batch_size = self._fix_dimensions(batch)
        x = self.unet(x=data, time=None, cond=conditions)
        loss = F.mse_loss(x, data)
        return loss

    def on_validation_start(self):
        self.fid = self.fid.to(self.device)
        self.val_generator = get_generator(self.hparams.seed, self.device)
        self.mse_sum = SumMetric().to(self.device)
        self.counter = SumMetric().to(self.device)

    def validation_step(self, batch, batch_idx):
        data, conditions, batch_size = self._fix_dimensions(batch)
        self.fid.update((data.detach() * 0.3801 + 0.1306).repeat(1, 3, 1, 1), real=True)

        recons = self.unet(
            x=data,
            time=None,
            cond=conditions,
        ).detach()
        mse_sum = F.mse_loss(data * 0.3801 + 0.1306, recons * 0.3801 + 0.1306, reduce="sum")
        self.fid.update((recons.detach() * 0.3801 + 0.1306).repeat(1, 3, 1, 1), real=False)
        self.mse_sum.update(mse_sum)
        self.counter.update(batch_size)

        if batch_idx == 0:
            n_samples = 20
            fig, axes = plt.subplots(nrows=n_samples, ncols=3, figsize=(5, 20))
            axes[0, 0].set_title("Orig.")
            axes[0, 1].set_title("Recon.")
            axes[0, 2].set_title("Diff.")
            for i in range(n_samples):
                data_sample = np.moveaxis(data[i].cpu().numpy(), 0, 2) * 0.3801 + 0.1306
                recon_sample = np.moveaxis(recons[i].cpu().numpy(), 0, 2) * 0.3801 + 0.1306
                diff = np.abs(data_sample - recon_sample).mean(axis=2)
                axes[i, 0].imshow(data_sample)
                axes[i, 1].imshow(recon_sample)
                axes[i, 2].imshow(diff, cmap="Reds", interpolation="nearest")
                for ax in axes[i]:
                    _edit_axis(ax)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            samples_img = np.array(Image.open(buf).convert("RGB")).astype(np.uint8)
            self.logger.experiment.add_image("Reconstruction", samples_img, self.global_step, dataformats="HWC")

    def on_validation_epoch_end(self):
        fid_score = self.fid.compute()
        mse = self.mse_sum.compute() / self.counter.compute()
        log_dict = {"val/mse": mse.item(), "val/fid": fid_score.item()}
        self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True)
        self.fid.reset()
        return log_dict

    def configure_optimizers(self):
        return torch.optim.Adam(self.unet.parameters(), lr=self.hparams.learning_rate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-variant", type=str, choices=["grey", "colour"], default="grey")
    parser.add_argument("--logdir", type=str, default="/vol/biomedic3/rrr2417/cxr-generation/experiments_mnist_unet")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dim-mults", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--no-tuner", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=200)
    args = parser.parse_args()

    seed_everything(args.seed, workers=True)

    if args.mnist_variant == "grey":
        dm = MNISTDataModule(
            data_dir="src/datasets/mnist/files/raw",
            seed=args.seed,
            split_ratio=(0.9, 0.1),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        channels = 1
    elif args.mnist_variant == "colour":
        channels = 3
        raise NotImplementedError()
    else:
        raise ValueError(f"mnist_variant must be {'grey', 'colour'}, not {args.mnist_variant}")
    context_dim = 10

    model = UnetLightningModule(
        dim=args.dim,
        dim_mults=args.dim_mults,
        seed=args.seed,
        channels=channels,
        fid_features=64,
        context_dim=context_dim,
    )
    tb_logger = TensorBoardLogger(save_dir=args.logdir)
    checkpoint_callback = ModelCheckpoint(save_last=True)
    trainer = Trainer(
        accelerator="gpu",
        devices=args.device_id,
        strategy="auto" if args.device_id != -1 else DDPStrategy(find_unused_parameters=True),
        deterministic=True,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=10,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=-1,
    )

    # Tuning batch size is unsupported with distributed strategies in lightning 2.2.4
    if args.device_id != -1 and not args.no_tuner:
        tuner = Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=dm, mode="power")

    trainer.fit(model, datamodule=dm)


def visualise_cross_attn():
    from ddpm.training.callbacks import CrossAttentionAggregateByDimCallback

    v = 7
    ckpt = f"/vol/biomedic3/rrr2417/cxr-generation/experiments_mnist_unet/lightning_logs/version_{v}/checkpoints/last.ckpt"
    model = UnetLightningModule.load_from_checkpoint(ckpt)

    bs = 25
    dm = MNISTDataModule(
        data_dir="src/datasets/mnist/files/raw",
        seed=0,
        split_ratio=(0.9, 0.1),
        batch_size=bs,
    ) 
    dm.setup("test")
    for data in dm.test_dataloader():
        data, conditions, batch_size = model._fix_dimensions(data)
        data = data.to(model.device)
        conditions = conditions.to(model.device)
        break

    agg_callback = CrossAttentionAggregateByDimCallback()
    recon = model.unet(
        x=data,
        time=None,
        cond=conditions,
        cross_attn_kwargs={"callbacks": [agg_callback]},
    )

    aggs = agg_callback.aggregate()
    fig, axes = plt.subplots(
        nrows=len(aggs),
        ncols=model.hparams.context_dim + 1,
        gridspec_kw={"hspace": 0.1, "wspace": 0.1},
    )
    for i, (k, v) in enumerate(aggs.items()):
        axes[i, 0].imshow(np.moveaxis(data[i].cpu().numpy(), 0, 2) * 0.3801 + 0.1306)
        axes[i, 0].set(ylabel=f"{k} x {k}")
        print("v:", v.shape)
        for  _v, ax in zip(v, axes[i, 1:].flatten()):
            ax.imshow(_v)
            _edit_axis(ax)
    fig.savefig(
        f"/vol/biomedic3/rrr2417/cxr-generation/experiments_mnist_unet/lightning_logs/version_{v}/cs_maps.pdf",
        bbox_inches="tight",
    )
