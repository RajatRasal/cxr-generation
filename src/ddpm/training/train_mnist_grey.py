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
from torch.utils.data.dataset import TensorDataset, StackDataset
from torchmetrics.aggregation import SumMetric
from torchmetrics.image.fid import FrechetInceptionDistance

from datasets.mnist.datamodules import MNISTDataModule
from datasets.mnist.io import load_idx
from ddpm.training.callbacks import TrajectoryCallback
from ddpm.models.two_dimensions.unet import Unet as Unet2d
from ddpm.diffusion.diffusion import Diffusion
from ddpm.utils import get_generator


class DiffusionLightningModule(L.LightningModule):

    def __init__(
        self,
        dim: int = 20,
        dim_mults: List[int] = [1, 2, 4, 8],
        resnet_block_groups: int = 4,
        sinusoidal_pos_emb_theta: int = 10000,
        context_dim: int = 2,
        train_timesteps: int = 1000,
        sample_timesteps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: Literal["linear", "scaled_linear", "cosine"] = "linear",
        rescale_betas_zero_snr: bool = False,
        uncond_prob: float = 0.25,
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
        # TODO: Setup null token in setup function depending on shape of condition
        self.null_token = torch.randn(
            (self.hparams.context_dim,),
            generator=get_generator(self.hparams.seed, "cpu"),
        )
        self.fid = FrechetInceptionDistance(
            feature=self.hparams.fid_features,
            normalize=True,
            compute_on_cpu=True,
        )

    def _get_diffusion(self) -> Diffusion:
        return Diffusion(
            self.unet,
            self.hparams.train_timesteps,
            self.hparams.sample_timesteps,
            self.hparams.beta_start,
            self.hparams.beta_end,
            self.hparams.beta_schedule,
            self.hparams.rescale_betas_zero_snr,
            self.device,
        )
   
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def _fix_dimensions(self, batch):
        data = batch["image"]
        batch_size = data.shape[0]
        conditions = batch["label"].float().reshape(batch_size, self.hparams.context_dim, 1, 1)
        null_token = self.null_token \
            .reshape(self.hparams.context_dim, 1, 1) \
            .repeat((batch_size, 1, 1, 1)) \
            .to(self.device)
        assert len(data.shape) == 4 and \
            len(conditions.shape) == 4 and \
            len(null_token.shape) == 4 and \
            data.shape[1] == self.hparams.channels and \
            conditions.shape[1] == self.hparams.context_dim and \
            null_token.shape[1] == self.hparams.context_dim
        return data, conditions, null_token, batch_size

    def training_step(self, batch, batch_idx):
        data, conditions, null_token, batch_size = self._fix_dimensions(batch)

        # Training model
        timesteps = torch.randint(0, self.hparams.train_timesteps, (batch_size,)).to(self.device).long()
        noise = torch.randn_like(data).to(self.device)
        # TODO: Move null_token reshaping into init
        if self.hparams.uncond_prob == 1.:
            training_conditions = None
        elif torch.rand(1) < self.hparams.uncond_prob:
            training_conditions = null_token  # self.null_token.repeat((batch_size, 1, 1, 1)).to(self.device)
        else:
            training_conditions = conditions

        diffusion = self._get_diffusion()
        noisy_data = diffusion.add_noise(data, noise, timesteps)
        noise_pred = diffusion.noise_pred(noisy_data, timesteps, training_conditions)

        loss = F.mse_loss(noise, noise_pred)
        return loss

    def on_validation_start(self):
        self.fid = self.fid.to(self.device)
        self.val_generator = get_generator(self.hparams.seed, self.device)
        self.mse_sum = SumMetric().to(self.device)
        self.counter = SumMetric().to(self.device)

    def validation_step(self, batch, batch_idx):
        diffusion = self._get_diffusion()

        data, conditions, null_token, batch_size = self._fix_dimensions(batch)
        self.fid.update((data.detach() * 0.3801 + 0.1306).repeat(1, 3, 1, 1), real=True)

        if self.hparams.uncond_prob == 1.:
            include_conditions = None
            null_token = None
        else:
            include_conditions = conditions
        noise = torch.randn(data.shape, generator=self.val_generator, device=self.device)

        # Guided Random samples
        samples = diffusion.sample(
            noise,
            null_token,
            include_conditions,
            guidance_scale=7.5,
            deterministic=False,
            timesteps="sample",
            generator=self.val_generator,
            disable_progress_bar=True,
        )
        self.fid.update((samples.detach() * 0.3801 + 0.1306).repeat(1, 3, 1, 1), real=False)

        if batch_idx == 0:
            n_samples = min(batch_size, 50)
            n_cols = 10
            buf = io.BytesIO()
            fig, axes = plt.subplots(
                nrows=n_samples // n_cols,
                ncols=n_cols,
                gridspec_kw={"wspace": 0.5},
                figsize=(10, 10),
            )
            fig.suptitle("Guided Random Samples")
            for i in range(n_samples): 
                title = "Digit"
                cond_text = "N/A" if include_conditions is None else torch.argmax(conditions[i]).item()
                ax = axes.flatten()[i]
                ax.set_title(f"{title}: {cond_text}")
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.imshow(np.moveaxis(samples[i].detach().cpu().numpy(), 0, 2) * 0.3801 + 0.1306)
            fig.savefig(buf, format="png", bbox_inches="tight")
            samples_img = np.array(Image.open(buf).convert("RGB")).astype(np.uint8)
            self.logger.experiment.add_image("Guided Random Samples", samples_img, self.global_step, dataformats="HWC")

        # Reconstructions
        inverted_latents = diffusion.ddim_inversion(
            data,
            include_conditions,
            timesteps="sample",
            disable_progress_bar=True,
        )
        recons = diffusion.sample(
            inverted_latents,
            null_token,
            include_conditions,
            guidance_scale=1.0,
            deterministic=False,
            timesteps="sample",
            generator=self.val_generator,
            disable_progress_bar=True,
        )

        # TODO: Include FID for reconstructions
        mse_sum = F.mse_loss(data * 0.3801 + 0.1306, recons * 0.3801 + 0.1306, reduce="sum")
        self.mse_sum.update(mse_sum)
        self.counter.update(batch_size)

        if batch_idx == 0:
            n_samples = 20
            fig, axes = plt.subplots(nrows=n_samples, ncols=3, figsize=(5, 20))
            fig.suptitle("DDIM Inversion -> CFG Recon.")
            axes[0, 0].set_title("Orig.")
            axes[0, 1].set_title("Recon.")
            axes[0, 2].set_title("Diff.")
            for i in range(n_samples):
                data_sample = np.moveaxis(data[i].detach().cpu().numpy(), 0, 2) * 0.3801 + 0.1306
                recon_sample = np.moveaxis(recons[i].detach().cpu().numpy(), 0, 2) * 0.3801 + 0.1306
                diff = np.abs(data_sample - recon_sample).mean(axis=2)
                axes[i, 0].imshow(data_sample)
                axes[i, 1].imshow(recon_sample)
                axes[i, 2].imshow(diff, cmap="Reds", interpolation="nearest")
                for ax in axes[i]:
                    ax.set_yticklabels([])
                    ax.set_xticklabels([])
                    ax.set_yticks([])
                    ax.set_xticks([])
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
    parser.add_argument("--logdir", type=str, default="/vol/biomedic3/rrr2417/cxr-generation/experiments_mnist_grey")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--uncond-prob", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--train-timesteps", type=int, default=200)
    parser.add_argument("--dim-mults", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--no-tuner", action="store_true")
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
        context_dim = 10
    elif args.mnist_variant == "colour":
        dm = MNISTDataModule(
            data_dir="src/datasets/mnist/files/raw",
            seed=args.seed,
            split_ratio=(0.9, 0.1),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        channels = 3
        context_dim = 1
    else:
        raise ValueError(f"mnist_variant must be {'grey', 'colour'}, not {args.mnist_variant}")

    model = DiffusionLightningModule(
        dim=args.dim,
        dim_mults=args.dim_mults,
        train_timesteps=args.train_timesteps,
        sample_timesteps=50,
        beta_schedule="cosine",
        uncond_prob=args.uncond_prob,
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
        max_epochs=1000,
        check_val_every_n_epoch=25,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=-1,
    )

    # Tuning batch size is unsupported with distributed strategies in lightning 2.2.4
    if args.device_id != -1 and not args.no_tuner:
        tuner = Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=dm, mode="power")

    trainer.fit(model, datamodule=dm)


def editing():
    from ddpm.utils import get_device
    from ddpm.editing.null_text_inversion import NullTokenOptimisation
    from datasets.mnist.datamodules import MNISTDataModule

    seed_everything(0)
    v = 72
    ckpt_path = f"/vol/biomedic3/rrr2417/cxr-generation/experiments_mnist_grey/lightning_logs/version_{v}/checkpoints/last.ckpt"
    model = DiffusionLightningModule.load_from_checkpoint(ckpt_path)

    bs = 10
    dm = MNISTDataModule(
        data_dir="src/datasets/mnist/files/raw",
        seed=0,
        split_ratio=(0.9, 0.1),
        batch_size=bs,
    ) 
    dm.setup("test")
    for data in dm.test_dataloader():
        data, conditions, null_token, batch_size = model._fix_dimensions(data)
        data = data.to(model.device)
        conditions = None if model.uncond_prob == 1.0 else conditions.to(model.device)
        null_token = null_token.to(model.device)
        break
    device = get_device()

    diffusion = model._get_diffusion()

    # Recon
    inverted_latents = diffusion.ddim_inversion(
        data,
        conditions,
        timesteps="sample",
        disable_progress_bar=True,
    )
    samples = diffusion.sample(
        inverted_latents,
        null_token,
        conditions,
        guidance_scale=1.,
        deterministic=True,
        timesteps="sample",
        generator=get_generator(model.hparams.seed, device),
        disable_progress_bar=True,
    )
    fig, axes = plt.subplots(nrows=bs, ncols=2, figsize=(5, 25))
    axes[0, 0].set_title("Original")
    axes[0, 1].set_title("Recon")
    for i in range(bs):
        axes[i, 0].imshow(np.moveaxis(data[i].cpu().numpy(), 0, 2) * 0.3801 + 0.1306)
        axes[i, 1].imshow(np.moveaxis(samples[i].cpu().numpy(), 0, 2) * 0.3801 + 0.1306)
    fig.savefig(f"/vol/biomedic3/rrr2417/cxr-generation/experiments_mnist_grey/lightning_logs/version_{v}/recon.pdf", bbox_inches="tight")

    if model.uncond_prob == 1.0:
        exit()

    # Editing
    # TODO: Include this case as unittest - should be equal to reconstruction
    # nto = NullTokenOptimisation(diffusion, null_token, 0, 1e-4, 1.0)
    nto = NullTokenOptimisation(diffusion, null_token, 10, 1e-4, 7.5)
    nto.fit(data, conditions, disable_progress_bar=False)
    recon = nto.generate(generator=get_generator(model.hparams.seed, device), disable_progress_bar=True).detach().cpu().numpy()

    fig, ax = plt.subplots(
        nrows=bs,
        ncols=12,
        gridspec_kw={"hspace": 0.1, "wspace": 0.1},
    )

    def _edit_axis(ax): 
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])

    fs = 5
    ax[0, 0].set_title(f"Orig.", fontsize=fs)
    ax[0, 1].set_title(f"Recon.", fontsize=fs)
    for j in range(bs):
        ax[j, 0].imshow(np.moveaxis(data[j].cpu().numpy(), 0, 2) * 0.3801 + 0.1306)
        _edit_axis(ax[j, 0])
        ax[j, 1].imshow(np.moveaxis(recon[j], 0, 2) * 0.3801 + 0.1306)
        _edit_axis(ax[j, 1])

    for i in range(10):
        edit_cond = F.one_hot(torch.tensor(i), 10) \
            .reshape(1, -1, 1, 1) \
            .repeat(batch_size, 1, 1, 1) \
            .to(model.device)
        edit = nto.generate(edit_cond, generator=get_generator(model.hparams.seed, device), corrections=False, swap_fraction=0.75)
        edit = edit.detach().cpu().numpy()
        for j in range(bs):
            if j == 0:
                ax[j, i + 2].set_title(f"Digit: {i}", fontsize=fs)
            ax[j, i + 2].imshow(np.moveaxis(edit[j], 0, 2) * 0.3801 + 0.1306)
            _edit_axis(ax[j, i + 2])

    fig.savefig(f"/vol/biomedic3/rrr2417/cxr-generation/experiments_mnist_grey/lightning_logs/version_{v}/edits.pdf", bbox_inches="tight")
