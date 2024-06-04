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
from lightning.pytorch.tuner.tuning import Tuner
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import TensorDataset, StackDataset
from torchmetrics.aggregation import SumMetric
from torchmetrics.image.fid import FrechetInceptionDistance

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
        batch_size: int = 128,
        learning_rate: float = 1e-4,
        seed: int = 0,
        fid_features: Literal[64, 192, 768, 2048] = 192,
        label_condition: bool = True,
        split_ratio: Tuple[int, int] = (0.9, 0.1),
    ):
        super().__init__()
        self.save_hyperparameters()

        self.unet = Unet2d(
            self.hparams.dim,
            self.hparams.dim_mults,
            3, # self.hparams.channels,
            self.hparams.resnet_block_groups,
            self.hparams.sinusoidal_pos_emb_theta,
            self.hparams.context_dim,
        )
        # TODO: Setup null token in setup function depending on shape of condition
        self.null_token = torch.randn((1,), generator=get_generator(self.hparams.seed, "cpu"))
        self.fid = FrechetInceptionDistance(feature=self.hparams.fid_features, compute_on_cpu=True)

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

    def _normalise(self, images: np.ndarray) -> np.ndarray:
        return (images - self.means[None, :, None, None]) / self.stddevs[None, :, None, None]
    
    def _unnormalise(self, images: np.ndarray) -> np.ndarray:
        return (images * self.stddevs[None, :, None, None]) + self.means[None, :, None, None] 

    def setup(self, stage):
        """
        Assign train/val split(s) for use in Dataloaders. This method is called from every process
        across all nodes, hence why state is set here.

        https://lightning.ai/docs/pytorch/stable/data/datamodule.html#setup
        """
        # TODO: Pass this in
        folder = "src/datasets/mnist/files/colour_with_fixed_saturation/0_1.0"

        def _preprocess_metadata(arr):
            return np.expand_dims(arr, axis=(1, 2, 3)).astype(np.float32)

        # Load train dataset
        images = load_idx(os.path.join(folder, "train-images-idx3-ubyte.gz"))
        self.means = images.mean(axis=(0, 2, 3))
        self.stddevs = images.std(axis=(0, 2, 3))
        images = self._normalise(images)
        hues = np.loadtxt(os.path.join(folder, "train-hues.txt"))
        hues = _preprocess_metadata(hues)
        labels = load_idx(os.path.join(folder, "train-labels-idx1-ubyte.gz"))
        labels = _preprocess_metadata(labels)
        dataset = StackDataset(
            images=TensorDataset(torch.tensor(images).float()),
            labels=TensorDataset(torch.tensor(labels)),
            hues=TensorDataset(torch.tensor(hues)),
        )
        self.train_dataset, self.val_dataset = random_split(
            dataset,
            self.hparams.split_ratio,
            torch.manual_seed(self.hparams.seed),
        )

        # Load test dataset 
        images = load_idx(os.path.join(folder, "t10k-images-idx3-ubyte.gz"))
        images = self._normalise(images)
        hues = np.loadtxt(os.path.join(folder, "t10k-hues.txt"))
        hues = _preprocess_metadata(hues)
        labels = load_idx(os.path.join(folder, "t10k-labels-idx1-ubyte.gz"))
        labels = _preprocess_metadata(labels)
        self.test_dataset = StackDataset(
            images=TensorDataset(torch.tensor(images)),
            labels=TensorDataset(torch.tensor(labels)),
            hues=TensorDataset(torch.tensor(hues)),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=10,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
        )
   
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx):
        data = batch["images"][0]
        conditions = batch["labels"][0] if self.hparams.label_condition else batch["hues"][0]
        batch_size = data.shape[0]

        # Training model
        timesteps = torch.randint(0, self.hparams.train_timesteps, (batch_size,)).to(self.device).long()
        noise = torch.randn_like(data).to(self.device)
        # TODO: Move null_token reshaping into init
        if torch.rand(1) < self.hparams.uncond_prob:
            training_conditions = self.null_token.repeat((data.shape[0], 1, 1, 1)).to(self.device)
        else:
            training_conditions = conditions

        diffusion = self._get_diffusion()
        noisy_data = diffusion.add_noise(data, noise, timesteps)
        noise_pred = diffusion.noise_pred(noisy_data, timesteps, training_conditions)

        loss = F.mse_loss(noise, noise_pred)
        return loss

    def on_validation_start(self):
        self.fid = self.fid.to(self.device)
        for x in self.train_dataloader():
            self.fid.update(
                torch.tensor(self._unnormalise(x["images"][0].detach().cpu().numpy())).to(self.device, dtype=torch.uint8),
                real=True,
            )
        self.val_generator = get_generator(self.hparams.seed, self.device)
        self.mse_sum = SumMetric().to(self.device)

    def validation_step(self, batch, batch_idx):
        diffusion = self._get_diffusion()

        data = batch["images"][0]
        conditions = batch["labels"][0] if self.hparams.label_condition else batch["hues"][0]
        include_conditions = conditions if self.hparams.uncond_prob < 1. else None
        noise = torch.randn(data.shape, generator=self.val_generator, device=self.device)
        null_token = self.null_token.repeat((data.shape[0], 1, 1, 1)).to(self.device)

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
        samples = self._unnormalise(samples.detach().cpu().numpy())
        self.fid.update(
            torch.tensor(samples, device=self.device, dtype=torch.uint8),
            real=False,
        )

        if batch_idx == 0:
            n_samples = 50
            n_cols = 10
            buf = io.BytesIO()
            fig, axes = plt.subplots(nrows=n_samples // n_cols, ncols=n_cols, gridspec_kw={"wspace": 0.5})
            fig.suptitle("Guided Random Samples")
            for i in range(n_samples): 
                title = "Digit" if self.hparams.label_condition else "Hue"
                cond_text = "N/A" if include_conditions is None else round(conditions[i].item(), 2)
                ax = axes.flatten()[i]
                ax.set_title(f"{title}: {cond_text}")
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.imshow((np.moveaxis(samples[i], 0, 2) * 255).astype(np.uint8))
            fig.savefig(buf, format="png", bbox_inches="tight")
            samples_img = np.array(Image.open(buf).convert("RGB")).astype(np.uint8)
            self.logger.experiment.add_image("Guided Random Samples", samples_img, self.global_step, dataformats="HWC")

        # Reconstructions
        # TODO: Show the path for inversion followed by reconstruction
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
        recons = self._unnormalise(recons.detach().cpu().numpy())

        # TODO: Include FID for reconstructions
        data = self._unnormalise(data.detach().cpu().numpy())
        mse_sum = F.mse_loss(
            torch.tensor(data, device=self.device, dtype=torch.float32),
            torch.tensor(recons, device=self.device, dtype=torch.float32),
            reduce="sum",
        )
        self.mse_sum.update(mse_sum)

        if batch_idx == 0:
            n_samples = 20
            fig, axes = plt.subplots(nrows=n_samples, ncols=3)
            fig.suptitle("DDIM Inversion -> CFG Recon.")
            axes[0, 0].set_title("Orig.")
            axes[0, 1].set_title("Recon.")
            axes[0, 2].set_title("Diff.")
            for i in range(n_samples):
                data_sample = (np.moveaxis(data[i], 0, 2) * 255).astype(np.uint8)
                recon_sample = (np.moveaxis(recons[i], 0, 2) * 255).astype(np.uint8)
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
        mse = self.mse_sum.compute() / len(self.val_dataset)
        log_dict = {"val/mse": mse.item(), "val/fid": fid_score.item()}
        self.log_dict(log_dict, on_step=False, on_epoch=True)
        self.fid.reset()
        return log_dict

    def configure_optimizers(self):
        return torch.optim.Adam(self.unet.parameters(), lr=self.hparams.learning_rate)


def main():
    # 5 settings: mnist, mnist with digit, colour mnist, colour mnist with digit, colour mnist with hue

    folder = "/vol/biomedic3/rrr2417/cxr-generation/experiments_mnist_3"

    seed_everything(0, workers=True)

    model = DiffusionLightningModule(
        dim=128,
        dim_mults=[1, 1, 2],
        train_timesteps=1000,
        sample_timesteps=50,
        beta_schedule="cosine",
        uncond_prob=.2,
    )
    tb_logger = TensorBoardLogger(save_dir=folder)
    checkpoint_callback = ModelCheckpoint(
        # TODO: Unable to track val/fid - fix this!
        # NOTE: Possibly fixed when switched on_validation_step_end to on_validation_epoch_end
        # monitor="val/fid",
        # mode="min",
        save_last=True,
    )
    trainer = Trainer(
        deterministic=True,
        max_epochs=1000,
        check_val_every_n_epoch=25,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=1,
    )
    # TODO: Figure out why batch_size is not included in the tensorboard
    # Relies on a batch_size hyperparameter in the LightningModule
    # which is passed into the dataloaders
    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, mode="power")

    trainer.fit(model)


# def editing():
#     from ddpm.utils import get_device
#     from ddpm.editing.null_text_inversion import NullTokenOptimisation
# 
#     seed_everything(0)
# 
#     ckpt_path = "/vol/biomedic3/rrr2417/cxr-generation/experiments_mnist_2/1/last.ckpt"
#     model = DiffusionLightningModule.load_from_checkpoint(ckpt_path)
#     model.setup("test")
#     test_dataloader = model.train_dataloader()
#     device = get_device()
#     gen = get_generator(model.hparams.seed, device)
# 
#     diffusion = model._get_diffusion()
#     noise = torch.randn((100, 3, 28, 28), generator=gen, device=device)
#     null_token = model.null_token.repeat((100, 1, 1, 1)).to(device)
#     samples = diffusion.sample(
#         noise,
#         null_token,
#         None,
#         guidance_scale=7.5,
#         deterministic=False,
#         timesteps="sample",
#         generator=gen,
#         disable_progress_bar=True,
#     )
#     fig, axes = plt.subplots(nrows=5, ncols=5)
#     for img, ax in zip(samples, axes.flatten()):
#         ax.imshow(np.moveaxis(img.cpu().numpy(), 0, 2))
#     fig.savefig("/vol/biomedic3/rrr2417/cxr-generation/experiments_mnist_2/random_samples.pdf")
# 
#     exit()
# 
#     nto = NullTokenOptimisation(diffusion, null_token, 10, 1e-3, 1.5)
# 
#     n = 5
#     fig, ax = plt.subplots(nrows=n, ncols=7, gridspec_kw={"hspace": 0.5, "wspace": 0.5})
#     ax[0, 0].set_title("Orig.")
#     ax[0, 1].set_title("Recon.")
#     ax[0, 2].set_title("Hue: 0")
#     ax[0, 3].set_title("Hue: 0.25")
#     ax[0, 4].set_title("Hue: 0.5")
#     ax[0, 5].set_title("Hue: 0.75")
#     ax[0, 6].set_title("Hue: 1.0")
#     for x in test_dataloader:
#         data = x["images"][0].to(device).float()[:n]
#         conditions = x["hues"][0].to(device).float().reshape(-1, 1, 1, 1)[:n]
#         nto.fit(data, conditions, disable_progress_bar=False)
#         recon = nto.generate().detach().cpu().numpy()
#         edit_0 = nto.generate(torch.ones_like(conditions) * 0, generator=gen, corrections=False).detach().cpu().numpy()
#         edit_1 = nto.generate(torch.ones_like(conditions) * 0.25, generator=gen, corrections=False).detach().cpu().numpy()
#         edit_2 = nto.generate(torch.ones_like(conditions) * 0.5, generator=gen, corrections=False).detach().cpu().numpy()
#         edit_3 = nto.generate(torch.ones_like(conditions) * 0.75, generator=gen, corrections=False).detach().cpu().numpy()
#         edit_4 = nto.generate(torch.ones_like(conditions), generator=gen, corrections=False).detach().cpu().numpy()
#         _data = data.detach().cpu().numpy()
#         for i in range(len(_data)):
#             # (orig, r, e0, e1, e2, e3, e4)
#             for _ax, img in zip(ax[i, :], [_data[i], recon[i], edit_0[i], edit_1[i], edit_2[i], edit_3[i], edit_4[i]]):
#                 _ax.imshow(np.moveaxis(img, 0, 2))
#                 _ax.set_xticks([])
#                 _ax.set_yticks([])
#                 _ax.set_yticklabels([])
#                 _ax.set_xticklabels([])
#         fig.savefig("mnist_edits.pdf", bbox_inches="tight")
#         break
