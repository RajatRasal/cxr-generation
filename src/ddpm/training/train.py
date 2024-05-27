import logging
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
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import TensorDataset, StackDataset
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.utilities.plot import plot_confusion_matrix

from ddpm.training.plots import trajectory_plot_1d, visualise_gmm, trajectory_plot_1d_with_inverse
from ddpm.training.callbacks import TrajectoryCallback
from ddpm.models.unet import Unet as Unet1d
from ddpm.diffusion.diffusion import Diffusion
from ddpm.datasets.bgmm import GMM, gmm_stddev_prior, gmm_means_prior, gmm_cluster_prior, gmm_mixture_probs_prior
from ddpm.utils import get_generator


class DiffusionLightningModule(L.LightningModule):

    def __init__(
        self,
        batch_size: int = 2048,
        dim: int = 20,
        dim_mults: List[int] = [1, 2, 4, 8],
        channels: int = 1,
        resnet_block_groups: int = 4,
        sinusoidal_pos_emb_theta: int = 10000,
        context_dim: int = 2,
        train_timesteps: int = 1000,
        sample_timesteps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: Literal["linear", "scaled_linear", "cosine"] = "linear",
        uncond_prob: float = 0.25,
        dataset_seed: int = 0,
        min_clusters: int = 2,
        max_clusters: int = 5,
        dataset_size: int = 1000,
        gmm_categories_concentration: float = 0.01,
        cluster_separation: float = 4.0,
        center_box: float = 5.,
        sanity_check: bool = False,
        folder: str = "/tmp",
    ):
        super().__init__()
        assert center_box > 0.

        self.save_hyperparameters()

        self.unet = Unet1d(
            self.hparams.dim,
            self.hparams.dim_mults,
            self.hparams.channels,
            self.hparams.resnet_block_groups,
            self.hparams.sinusoidal_pos_emb_theta,
            self.hparams.context_dim,
        )
        self.center_box = (-self.hparams.center_box, self.hparams.center_box)
        self.null_token = torch.randn((2, 1), generator=get_generator(0, "cpu"))

    def _get_diffusion(self) -> Diffusion:
        return Diffusion(
            self.unet,
            self.hparams.train_timesteps,
            self.hparams.sample_timesteps,
            self.hparams.beta_start,
            self.hparams.beta_end,
            self.hparams.beta_schedule,
            self.device,
        )

    def setup(self, stage):
        """
        Assign train/val split(s) for use in Dataloaders. This method is called from every process
        across all nodes, hence why state is set here.

        https://lightning.ai/docs/pytorch/stable/data/datamodule.html#setup
        """
        logger = logging.getLogger("lightning.pytorch")
        logger.info("Creating dataset")

        generator = torch.manual_seed(self.hparams.dataset_seed)

        # Sampling from GMM priors
        if self.hparams.sanity_check:
            n_clusters = 3
        else:
            n_clusters = gmm_cluster_prior(
                min_clusters=self.hparams.min_clusters,
                max_clusters=self.hparams.max_clusters,
            )

        if self.hparams.sanity_check:
            means = [torch.tensor([[-3., 3.]]), torch.tensor([[3., 3.]]), torch.tensor([[0., -3.]])]
            self.means = torch.cat(means)
        else:
            # each cluster mean should be separated by cluster_separation from
            # every other cluster mean
            retries = 1000
            for retry in range(retries):
                self.means = gmm_means_prior(
                    n_clusters=n_clusters,
                    dimensions=2,
                    center_box=self.center_box,
                )
                _means = torch.cat([x.unsqueeze(0) for x in self.means])
                pairwise = torch.cdist(_means, _means) > self.hparams.cluster_separation
                # diagonals will always be less than cluster_separation
                if pairwise.sum() == n_clusters ** 2 - n_clusters:
                    break
                if retry == retries - 1:
                    raise ValueError("Try a different seed for the training run")

        dims = self.means[0].shape[0]
        if self.hparams.sanity_check:
            covs = torch.cat([torch.eye(dims).unsqueeze(0) * 0.5 for _ in range(n_clusters)])
        else:
            stddevs = gmm_stddev_prior(
                n_clusters=n_clusters,
                stddev_concentration=2.,
                stddev_rate=2.,
            )
            covs = torch.cat([(torch.eye(dims) * stddev).unsqueeze(0) for stddev in stddevs])

        if self.hparams.sanity_check:
            mixture_probs = torch.tensor([1 / n_clusters for _ in range(n_clusters)])
        else:
            mixture_probs = gmm_mixture_probs_prior(
                self.hparams.gmm_categories_concentration,
                n_clusters,
            )

        logger.debug(f"no. of clusters: {n_clusters}")
        logger.debug(f"means: {self.means}")
        logger.debug(f"covs:\n{covs}")
        logger.debug(f"mixture probs: {mixture_probs}")

        # Sample the GMM to generate dataset
        self.gmm = GMM(
            self.means,
            covs,
            mixture_probs,
        )
        samples = self.gmm.samples(self.hparams.dataset_size)
        data = samples.samples
        mixtures = samples.mixtures

        # Plot GMM to visualise
        visualise_gmm(
            data,
            self.center_box,
            self.means,
            os.path.join(self.hparams.folder, "data"),
            "pdf",
        )

        # Scaling data for VP-SDE
        # TODO: Implement MinMaxScaler in pytorch
        self.scaler = MinMaxScaler((-1, 1))
        data = torch.from_numpy(self.scaler.fit_transform(data)).unsqueeze(1).float()

        # Conditioning the DDPM on the GMM cluster means = mode
        conditions = torch.cat([
            self.gmm.get_mixture_parameters(mixture).mean.unsqueeze(0)
            for mixture in mixtures
        ]).unsqueeze(2)

        # Create dataset to be used in prepare_data
        dataset = StackDataset(data=TensorDataset(data), conditions=TensorDataset(conditions))
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [0.8, 0.1, 0.1], generator)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=2048,
            shuffle=True,
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

    def _calculate_nll(
        self,
        diffusion: Diffusion,
        noise: torch.FloatTensor,
        conditions: Optional[torch.FloatTensor],
        guidance_scale: float,
        generator: torch.Generator,
        deterministic: bool,
    ) -> torch.FloatTensor:
        recon = diffusion.sample(
            noise,
            self.null_token.unsqueeze(0).repeat((conditions.shape[0], 1, 1)).to(self.device) if conditions is not None else None,
            conditions,
            guidance_scale=guidance_scale,
            deterministic=deterministic,
            timesteps="sample",
            generator=generator,
            disable_progress_bar=True,
        )
        recon = recon.detach().cpu().squeeze(1).numpy()
        recon = self.scaler.inverse_transform(recon)
        recon = torch.from_numpy(recon).unsqueeze(1)
        recon_nll = -self.gmm.log_likelihood(recon)
        recon_nll = recon_nll.mean()
        return recon_nll

    def _calculate_reconstruction(
        self,
        diffusion: Diffusion,
        data: torch.FloatTensor,
        noise: torch.FloatTensor,
        conditions: Optional[torch.FloatTensor],
        guidance_scale: float,
        generator: torch.Generator,
        deterministic: bool,
    ) -> torch.FloatTensor:
        recon = diffusion.sample(
            noise,
            self.null_token.unsqueeze(0).repeat((conditions.shape[0], 1, 1)).to(self.device) if conditions is not None else None,
            conditions,
            guidance_scale=guidance_scale,
            deterministic=deterministic,
            timesteps="sample",
            generator=generator,
            disable_progress_bar=True,
        )
        recon = recon.detach().cpu().squeeze(1).numpy()
        recon = self.scaler.inverse_transform(recon)
        recon = torch.from_numpy(recon).unsqueeze(1).to(self.device)
        mse = F.mse_loss(data, recon)
        return mse

    def _calculate_metrics_guidance(
        self,
        data: torch.FloatTensor,
        conditions: torch.FloatTensor,
        generator: torch.Generator,
        inversion: bool,
        prefix: str,
    ) -> Dict[str, torch.FloatTensor]:
        # TODO: Include KL and Wasserstein distance
        diffusion = self._get_diffusion()
        # DDIM inversion
        if inversion:
            noise = diffusion.ddim_inversion(data, conditions, timesteps="sample", disable_progress_bar=True)
        else:
            noise = data
        # Reconstruction
        recon_nll_ddpm = self._calculate_nll(diffusion, noise, conditions, 1.0, generator, False)
        recon_mse_ddpm = self._calculate_reconstruction(diffusion, data, noise, conditions, 1.0, generator, False)
        recon_nll_ddim = self._calculate_nll(diffusion, noise, conditions, 1.0, generator, True)
        recon_mse_ddim = self._calculate_reconstruction(diffusion, data, noise, conditions, 1.0, generator, True)
        # Guided sampling with DDPM
        guided_nll_45_ddpm = self._calculate_nll(diffusion, noise, conditions, 4.5, generator, False)
        guided_nll_75_ddpm = self._calculate_nll(diffusion, noise, conditions, 7.5, generator, False)
        # Guided sampling with DDIM
        guided_nll_45_ddim = self._calculate_nll(diffusion, noise, conditions, 4.5, generator, True)
        guided_nll_75_ddim = self._calculate_nll(diffusion, noise, conditions, 7.5, generator, True)
        # Metrics dict
        return {
            f"{prefix} / DDIM Sampling / NLL / w = 1.0 (recon)": recon_nll_ddim,
            f"{prefix} / DDIM Sampling / MSE / w = 1.0 (recon)": recon_mse_ddim,
            f"{prefix} / DDIM Sampling / NLL / w = 4.5": guided_nll_45_ddim,
            f"{prefix} / DDIM Sampling / NLL / w = 7.5": guided_nll_75_ddim,
            f"{prefix} / DDPM Sampling / NLL / w = 1.0 (recon)": recon_nll_ddpm,
            f"{prefix} / DDPM Sampling / MSE / w = 1.0 (recon)": recon_mse_ddpm,
            f"{prefix} / DDPM Sampling / NLL / w = 4.5": guided_nll_45_ddpm,
            f"{prefix} / DDPM Sampling / NLL / w = 7.5": guided_nll_75_ddpm,
        }
    
    def training_step(self, batch, batch_idx):
        data = batch["data"][0]
        conditions = batch["conditions"][0]
        batch_size = data.shape[0]

        # Training model
        timesteps = torch.randint(0, self.hparams.train_timesteps, (batch_size,)).to(self.device).long()
        noise = torch.randn_like(data).to(self.device)
        # training_conditions = None if torch.rand(1) < self.hparams.uncond_prob else conditions
        training_conditions = self.null_token.unsqueeze(0).repeat((conditions.shape[0], 1, 1)).to(self.device) if torch.rand(1) < self.hparams.uncond_prob else conditions

        diffusion = self._get_diffusion()
        noisy_data = diffusion.add_noise(data, noise, timesteps)
        noise_pred = diffusion.noise_pred(noisy_data, timesteps, training_conditions)

        loss = F.l1_loss(noise, noise_pred)

        # Training metrics
        # TODO: log at same intervals as validation
        if self.current_epoch % 100 == 0:
            generator = get_generator(self.hparams.dataset_seed, self.device)
            unguided_metrics = self._calculate_metrics_guidance(noise, None, generator, inversion=False, prefix="Unconditional")
            guidance_metrics = self._calculate_metrics_guidance(noise, conditions, generator, inversion=False, prefix="Guidance")
            inversion_guidance_metrics = self._calculate_metrics_guidance(data, conditions, generator, inversion=True, prefix="DDIM Inversion + Guidance")
            metrics = {**unguided_metrics, **guidance_metrics, **inversion_guidance_metrics}
            metrics = {f"Train / {name}": metric for name, metric in metrics.items()}
            self.log_dict(metrics, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        data = batch["data"][0]
        conditions = batch["conditions"][0]

        generator = get_generator(self.hparams.dataset_seed, self.device)
        noise = torch.randn(data.shape, generator=generator, device=self.device)
        unguided_metrics = self._calculate_metrics_guidance(noise, None, generator, inversion=False, prefix="Unconditional")
        guidance_metrics = self._calculate_metrics_guidance(noise, conditions, generator, inversion=False, prefix="Guidance")
        inversion_guidance_metrics = self._calculate_metrics_guidance(data, conditions, generator, inversion=True, prefix="DDIM Inversion + Guidance")
        metrics = {**unguided_metrics, **guidance_metrics, **inversion_guidance_metrics}
        metrics = {f"Val / {name}": metric for name, metric in metrics.items()}
        self.log_dict(metrics, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        return

    def on_test_epoch_end(self):
        data = torch.cat([x["data"][0] for x in self.test_dataloader()]).to(self.device)
        conditions = torch.cat([x["conditions"][0] for x in self.test_dataloader()]).to(self.device)
        conditions_idx = []
        for c in conditions:
            for i, mean in enumerate(self.gmm.means):
                if (c.cpu().flatten() == mean).all():
                    conditions_idx.append(i)
                    break
        conditions_idx = torch.tensor(conditions_idx)

        # TODO: kde plots
        diffusion = self._get_diffusion()

        # Random noise - different to noise from val step
        generator = get_generator(0, self.device)
        n_samples = data.shape[0]
        random_noise = torch.randn(data.shape, generator=generator, device=self.device)

        # DDIM Inversion
        ddim_inversion_callback = TrajectoryCallback()
        ddim_inversion = diffusion.ddim_inversion(data, conditions, timesteps="sample", callbacks=[ddim_inversion_callback])
        ddim_guided_x = ddim_inversion_callback.sample(n_samples, dim=0)
        ddim_guided_y = ddim_inversion_callback.sample(n_samples, dim=1)

        def _trajectories_plot(deterministic, plot_name, conditions, guidance_scale = 0.0, inverse = False):
            sample_callback = TrajectoryCallback()
            samples = diffusion.sample(
                ddim_inversion if inverse else random_noise,
                self.null_token.unsqueeze(0).repeat((conditions.shape[0], 1, 1)).to(self.device) if conditions is not None else None,
                conditions=conditions,
                deterministic=deterministic,
                timesteps="sample",
                guidance_scale=guidance_scale,
                generator=generator,
                callbacks=[sample_callback],
                disable_progress_bar=True,
            )
            predictions = self.gmm.predict(samples.cpu())

            sample_x = sample_callback.sample(n_samples, dim=0)
            sample_y = sample_callback.sample(n_samples, dim=1)

            if inverse:
                plotter_x = partial(
                    trajectory_plot_1d_with_inverse,
                    trajectories=[sample.numpy().flatten() for sample in sample_x],
                    inverse_trajectories=[sample.numpy().flatten() for sample in ddim_guided_x],
                    save_path=os.path.join(self.hparams.folder, f"{plot_name}_x"),
                    true_data=data[:, 0, 0].cpu().numpy(),
                )
                plotter_y = partial(
                    trajectory_plot_1d_with_inverse,
                    trajectories=[sample.numpy().flatten() for sample in sample_y],
                    inverse_trajectories=[sample.numpy().flatten() for sample in ddim_guided_y],
                    save_path=os.path.join(self.hparams.folder, f"{plot_name}_y"),
                    true_data=data[:, 0, 1].cpu().numpy(),
                )
                # TODO: KDE plot 
            else:
                plotter_x = partial(
                    trajectory_plot_1d,
                    trajectories=[sample.numpy() for sample in sample_x],
                    save_path=os.path.join(self.hparams.folder, f"{plot_name}_x"),
                    true_data=data[:, 0, 0].cpu().numpy(),
                )
                plotter_y = partial(
                    trajectory_plot_1d,
                    trajectories=[sample.numpy() for sample in sample_y],
                    save_path=os.path.join(self.hparams.folder, f"{plot_name}_y"),
                    true_data=data[:, 0, 1].cpu().numpy(),
                )
                # TODO: KDE plot 

            kwargs = dict(
                T=self.hparams.sample_timesteps,
                y_lims=(-2, 2),
                kde_bandwidth=0.1,
                output_type="pdf",
                fast=True,
            )
            plotter_x(**kwargs)
            plotter_y(**kwargs)

            return samples, predictions

        # Guidance scales for tests 
        guidance_scales = [1.0, 4.5, 7.5]

        def _sampling(plot_name_prefix, axes, deterministic, inverse):
            for j, w in enumerate(guidance_scales):
                if inverse:
                    acc_matrix = torch.empty((self.means.shape[0], self.means.shape[0]))
                else:
                    classification_metric = MulticlassConfusionMatrix(
                        num_classes=len(self.means),
                        normalize="true",
                    )
                all_targets = []
                all_preds = []
                for i, mean in enumerate(self.means):
                    _mean = mean.unsqueeze(0).repeat((n_samples, 1)).unsqueeze(-1).to(self.device)
                    samples, preds = _trajectories_plot(deterministic, f"{plot_name_prefix}_{i}_{w}", _mean, w, inverse)
                    if inverse:
                        # TODO: Use plot_confusion_matrix from torchmetric
                        for k, mean in enumerate(self.means):
                            original_class_mask = conditions_idx == k
                            _preds = preds[original_class_mask]
                            acc = (_preds == i).sum() / _preds.shape[0]
                            acc_matrix[k, i] = acc
                    else:
                        classification_metric.update(
                            torch.tensor([i for _ in range(preds.shape[0])]),
                            torch.tensor(preds.flatten().tolist()),
                        )
                if inverse:
                    plot_confusion_matrix(acc_matrix, axes[j])
                else:
                    classification_metric.plot(ax=axes[j])

        def _confusion_matrix_subplots():
            return plt.subplots(
                nrows=2,
                ncols=len(guidance_scales),
                gridspec_kw={
                    "wspace": 0.05, "width_ratios": [1 for _ in range(len(guidance_scales))],
                    "hspace": 0.01, "height_ratios": [1, 1],
                },
            )

        def _confusion_matrix_axes_formatting(axes):
            for i, w in enumerate(guidance_scales):
                axes[0, i].set_title(rf"$\omega = {w}$")
            for ax in axes.flatten():
                ax.set_xlabel("")
                ax.set_ylabel("")
            for ax in axes[0, :].flatten():
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.set_xticks([])
            for ax in axes[:, 1:].flatten():
                ax.yaxis.set_tick_params(labelleft=False)
                ax.set_yticks([])
            axes[0, 0].set_ylabel("DDIM")
            axes[1, 0].set_ylabel("DDPM")

        # Confusion matrices for guided generations from random noise (below)
        fig, axes = _confusion_matrix_subplots() 
        _sampling("guided_sample_ddpm", axes[0, :], False, False)
        _sampling("guided_sample_ddim", axes[1, :], True, False)
        _confusion_matrix_axes_formatting(axes)
        fig.savefig(
            os.path.join(self.hparams.folder, "guided_confusion_matrix.pdf"),
            bbox_inches="tight",
        )

        # Confusion matrices for DDIM inversion + guided generations (below)
        fig_edit, axes_edit = _confusion_matrix_subplots() 
        _sampling("guided_recon_ddpm", axes_edit[0, :], False, True)
        _sampling("guided_recon_ddim", axes_edit[1, :], True, True)
        _confusion_matrix_axes_formatting(axes_edit)
        fig_edit.savefig(
            os.path.join(self.hparams.folder, "edit_confusion_matrix.pdf"),
            bbox_inches="tight",
        )

        # TODO: KDE plots for edits and cfg guidance
        # TODO: Interpolation - pick a few ddim_inversion and random samples

    def configure_optimizers(self):
        return torch.optim.Adam(self.unet.parameters(), lr=1e-5)


def main():
    folder = "experiment_samples"

    for seed in range(0, 30):
        seed_everything(seed)

        subfolder = f"{folder}/{seed}"
        os.makedirs(subfolder, exist_ok=True)

        diffusion = DiffusionLightningModule(
            dim=20,
            dim_mults=[1],
            channels=1,
            train_timesteps=1000,
            sample_timesteps=50,
            dataset_size=10000,
            beta_schedule="cosine",
            min_clusters=2,
            max_clusters=7,
            gmm_categories_concentration=5,
            center_box=10.,
            cluster_separation=4.,
            folder=subfolder,
            sanity_check=False,
        )

        tb_logger = TensorBoardLogger(save_dir=folder, version=seed)
        checkpoint_callback = ModelCheckpoint(
            dirpath=subfolder,
            filename=f"model_{seed}",
        )
        trainer = Trainer(
            accelerator="gpu",
            max_epochs=1000,
            check_val_every_n_epoch=100,
            logger=tb_logger,
            callbacks=[checkpoint_callback],
        )

        trainer.fit(diffusion)
        trainer.test(diffusion)
