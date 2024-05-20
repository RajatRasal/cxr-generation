import os
import random
from typing import List, Literal, Tuple

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset, StackDataset

from ddpm.training.callbacks import TrajectoryCallback
from ddpm.training.plots import trajectory_plot_1d, kde_plot_2d_compare, kde_plot_2d
from ddpm.models.unet import Unet as Unet1d
from ddpm.diffusion.diffusion import Diffusion
from ddpm.datasets.bgmm import GMM, gmm_stddev_prior, gmm_means_prior, gmm_cluster_prior, gmm_mixture_probs_prior


class DiffusionLightningModule(L.LightningModule):

    def __init__(
        self,
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
        # TODO: Use prepare_data
        if stage != "fit":
            return

        print("Setting up datasets")
        # Sampling from Bayesian GMM
        if self.hparams.sanity_check:
            n_clusters = 3
        else:
            n_clusters = gmm_cluster_prior(
                min_clusters=self.hparams.min_clusters,
                max_clusters=self.hparams.max_clusters,
            )

        if self.hparams.sanity_check:
            self.means = [torch.tensor([-2., 2.]), torch.tensor([2., 2.]), torch.tensor([0., -2.])]
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
            covs = [torch.eye(dims) * 0.5 for _ in range(n_clusters)]
        else:
            stddevs = gmm_stddev_prior(
                n_clusters=n_clusters,
                stddev_concentration=2.,
                stddev_rate=2.,
            )
            covs = [torch.eye(dims) * stddev for stddev in stddevs]

        if self.hparams.sanity_check:
            mixture_probs = torch.tensor([1 / n_clusters for _ in range(n_clusters)])
        else:
            mixture_probs = gmm_mixture_probs_prior(
                self.hparams.gmm_categories_concentration,
                n_clusters,
            )

        print(f"no. of clusters: {n_clusters}")
        print(f"means: {self.means}")
        print(f"covs:\n{covs}")
        print(f"mixture probs: {mixture_probs}")

        self.data_gen_process = GMM(
            self.means,
            covs,
            mixture_probs,
        )
        data, mixtures = self.data_gen_process.samples(self.hparams.dataset_size)
        data = data.unsqueeze(1)
        self._data_df = pd.concat([pd.DataFrame(x, columns=["x", "y"]) for x in data])

        image_name = kde_plot_2d(
            data.squeeze(1),
            50,
            (self.center_box[0] - 2, self.center_box[1] + 2),
            [m.tolist() for m in self.means],
            os.path.join(self.hparams.folder, "data"),
        )
        self.logger.experiment.add_image("Data", (np.array(Image.open(image_name).convert("RGB")) * 255).astype(np.uint8), dataformats="HWC")

        # Scaling data for VP-SDE
        self.data = self._scaling(data)

        # Conditioning the DDPM on the GMM cluster means = mode
        conditions = torch.cat([
            self.data_gen_process.get_mixture_parameters(mixture)[0].unsqueeze(0)
            for mixture in mixtures
        ])
        conditions = conditions.unsqueeze(2)

        self.dataset = StackDataset(data=TensorDataset(self.data), conditions=TensorDataset(conditions))

    def _scaling(self, data):
        self._data_min = data.min(axis=0).values
        self._data_max = data.max(axis=0).values
        _data_std = (data - self._data_min) / (self._data_max - self._data_min)
        data = _data_std * 2 - 1
        return data
    
    def _inverse(self, data):
        return 0.5 * (data + 1) * (self._data_max - self._data_min) + self._data_min

    def _inverse_dim(self, data, i):
        return 0.5 * (data + 1) * (self._data_max[0, i] - self._data_min[0, i]) + self._data_min[0, i]

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=2048,
            shuffle=True,
            num_workers=1,
        )

    def val_dataloader(self):
        # TODO: Include actual val dataset
        return DataLoader(
            dataset=self.dataset,
            batch_size=128,
            shuffle=True,
            num_workers=1,
        )
    
    def test_dataloader(self):
        # TODO: Include actual test dataset
        return DataLoader(
            dataset=self.dataset,
            batch_size=128,
            shuffle=True,
            num_workers=1,
        )

    def training_step(self, batch, batch_idx):
        data = batch["data"][0]
        conditions = batch["conditions"][0]
        batch_size = data.shape[0]

        timesteps = torch.randint(0, self.hparams.train_timesteps, (batch_size,)).to(self.device).long()
        noise = torch.randn_like(data).to(self.device)
        conditions = None if torch.rand(1) < self.hparams.uncond_prob else conditions

        diffusion = self._get_diffusion()
        noisy_data = diffusion.add_noise(data, noise, timesteps)
        noise_pred = diffusion.noise_pred(noisy_data, timesteps, conditions)

        loss = F.l1_loss(noise, noise_pred)
        return loss
    
    def validation_step(self, batch, batch_idx):
        return

    def on_validation_epoch_end(self):
        diffusion = self._get_diffusion()

        def _unconditional_likelihoods(deterministic, noisy_data):
            x_uncond = diffusion.sample(noisy_data, deterministic=deterministic)
            samples = self._inverse(x_uncond.cpu())
            nll = -self.data_gen_process.log_likelihood(samples)
            return nll

        # Train nll
        generator = torch.cuda.manual_seed(1) if "cuda" in str(self.device) else torch.manual_seed(1)
        train_noise = torch.randn(self.data.shape, generator=generator, device=self.device)
        timestep = torch.full((self.data.shape[0],), 0, dtype=torch.long, device=self.device)
        noisy_train_data = diffusion.add_noise(self.data.to(self.device), train_noise, timestep)

        ddim_nll_train = _unconditional_likelihoods(True, noisy_train_data)
        ddpm_nll_train = _unconditional_likelihoods(False, noisy_train_data)

        # Validation nll
        n_samples = min(10000, self.data.shape[0])
        generator = torch.cuda.manual_seed(1) if "cuda" in str(self.device) else torch.manual_seed(1)
        noisy_val_data = torch.randn(self.data[:n_samples].shape, generator=generator, device=self.device)

        ddim_nll_val = _unconditional_likelihoods(True, noisy_val_data)
        ddpm_nll_val = _unconditional_likelihoods(False, noisy_val_data)

        self.log_dict({
            "train/ddim_nll": ddim_nll_train,
            "train/ddpm_nll": ddpm_nll_train,
            "val/ddim_nll": ddim_nll_val,
            "val/ddpm_nll": ddpm_nll_val,
        })

    def test_step(self, batch, batch_idx):
        return

    def on_test_epoch_end(self):
        diffusion = self._get_diffusion()

        # Validation example for visualisation
        n_samples = min(10000, self.data.shape[0])
        generator = torch.cuda.manual_seed(0) if "cuda" in str(self.device) else torch.manual_seed(0)
        val_noise_for_vis = torch.randn(self.data[:n_samples].shape, generator=generator, device=self.device)

        def _unconditional_trajectories(deterministic, range_skip):
            title = "DDIM" if deterministic else "DDPM"
            trajectory_callback = TrajectoryCallback()
            callbacks = [trajectory_callback]
            x_uncond = diffusion.sample(val_noise_for_vis, deterministic=deterministic, callbacks=callbacks)

            # Trajectories
            timesteps = self.hparams.sample_timesteps if deterministic else self.hparams.train_timesteps
            x_timesteps, x_trajs = trajectory_callback.sample(n=min(1000, self.data.shape[0]), dim=0)
            y_timesteps, y_trajs = trajectory_callback.sample(n=min(1000, self.data.shape[0]), dim=1)
            _len = len(x_timesteps)
            idxs = list(range(0, _len, range_skip))
            if idxs[-1] != _len - 1:
                idxs += [_len - 1]

            # TODO: Scale the y-axis on the plot but not the data
            x_trajs = [[x_traj[i] for i in idxs] for x_traj in x_trajs]
            y_trajs = [[y_traj[i] for i in idxs] for y_traj in y_trajs]
            x_timesteps = [x_timesteps[i] for i in idxs]
            y_timesteps = [y_timesteps[i] for i in idxs]

            lims = (-3, 3)
            image_name_x = trajectory_plot_1d(x_timesteps, x_trajs, timesteps, lims, 4, os.path.join(self.hparams.folder, f"unconditional_trajectory_x_{title}_{self.current_epoch}"), 0.2, 0.2)
            image_name_y = trajectory_plot_1d(y_timesteps, y_trajs, timesteps, lims, 4, os.path.join(self.hparams.folder, f"unconditional_trajectory_y_{title}_{self.current_epoch}"), 0.2, 0.2)

            self.logger.experiment.add_image("test/x_trajectory", (np.array(Image.open(image_name_x).convert("RGB")) * 255).astype(np.uint8), dataformats="HWC")
            self.logger.experiment.add_image("test/y_trajectory", (np.array(Image.open(image_name_y).convert("RGB")) * 255).astype(np.uint8), dataformats="HWC")

        def _kde_plots():
            ddim = diffusion.sample(val_noise_for_vis, deterministic=True)
            ddim = self._inverse(ddim.cpu())
            ddpm = diffusion.sample(val_noise_for_vis, deterministic=False)
            ddpm = self._inverse(ddpm.cpu())
            original = self._inverse(self.data)
            lims = (self.center_box[0] - 2, self.center_box[1] + 2)
            image_name = kde_plot_2d_compare(ddim.squeeze(1), ddpm.squeeze(1), original.squeeze(1), lims, lims, os.path.join(self.hparams.folder, f"unconditional_kde_{self.current_epoch}"))
            self.logger.experiment.add_image("test/kde", (np.array(Image.open(image_name).convert("RGB")) * 255).astype(np.uint8), dataformats="HWC")

        # _unconditional_trajectories(True, 1)
        # _unconditional_trajectories(False, 10)
        _kde_plots()

    def configure_optimizers(self):
        return torch.optim.Adam(self.unet.parameters(), lr=1e-5)


def main():
    folder = "experiment_samples"

    # TODO: run experiments where we vary all aspects of the dataset, including number of cluster
    for seed in range(0, 30):
        seed_everything(seed)

        subfolder = f"{folder}/{seed}"
        os.makedirs(subfolder, exist_ok=True)

        tb_logger = TensorBoardLogger(save_dir=folder, version=seed)

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
            cluster_separation=3.,
            folder=subfolder,
            sanity_check=False,
        )
        trainer = Trainer(
            accelerator="gpu",
            max_epochs=2000,
            check_val_every_n_epoch=100,
            logger=tb_logger,
        )
        trainer.fit(diffusion)
        trainer.test(diffusion)
