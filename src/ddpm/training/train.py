from typing import List, Literal

import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer, seed_everything
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset, StackDataset

from ddpm.training.callbacks import TrajectoryCallback
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
        sanity_check: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.unet = Unet1d(
            self.hparams.dim,
            self.hparams.dim_mults,
            self.hparams.channels,
            self.hparams.resnet_block_groups,
            self.hparams.sinusoidal_pos_emb_theta,
            self.hparams.context_dim,
        )

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

    def setup(self, *args, **kwargs):
        print("Setting up datasets")

        # Training data
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
            self.means = gmm_means_prior(
                n_clusters=n_clusters,
                dimensions=2,
                center_box=(-5., 5.),
            )
        dims = self.means[0].shape[0]
        if self.hparams.sanity_check:
            covs = [torch.eye(dims) * 0.5 for _ in range(n_clusters)]
        else:
            stddevs = gmm_stddev_prior(
                n_clusters=n_clusters,
                stddev_concentration=6.,
                stddev_rate=4.,
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

        self.data_gen_process = GMM(
            self.means,
            covs,
            mixture_probs,
        )
        data, mixtures = self.data_gen_process.samples(self.hparams.dataset_size)
        data = data.unsqueeze(1)
        self._data_df = pd.concat([pd.DataFrame(x, columns=["x", "y"]) for x in data])
        # self._data_mean = data.mean(axis=0)
        # self._data_stddev = data.std(axis=0)
        self.data = data  # (data - self._data_mean) / self._data_stddev
        # conditioning the DDPM on the cluster means = mode
        conditions = torch.cat([
            self.data_gen_process.get_mixture_parameters(mixture)[0].unsqueeze(0)
            for mixture in mixtures
        ])
        conditions = conditions.unsqueeze(2)  # (cond.unsqueeze(2) - self._data_mean) / self._data_stddev
        self.dataset = StackDataset(data=TensorDataset(self.data), conditions=TensorDataset(conditions))

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=2048,
            shuffle=True,
            num_workers=1,
        )

    def val_dataloader(self):
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

    def on_validation_epoch_end(self): # , batch, batch_idx):
        diffusion = self._get_diffusion()

        # Validation example for visualisation
        n_samples = min(5000, self.data.shape[0])
        generator = torch.cuda.manual_seed(0) if "cuda" in str(self.device) else torch.manual_seed(0)
        val_noise_for_vis = torch.randn(self.data[:n_samples].shape, generator=generator, device=self.device)

        ## Unconditional sampling visualisations
        df_x_uncond_original = self._data_df.copy()
        df_x_uncond_original["Dataset"] = "Original"

        fig, axes = plt.subplots(nrows=1, ncols=2) # , figsize=(15, 5))

        def _unconditional_samples(deterministic: bool, n_trajs, iter_chunks, ax_traj):
            title = "DDIM" if deterministic else "DDPM"
            trajectory_callback = TrajectoryCallback()
            callbacks = [trajectory_callback]
            x_uncond = diffusion.sample(val_noise_for_vis, deterministic=deterministic, callbacks=callbacks)

            # Trajectory plot overlaid with KDE from data
            ax_traj.set_title(f"{title} Trajectory")
            sns.kdeplot(
                df_x_uncond_original,
                x="x",
                y="y",
                ax=ax_traj,
                fill=True,
            )
            trajectory_callback.plot(n=n_trajs, iter_chunks=iter_chunks, ax=ax_traj)
            ax_traj.set_xlabel("")
            ax_traj.set_ylabel("")
            # ax_traj.get_legend().remove()

            # KDE from data vs KDE from samples
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.set_xlim(-7, 7)
            ax.set_ylim(-7, 7)
            ax.set_title(f"{title} Samples")
            df_x_uncond = pd.concat([pd.DataFrame(x.cpu().numpy(), columns=["x", "y"]) for x in x_uncond])
            sns.kdeplot(
                df_x_uncond_original,
                x="x",
                y="y",
                label="Original",
                ax=ax,
            )
            sns.kdeplot(
                df_x_uncond,
                x="x",
                y="y",
                label="Samples",
                ax=ax,
            )
            fig.tight_layout()
            fig.savefig(f"experiment_samples/unconditional_{title}_{self.current_epoch}.png", format="png")

        _unconditional_samples(True, 20, 1, axes[0])
        _unconditional_samples(False, 20, self.hparams.train_timesteps // 10, axes[1])

        fig.tight_layout()
        fig.savefig(f"experiment_samples/unconditional_trajectory_{self.current_epoch}.pdf", format="pdf")

        # fig, axes = plt.subplots(nrows=1, ncols=len(self.means))
        # for ax, mean in zip(axes, self.means):
        #     mean_str = f"x: {round(mean[0].item(), 3)}, y: {round(mean[1].item(), 3)}"
        #     ax.set_title(mean_str)
        #     mean = mean.view(1, -1, 1).repeat(n_samples, 1, 1).to(self.device)
        #     x_cond_deterministic = diffusion.sample(val_noise_for_vis, mean, 7.5, True)
        #     dfs = []
        #     for i, sample in enumerate(x_cond_deterministic):
        #         df = pd.DataFrame(sample.squeeze(1).transpose(0, 1).cpu().numpy(), columns=["x", "y"])
        #         df["Sample"] = str(i + 1)
        #         dfs.append(df)
        #     sns.kdeplot(pd.concat(dfs), x="x", y="y", hue="Sample", ax=ax)
        # fig.savefig(f"{self.current_epoch}.png")

    def configure_optimizers(self):
        return torch.optim.Adam(self.unet.parameters(), lr=1e-5)


def main():
    # TODO: run experiments where we vary all aspects of the dataset, including number of cluster
    seed_everything(0)

    diffusion = DiffusionLightningModule(
        dim=20,
        dim_mults=[1],
        channels=1,
        train_timesteps=1000,
        sample_timesteps=50,
        dataset_size=10000,
        beta_schedule="cosine",
        min_clusters=2,
        max_clusters=5,
        sanity_check=True,
    )
    trainer = Trainer(accelerator="gpu", max_epochs=1000, check_val_every_n_epoch=100)
    trainer.fit(diffusion)
