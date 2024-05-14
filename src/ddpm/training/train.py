from typing import List, Literal

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer, seed_everything
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from ddpm.models.unet import Unet as Unet1d
from ddpm.diffusion.diffusion import Diffusion
from ddpm.datasets.bgmm import gmm_dataset, gmm_stddev_prior, gmm_means_prior, gmm_cluster_prior


class Custom_Dataset(Dataset):
    def __init__(self, _dataset, _conds):
        self.dataset = _dataset
        self.conds = _conds

    def __getitem__(self, index):
        example = self.dataset[index]
        cond = self.conds[index]
        return np.array(example), np.array(cond)

    def __len__(self):
        return len(self.dataset)


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
        # n_clusters = gmm_cluster_prior(
        #     min_clusters=self.hparams.min_clusters,
        #     max_clusters=self.hparams.max_clusters,
        # )
        n_clusters = 3
        self.means = [torch.tensor([-1., 1.]), torch.tensor([1., 1.]), torch.tensor([0., -1.])]
        # self.means = gmm_means_prior(
        #     n_clusters=n_clusters,
        #     dimensions=2, # self.hparams.channels,
        #     center_box=(-2., 2.),
        # )
        stddevs = [torch.tensor(0.5), torch.tensor(0.5), torch.tensor(0.5)]
        # stddevs = gmm_stddev_prior(
        #     n_clusters=n_clusters,
        #     stddev_concentration=1.5,
        #     stddev_rate=1.,
        # )
        print(f"no. of clusters: {n_clusters}, means: {self.means}, stddevs: {stddevs}")
        data, cond = gmm_dataset(
            self.hparams.dataset_size,
            means=self.means,
            stddevs=stddevs,
            categories_concentration=self.hparams.gmm_categories_concentration,
        )
        self.data = data.unsqueeze(1)
        cond = cond.unsqueeze(2)
        self._data_df = pd.concat([pd.DataFrame(x, columns=["x", "y"]) for x in self.data])
        self.dataset = Custom_Dataset(self.data, cond)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=1024,
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
        data, conditions = batch
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
        # TODO: FID and IS score
        print("validation")

        diffusion = self._get_diffusion()

        # Validation example for visualisation
        n_samples = min(5000, self.data.shape[0])
        generator = torch.cuda.manual_seed(0) if "cuda" in str(self.device) else torch.manual_seed(0)
        val_noise_for_vis = torch.randn(self.data[:n_samples].shape, generator=generator, device=self.device)

        ## Unconditional sampling visualisations
        df_x_uncond_original = self._data_df.copy()
        df_x_uncond_original["Dataset"] = "Original"

        def _unconditional_samples(deterministic: bool):
            title = "DDIM" if deterministic else "DDPM"
            x_uncond = diffusion.sample(val_noise_for_vis, deterministic=deterministic)

            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_title(f"Unconditional {title} Samples")
            df_x_uncond = pd.concat([pd.DataFrame(x.cpu().numpy(), columns=["x", "y"]) for x in x_uncond])
            df_x_uncond["Dataset"] = "Samples"
            sns.kdeplot(
                pd.concat([df_x_uncond, df_x_uncond_original]),
                x="x",
                y="y",
                hue="Dataset",
                ax=ax,
            )
            fig.tight_layout()
            fig.savefig(f"experiment_samples/unconditional_{title}_{self.current_epoch}.png", format="png")

        _unconditional_samples(True)
        _unconditional_samples(False)

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
    seed_everything(10)

    diffusion = DiffusionLightningModule(
        dim=20,
        dim_mults=[1],
        channels=1,
        train_timesteps=1000,
        sample_timesteps=50,
        dataset_size=10000,
        beta_schedule="cosine",
    )
    trainer = Trainer(accelerator="gpu", max_epochs=1000, check_val_every_n_epoch=100)
    trainer.fit(diffusion)
