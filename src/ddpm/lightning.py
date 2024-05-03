from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from torchvision.datasets.mnist import MNIST
from torchvision import transforms


class DDPMMNIST(pl.LightningModule):

    def __init__(
        self,
        batch_size: int = 512,
        train_timesteps: int = 200,
        sampling_timesteps: int = 50,
        cond_prob_drop: float = 0.6,
        latent_dim: int = 128,
        dim_mults: List[int] = [1, 1, 2],
        guidance_scale: float = 7.5,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.unet = Unet(
            dim=self.hparams.latent_dim,
            dim_mults=self.hparams.dim_mults,
            num_classes=10,
            cond_drop_prob=self.hparams.cond_prob_drop,
            channels=1,
        )
        self.diffusion = GaussianDiffusion(
            model=self.unet,
            image_size=28,
            timesteps=self.hparams.train_timestep,
            sampling_timesteps=self.hparams.sampling_timesteps,
            objective="pred_noise",
        )

    def forward(self, images, labels):
        images, labels = x
        pred_noise, timesteps, target_noise = self.diffusion(images, classes=labels)
        return pred_noise, timesteps, target_noise

    def training_step(self, batch, batch_idx):
        images, labels = batch
        pred_noise, timesteps, target_noise = self(images, labels)
        loss = self.diffusion.p_losses(pred_noise, timesteps, target_noise)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        pred_noise, timesteps, target_noise = self(images, labels)
        loss = self.diffusion.p_losses(pred_noise, timesteps, target_noise)
        self.log('validation_loss', loss)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        pred_noise, timesteps, target_noise = self(images, labels)
        loss = self.diffusion.p_losses(pred_noise, timesteps, target_noise)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.diffusion.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument('--hidden_dim', type=int, default=128)
        # parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser
