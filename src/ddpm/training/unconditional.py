import io
import logging
from typing import Dict, List, Literal, Optional, Tuple

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import UNet2DModel
from torchvision.utils import make_grid, save_image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from ddpm.diffusion.diffusion import Diffusion
from ddpm.utils import get_generator


logger = logging.getLogger("lightning.pytorch")


class DiffusionLightningModule(L.LightningModule):

    def __init__(
        self,
        sample_shape: int = 28,
        dim: int = 16,
        dim_mults: List[int] = [2, 4, 4],
        blocks: List[Literal["Block", "ResnetBlock", "AttnBlock"]] = ["Block", "Block", "Block"],
        layers_per_block: int = 2,
        train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: Literal["linear", "scaled_linear", "cosine", "sigmoid"] = "linear",
        rescale_betas_zero_snr: bool = False,
        learning_rate: float = 1e-4,
        seed: int = 0,
        channels: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        assert len(self.hparams.dim_mults) == len(self.hparams.blocks)

        blocks_search = {
            up_down: {
                block_name.replace("XX", ""): block_name.replace("XX", up_down) + "2D"
                for block_name in ["XXBlock", "ResnetXXsampleBlock", "AttnXXBlock"]
            }
            for up_down in ["Down", "Up"]
        }
        down_block_types = [blocks_search["Down"][b] for b in self.hparams.blocks]
        up_block_types = [blocks_search["Up"][b] for b in self.hparams.blocks[::-1]]
        block_out_channels = [self.hparams.dim * dim for dim in self.hparams.dim_mults]
        logger.info(f"Down block types: {down_block_types}")
        logger.info(f"Up block types: {up_block_types}")
        logger.info(f"Block channels: {block_out_channels}")

        self.unet = UNet2DModel(
            sample_size=self.hparams.sample_shape,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            in_channels=self.hparams.channels,
            out_channels=self.hparams.channels,
            block_out_channels=block_out_channels,
            layers_per_block=self.hparams.layers_per_block,
            dropout=self.hparams.dropout,
        )

    def _get_diffusion(self) -> Diffusion:
        return Diffusion(
            self.unet,
            self.hparams.train_timesteps,
            self.hparams.beta_start,
            self.hparams.beta_end,
            self.hparams.beta_schedule,
            self.hparams.rescale_betas_zero_snr,
            self.device,
        )

    def _noise_pred(self, diffusion, data):
        timesteps = torch.randint(
            low=0,
            high=self.hparams.train_timesteps,
            size=(data.shape[0],),
            dtype=torch.long,
            device=self.device,
        )
        noise = torch.randn_like(data).to(self.device)

        noisy_data = diffusion.add_noise(data, noise, timesteps)
        noise_pred = diffusion.noise_pred(noisy_data, timesteps)

        loss = F.mse_loss(noise, noise_pred)

        return noise, noise_pred

    def _loss(self, diffusion, data):
        noise, noise_pred = self._noise_pred(diffusion, data)
        loss = F.mse_loss(noise, noise_pred)
        return loss

    def _preprocess_images(self, images: torch.FloatTensor) -> torch.FloatTensor:
        # # Assuming all image pixels have values between 0 and 255
        # # [0, 1]
        # images = images / 255
        # # [-0.5, 0.5]
        # images = images - 0.5
        # # [-1, 1]
        # images = images * 2  # .clamp(-1, 1)
        return images

    def _postprocess_images(self, images: torch.FloatTensor) -> torch.FloatTensor:
        # [0, 1]
        return images * 0.3801 + 0.1307
        # return (images / 2 + 0.5).clamp(0, 1)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx):
        data = batch["image"]
        data = self._preprocess_images(data)
        diffusion = self._get_diffusion()

        loss = self._loss(diffusion, data)
        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def _samples_plot(self, diffusion, noise, loc, clipping, nrows, ncols, ndisplay):
        assert nrows * ncols == ndisplay
        assert noise.shape[0] == ndisplay

        fs = 20
        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(10, 10),
            gridspec_kw={"hspace": 0.1, "wspace": 0.1},
        )
        for i, deterministic in enumerate([False, True]):
            axes[0, i].set_title("DDIM" if deterministic else "DDPM", fontsize=fs)
            for j, ts_scale in enumerate([20, 1]):
                pred_timesteps = self.hparams.train_timesteps // ts_scale
                if i == 0:
                    axes[j, 0].text(
                        -10, 100,
                        f"T = {pred_timesteps}",
                        verticalalignment="center",
                        rotation=90, fontsize=fs,
                    )
                samples = diffusion.sample(
                    noise,
                    deterministic=deterministic,
                    timesteps=pred_timesteps,
                    disable_progress_bar=True,
                    scheduler_step_kwargs={"use_clipped_model_output": clipping} if deterministic else {},
                )
                samples = self._postprocess_images(samples.detach()).cpu().numpy()
                _fig, _axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3, 3))
                for k in range(ndisplay):
                    ax = _axes.flatten()[k]
                    # TODO: Plot as grayscale
                    ax.imshow(np.moveaxis(samples[k], 0, 2), cmap="gray")
                    ax.set_axis_off()
                buf = io.BytesIO()
                _fig.savefig(buf, format="png", bbox_inches="tight")
                axes[j, i].imshow(Image.open(buf).convert("RGB"))  # , cmap="gray")
                axes[j, i].set_axis_off()
        fig.savefig(loc, format="png", bbox_inches="tight")
        plt.close("all")

    def _samples_plot_to_logger(self, diffusion, noise, name, nrows, ncols, ndisplay):
        big_buf = io.BytesIO()
        self._samples_plot(diffusion, noise, big_buf, True, nrows, ncols, ndisplay)
        big_img = np.array(Image.open(big_buf).convert("RGB")).astype(np.uint8)
        self.logger.experiment.add_image(
            name,
            big_img,
            self.global_step,
            dataformats="HWC",
        )

    def _val_step(self, batch, batch_idx, step_type: Literal["val", "test"]):
        data = batch["image"]
        data = self._preprocess_images(data)
        diffusion = self._get_diffusion()

        # Loss
        loss = self._loss(diffusion, data)
        self.log(f"{step_type}/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        if step_type == "test":
            # Calculate FID and IS scores
            noise = torch.randn_like(data)
            samples = diffusion.sample(
                noise,
                deterministic=False,
                timesteps=self.hparams.train_timesteps,
                disable_progress_bar=True,
            )
            samples = self._postprocess_images(samples.detach()).repeat(1, 3, 1, 1)
            unnormalised_data = self._postprocess_images(data.detach()).repeat(1, 3, 1, 1)
            self.fid.update(samples, real=False)
            self.fid.update(unnormalised_data, real=True)
            self.is_score.update(samples)
            
        if batch_idx == 0:
            pred_timesteps = 1000
            nrows = ncols = 10
            ndisplay = ncols * nrows

            if self.global_step == 0:
                # Display data
                display_data = self._postprocess_images(data[:ndisplay])
                grid = make_grid(display_data, nrow=nrows, normalize=True)
                self.logger.experiment.add_image(f"{step_type}/data", grid, self.global_step)
            
            # Random samples
            noise = torch.randn(
                (ndisplay, self.hparams.channels, self.hparams.sample_shape, self.hparams.sample_shape),
                generator=self.generator,
                device=self.device,
            )
            self._samples_plot_to_logger(diffusion, noise, f"{step_type}/Random Samples", nrows, ncols, ndisplay)

            # Reconstruction samples
            inverted_latents = diffusion.ddim_inversion(
                data[:ndisplay],
                timesteps=pred_timesteps,
                use_clipping=True,
                disable_progress_bar=True,
            ).detach()
            self._samples_plot_to_logger(diffusion, inverted_latents, f"{step_type}/Reconstruction", nrows, ncols, ndisplay)
 
    def on_validation_epoch_start(self):
        self.generator = get_generator(self.hparams.seed, self.device)

    def validation_step(self, batch, batch_idx):
        return self._val_step(batch, batch_idx, "val")

    def on_test_epoch_start(self):
        self.val_generator = get_generator(self.hparams.seed, self.device)
        self.fid = FrechetInceptionDistance(
            feature=64,
            normalize=True,
            compute_on_cpu=True,
        ).to(self.device)
        self.is_score = InceptionScore(
            feature=64,
            normalize=True,
            compute_on_cpu=True,
        ).to(self.device)

    def test_step(self, batch, batch_idx):
        return self._val_step(batch, batch_idx, "test")

    def on_test_epoch_end(self):
        fid_score = self.fid.compute()
        is_score = self.is_score.compute()[0]
        log_dict = {"test/is": is_score.item(), "test/fid": fid_score.item()}
        self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True)
        return log_dict

    def configure_optimizers(self):
        return torch.optim.Adam(self.unet.parameters(), lr=self.hparams.learning_rate)
