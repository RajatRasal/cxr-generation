import argparse
import os
from itertools import product
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.tuner.tuning import Tuner

from datasets.mnist.datamodules import MNISTDataModule, get_mnist_variant
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
        num_sanity_val_steps=-1,
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
        num_sanity_val_steps=-1,
    )

    # Tuning batch size is unsupported with distributed strategies in lightning 2.2.4
    if args.device_id != -1 and not args.no_tuner:
        tuner = Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=dm, mode="power")

    trainer.fit(model, datamodule=dm)


def mnist_editing_visualisations():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--version", type=int, default=0)
    args = parser.parse_args()

    output_dir = os.path.join(args.logdir, "editing")
    os.makedirs(output_dir, exist_ok=True)

    ckpt_path = os.path.join(args.logdir, f"lightning_logs/version_{args.version}/checkpoints/last.ckpt")
    model = GuidedDiffusionLightningModule.load_from_checkpoint(ckpt_path, device="cuda")

    seed_everything(model.hparams.seed)
    generator = get_generator(model.hparams.seed, model.device)

    # TODO: Load colour from a file
    bs = 10
    dm = get_mnist_variant("grey", model.hparams.seed, bs, 0)
    dm.setup("test")
    for batch in dm.test_dataloader():
        data = batch["image"].to("cuda")
        labels = batch["label"].to("cuda")
        background = torch.ones_like(labels) * (model.hparams.classes + 1)
        null_tokens = torch.ones_like(labels) * model.hparams.classes
        conditions = model.class_embeddings(torch.cat([labels, background], dim=1))
        null_tokens = model.class_embeddings(torch.cat([null_tokens, background], dim=1))
        break

    diffusion = model._get_diffusion()
    inverted_latents = diffusion.ddim_inversion(
        data,
        conditions,    
        50,
        disable_progress_bar=True,
    )
    cfg_recons = diffusion.sample(
        xT=inverted_latents,
        null_token=null_tokens,
        conditions=conditions,
        guidance_scale=1.0,
        deterministic=False,
        timesteps=50,
        do_cfg=True,
        disable_progress_bar=True,
    )

    # Define CFG edit plots
    fig, axes = plt.subplots(
        nrows=bs,
        ncols=10 + 2,
        figsize=(10, 10),
        gridspec_kw={"hspace": 0.1, "wspace": 0.1},
    )  
    _data = model._postprocess_images(data).detach().cpu().numpy()
    cfg_recons = model._postprocess_images(cfg_recons).detach().cpu().numpy()
    # Plot CFG reconstructions
    for i in range(bs):
        axes[i, 0].imshow(np.moveaxis(_data[i], 0, 2), cmap="grey")
        axes[i, 0].set_axis_off()
        axes[i, 1].imshow(np.moveaxis(cfg_recons[i], 0, 2), cmap="grey")
        axes[i, 1].set_axis_off()

    for target in range(10): 
        # Prepare edit condition
        edit_conditions = torch.tensor(target).unsqueeze(-1).repeat(data.shape[0], 1)
        background = torch.ones_like(edit_conditions) * (model.hparams.classes + 1)
        edit_conditions = torch.cat([edit_conditions, background], dim=1).to("cuda")
        edit_conditions = model.class_embeddings(edit_conditions)
        # Perform editing via classifier-free guidance
        edits = diffusion.sample(
            xT=inverted_latents,
            null_token=null_tokens,
            conditions=edit_conditions,
            guidance_scale=5.0,
            deterministic=False,
            timesteps=50,
            do_cfg=True,
            disable_progress_bar=True,
        )
        # Unnormalise image for display
        edits = model._postprocess_images(edits).detach().cpu().numpy()
        # Display images
        for row in range(bs):
            axes[row, target + 2].imshow(np.moveaxis(edits[row], 0, 2), cmap="grey")
            axes[row, target + 2].set_axis_off()

    fig.savefig(os.path.join(output_dir, "edits_cfg.pdf"), format="pdf", bbox_inches="tight")

    # Set up UNet for cross-attention control editing
    model.unet = set_attention_processor(model.unet)

    fontsize = 10
    for nti_steps, guidance_scale, swap_fraction in product([10, 50], [2.0, 6.0, 10.], [0.1, 0.2, 0.3]):
        print(f"NTI steps: {nti_steps}, Guidance scale: {guidance_scale}, swap_fraction: {swap_fraction}")
        # Define plots for NTO
        fig, axes = plt.subplots(
            nrows=bs,
            ncols=10 + 2,
            figsize=(10, 10),
            gridspec_kw={"hspace": 0.1, "wspace": 0.1},
        )
        fig.suptitle(rf"NTI steps: {nti_steps} | $\omega$ = {guidance_scale} | Swap fraction: {swap_fraction}", fontsize=fontsize)
        axes[0, 0].set_title("Orig.", fontsize=fontsize)
        axes[0, 1].set_title("Recon.", fontsize=fontsize)
        for i in range(bs):
            axes[0, i + 2].set_title(f"Label: {i}", fontsize=fontsize)

        # Perform null-token optimisation
        nto = NullTokenOptimisation(diffusion, null_tokens, nti_steps, 1e-3, guidance_scale, 50)
        nto.fit(data, conditions, disable_progress_bar=True, generator=generator)

        # Plot NTO reconstructions
        nto_recons = nto.generate(generator=generator, disable_progress_bar=True)
        nto_recons = model._postprocess_images(nto_recons).detach().cpu().numpy()
        for i in range(bs):
            axes[i, 0].imshow(np.moveaxis(_data[i], 0, 2), cmap="grey")
            axes[i, 0].set_axis_off()
            axes[i, 1].imshow(np.moveaxis(nto_recons[i], 0, 2), cmap="grey")
            axes[i, 1].set_axis_off()

        for target in range(10):
            # Prepare edit condition
            edit_conditions = torch.tensor(target).unsqueeze(-1).repeat(data.shape[0], 1)
            background = torch.ones_like(edit_conditions) * (model.hparams.classes + 1)
            edit_conditions = torch.cat([edit_conditions, background], dim=1).to("cuda")
            edit_conditions = model.class_embeddings(edit_conditions)
            # Perform editing via cross-attention control
            edits = nto.generate(
                condition=edit_conditions,
                generator=generator,
                disable_progress_bar=True,
                corrections=False,
                swap_fraction=swap_fraction,
            )
            # Unnormalise image for display
            edits = model._postprocess_images(edits).detach().cpu().numpy()
            # Display images
            for row in range(bs):
                axes[row, target + 2].imshow(np.moveaxis(edits[row], 0, 2), cmap="grey")
                axes[row, target + 2].set_axis_off()

        fig.savefig(
            os.path.join(output_dir, f"edits_nto_{nti_steps}_{guidance_scale}_{swap_fraction}.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        plt.close()

