import os

import pytest
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from ddpm.training.callbacks import TrajectoryCallback
from ddpm.editing.null_text_inversion import NullTokenOptimisation
from ddpm.training.train import DiffusionLightningModule
from ddpm.training.plots import trajectory_plot_1d_with_inverse, trajectory_plots_1d_with_inverse
from ddpm.utils import get_device


@pytest.mark.dependency(depends=["training"])
def test_nto(seed, model_folder_and_name_and_exp):
    # Setup directories
    model_folder, model_name, exp_name = model_folder_and_name_and_exp
    path = os.path.join(model_folder, f"{model_name}.ckpt")

    # Load model
    model = DiffusionLightningModule.load_from_checkpoint(path)
    model.setup("test")
    test_dataloader = model.test_dataloader()

    # Diffusion
    diffusion = model._get_diffusion()

    # NTI
    n_samples = 1000
    device = get_device()
    errors = {(m.flatten()[0].item(), m.flatten()[1].item()): [] for m in model.means}

    nto = NullTokenOptimisation(diffusion, model.null_token.to(device), 50, 1e-4, 1)

    # Test data
    ddim_trajectory_callback = TrajectoryCallback() 
    nto_trajectory_callback = TrajectoryCallback() 
    for x in test_dataloader:
        data = x["data"][0].to(device)
        conditions = x["conditions"][0].to(device)
        nto.fit(data, conditions, ddim_inversion_callbacks=[ddim_trajectory_callback])
        recon = nto.generate(callbacks=[nto_trajectory_callback])
        errs = F.mse_loss(recon, data, reduce=False)
        for cond, err in zip(conditions, errs):
            _cond = (cond.flatten()[0].item(), cond.flatten()[1].item())
            errors[_cond].append(err)

    # Aggregate test results 
    for cond, losses in errors.items():
        print(cond)
        if len(losses) != 0:
            print(sum(losses) / len(losses))
        print()

    # DDIM Inversion + NTO + Guided Recon (DDPM)
    ddim_inversion_x = ddim_trajectory_callback.sample(n_samples, dim=0)
    ddim_inversion_y = ddim_trajectory_callback.sample(n_samples, dim=1)
    nto_recon_x = nto_trajectory_callback.sample(n_samples, dim=0)
    nto_recon_y = nto_trajectory_callback.sample(n_samples, dim=1)
    trajectory_plot_1d_with_inverse(
        trajectories=[x.detach().numpy().flatten() for x in nto_recon_x],
        inverse_trajectories=[x.detach().numpy().flatten() for x in ddim_inversion_x],
        save_path=os.path.join(model.hparams.folder, "nto_recon_x"),
        true_data=None,
        T=model.hparams.sample_timesteps,
        y_lims=(-2, 2),
        kde_bandwidth=0.1,
        output_type="pdf",
        fast=True,
    )
    trajectory_plot_1d_with_inverse(
        trajectories=[x.detach().numpy().flatten() for x in nto_recon_y],
        inverse_trajectories=[x.detach().numpy().flatten() for x in ddim_inversion_y],
        save_path=os.path.join(model.hparams.folder, "nto_recon_y"),
        true_data=None,
        T=model.hparams.sample_timesteps,
        y_lims=(-2, 2),
        kde_bandwidth=0.1,
        output_type="pdf",
        fast=True,
    )


@pytest.mark.dependency(depends=["training"])
def test_nto_edit(seed, model_folder_and_name_and_exp):
    # Setup directories
    model_folder, model_name, exp_name = model_folder_and_name_and_exp
    path = os.path.join(model_folder, f"{model_name}.ckpt")

    # Load model
    model = DiffusionLightningModule.load_from_checkpoint(path)
    model.setup("test")
    test_dataloader = model.test_dataloader()

    # Diffusion
    diffusion = model._get_diffusion()

    # NTI
    device = get_device()
    errors = {(m.flatten()[0].item(), m.flatten()[1].item()): [] for m in model.means}

    nto = NullTokenOptimisation(diffusion, model.null_token.to(device), 100, 1e-3, 1.1)

    for seed in range(5):
        generator = torch.manual_seed(seed)

        # Test data
        ddim_trajectory_callback = TrajectoryCallback() 
        recon_trajectory_callback = TrajectoryCallback() 
        nto_trajectory_callback = TrajectoryCallback() 
        n_samples = 0
        for x in test_dataloader:
            data = x["data"][0].to(device)
            conditions = x["conditions"][0].to(device)
            nto.fit(data, conditions, generator=generator, ddim_inversion_callbacks=[ddim_trajectory_callback])
            recons = nto.generate(generator=generator, callbacks=[recon_trajectory_callback])
            edit_cond = model.means[2].unsqueeze(-1)
            edit = nto.generate(edit_cond, generator=generator, callbacks=[nto_trajectory_callback])
            n_samples += data.shape[0]

        # TODO: Over timesteps what is the MSE between ddim and optimised with error bars.
        # TODO: When does the trajectory converge for DDPM vs NTO with different injection timesteps
        # DDIM Inversion + NTO + Guided Recon (DDPM)
        ddim_inversion_x = ddim_trajectory_callback.sample(n_samples, dim=0)
        ddim_inversion_y = ddim_trajectory_callback.sample(n_samples, dim=1)
        recon_x = recon_trajectory_callback.sample(n_samples, dim=0)
        recon_y = recon_trajectory_callback.sample(n_samples, dim=1)
        nto_edit_x = nto_trajectory_callback.sample(n_samples, dim=0)
        nto_edit_y = nto_trajectory_callback.sample(n_samples, dim=1)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 3))
        traj_errs = []
        recon_diffs = []
        for ddim_x_traj, ddim_y_traj, recon_x, recon_y in zip(ddim_inversion_x, ddim_inversion_y, recon_x, recon_y):
            recon_traj = torch.cat([recon_x.unsqueeze(0), recon_y.unsqueeze(0)])
            recon_diff = (recon_traj[1:] - recon_traj[:-1]).abs().mean(axis=0)
            recon_diffs.append(recon_diff)
            traj_err = ((ddim_x_traj.detach() - recon_x.detach()) ** 2 + (ddim_y_traj.detach() - recon_y.detach()) ** 2).sqrt()
            traj_errs.append(traj_err.unsqueeze(0))
        traj_errs = pd.DataFrame(
            data=[(t, err.item()) for traj_err in traj_errs for t, err in enumerate(traj_err[0])],
            columns=["t", r"$\left\|z_{t} - \hat{z}_{t}\right\|_2$"],
        )
        recons_diff = pd.DataFrame(
            data=[(t, diff.item()) for recon_diff in recon_diffs for t, diff in enumerate(recon_diff)],
            columns=["t", r"$\left\|z_{t} - z_{t-1}\right\|_1$"]
        )
        ax[0].set_title(r"$\text{Trajectory Alignment Error}$")
        sns.boxplot(data=recons_diff, y=r"$\left\|z_{t} - z_{t-1}\right\|_1$", x="t", ax=ax[1])
        ax[1].set_title(r"$\text{Trajectory Convergence}$")
        sns.boxplot(data=traj_errs, y=r"$\left\|z_{t} - \hat{z}_{t}\right\|_2$", x="t", ax=ax[0])
        fig.savefig(os.path.join(model.hparams.folder, f"traj_stats_{seed}.pdf"), bbox_inches="tight")

        trajectory_plots_1d_with_inverse(
            trajectories_x=[x.detach().numpy().flatten() for x in nto_edit_x],
            inverse_trajectories_x=[x.detach().numpy().flatten() for x in ddim_inversion_x],
            trajectories_y=[x.detach().numpy().flatten() for x in nto_edit_y],
            inverse_trajectories_y=[x.detach().numpy().flatten() for x in ddim_inversion_y],
            recons=recons.detach().cpu().numpy(),
            save_path=os.path.join(model.hparams.folder, f"nto_edit_{seed}"),
            true_data=[x["data"][0] for x in model.train_dataloader()][0].squeeze(0).squeeze(1),
            T=model.hparams.sample_timesteps,
            y_lims=(-2, 2),
            kde_bandwidth=0.05,
            output_type="pdf",
            fast=True,
            title=r"$\longrightarrow \text{Post-NTO Guided Edit with DDPM} \longrightarrow$",
        )