import os

import pytest
import torch.nn.functional as F
from tqdm import tqdm

from ddpm.training.callbacks import TrajectoryCallback
from ddpm.editing.null_text_inversion import NullTokenOptimisation
from ddpm.training.train import DiffusionLightningModule
from ddpm.training.plots import trajectory_plot_1d_with_inverse
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
    n_samples = 100
    device = get_device()
    errors = {(m.flatten()[0].item(), m.flatten()[1].item()): [] for m in model.means}

    nto = NullTokenOptimisation(diffusion, model.null_token.to(device), 50, 1e-4, 7.5)

    # Test data
    ddim_trajectory_callback = TrajectoryCallback() 
    nto_trajectory_callback = TrajectoryCallback() 
    for x in test_dataloader:
        data = x["data"][0].to(device)
        conditions = x["conditions"][0].to(device)
        nto.fit(data, conditions, ddim_inversion_callbacks=[ddim_trajectory_callback])
        recon = nto.generate([nto_trajectory_callback])
        # TODO: Euclidean distance between recon and data
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

    # DDIM Inversion + Guided Recon (DDPM)

    # DDIM Inversion + NTO + Guided Recon (DDPM)
    ddim_inversion_x = ddim_trajectory_callback.sample(1000, dim=0)
    ddim_inversion_y = ddim_trajectory_callback.sample(1000, dim=1)
    nto_guided_recon_x = nto_trajectory_callback.sample(1000, dim=0)
    nto_guided_recon_y = nto_trajectory_callback.sample(1000, dim=1)
    trajectory_plot_1d_with_inverse(
        trajectories=[x.detach().numpy().flatten() for x in nto_guided_recon_x],
        inverse_trajectories=[x.detach().numpy().flatten() for x in ddim_inversion_x],
        save_path=os.path.join(model.hparams.folder, "nto_guided_recon_x"),
        true_data=None,
        T=model.hparams.sample_timesteps,
        y_lims=(-2, 2),
        kde_bandwidth=0.1,
        output_type="pdf",
        fast=True,
    )
    trajectory_plot_1d_with_inverse(
        trajectories=[x.detach().numpy().flatten() for x in nto_guided_recon_y],
        inverse_trajectories=[x.detach().numpy().flatten() for x in ddim_inversion_y],
        save_path=os.path.join(model.hparams.folder, "nto_guided_recon_y"),
        true_data=None,
        T=model.hparams.sample_timesteps,
        y_lims=(-2, 2),
        kde_bandwidth=0.1,
        output_type="pdf",
        fast=True,
    )
