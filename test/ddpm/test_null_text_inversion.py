import os

import pytest
import torch.nn.functional as F
from tqdm import tqdm

from ddpm.editing.null_text_inversion import NullTokenOptimisation
from ddpm.training.train import DiffusionLightningModule
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

    # Test data
    for x in test_dataloader:
        data = x["data"][0].to(device)
        conditions = x["conditions"][0].to(device)
        break

    # NTI
    n_samples = 100
    device = get_device()
    errors = {(m.flatten()[0].item(), m.flatten()[1].item()): [] for m in model.means}

    nto = NullTokenOptimisation(diffusion, model.null_token.to(device), 10, 0.1, 7.5)

    for i, _ in tqdm(zip(range(data.shape[0]), range(n_samples)), total=n_samples):
        nto.fit(data[i].unsqueeze(0), conditions[i].unsqueeze(0))
        recon = nto.generate()
        cond = (conditions[i].flatten()[0].item(), conditions[i].flatten()[1].item())
        err = F.mse_loss(recon, data[i].unsqueeze(0)).item()
        errors[cond].append(err)
    
    for cond, losses in errors.items():
        print(cond)
        if len(losses) != 0:
            print(sum(losses) / len(losses))
        print()
