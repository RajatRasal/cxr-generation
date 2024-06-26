import torch
import torch.nn.functional as F
from tqdm import tqdm

from ddpm.diffusion.diffusion import Diffusion


@torch.no_grad()
def classify(
    diffusion: Diffusion,
    data: torch.FloatTensor,
    classes: torch.FloatTensor,
    n_trials: int = 3,
) -> torch.LongTensor:
    """
    data = (batch_size, channels, h, w)
    classes = (n_classes, 2 = (class, background), embedding_dim)
    """
    errors = {c: [] for c in range(classes.shape[0])}
    for trial in tqdm(range(n_trials), total=n_trials, desc="Generative Classification"):
        timesteps = torch.randint(
            low=0,
            high=diffusion.train_timesteps,
            size=(data.shape[0],),
            dtype=torch.long,
            device=data.device,
        )
        noise = torch.randn_like(data)
        noisy_data = diffusion.add_noise(data, noise, timesteps)
        for c in range(classes.shape[0]):
            # TODO: Make the usage of repeat method general for all shapes of conditions
            conditions = classes[c].unsqueeze(0).repeat(data.shape[0], 1, 1)
            noise_pred = diffusion.noise_pred(noisy_data, timesteps, conditions)
            # (batch_size,)
            error = F.mse_loss(noise, noise_pred, reduction="none").mean(dim=(1, 2, 3)).cpu()
            errors[c].append(error.unsqueeze(0))
    errors_by_classes = [
        torch.cat(errors[c]).mean(dim=0).unsqueeze(0)
        for c in range(classes.shape[0])
    ]
    preds = torch.cat(errors_by_classes).argmin(dim=0) 
    return preds.to(data.device)
