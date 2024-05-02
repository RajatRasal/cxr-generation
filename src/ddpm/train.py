import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from tqdm import tqdm

from ddpm.ddpm import GaussianDiffusion
from ddpm.unet import Unet


def main():
    # TODO: Copy in the MNIST binaries into this repo
    # Hyperparamers
    batch_size = 512
    train_timesteps = 200
    sampling_timesteps = 50
    cond_prob_drop = 0.6
    latent_dim = 128
    dataset_cache = "/data2/mnist"
    dim_mults = (1, 1, 2)
    epochs = 300
    guidance_scale = 7.5
    lr = 1e-4

    # Load dataset
    preprocess = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(root=dataset_cache, download=True, train=True, transform=preprocess)
    num_classes = len(train_dataset.classes)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    channels = train_dataset[0][0].shape[0]
    image_size = train_dataset[0][0].shape[1]

    # Load model
    model = Unet(
        dim=latent_dim,
        dim_mults=dim_mults,
        num_classes=num_classes,
        cond_drop_prob=cond_prob_drop,
        channels=channels,
    )
    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=train_timesteps,
        sampling_timesteps=sampling_timesteps,
    ).cuda()

    # Optimiser
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=lr)

    # Train model
    for i in range(epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch: {i} | Loss: N/A")
        for j, (images, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = diffusion(images.cuda(), classes=labels.cuda())
            loss.backward()
            optimizer.step()
            if j % 10 == 0:
                progress_bar.update(10)
                progress_bar.set_description(f"Epoch: {i} | Loss: {round(loss.item(), 5)}")
        # Sample images from model
        sampled_images = diffusion.sample(
            classes=torch.arange(num_classes).cuda(),
            cond_scale=guidance_scale,
        )
        # Save images
        grid = to_pil_image(make_grid(sampled_images, nrow=num_classes))
        grid.save(f"{i}_grid.png")

    # TODO: Interpolation to show density of latent space
    # # Interpolation
    # interpolate_out = diffusion.interpolate(
    #     training_images[:1],
    #     training_images[:1],
    #     image_classes[:1]
    # )

