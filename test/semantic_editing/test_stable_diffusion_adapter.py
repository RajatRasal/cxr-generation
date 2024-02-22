import pytest

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, DDIMScheduler

from semantic_editing.diffusion import StableDiffusionAdapter, ddim_inversion, classifier_free_guidance


def test_decode_encode_image(sd_adapter, image_prompt):
    original_image, _ = image_prompt
    latent = sd_adapter.encode_image(original_image)
    decoded_image = sd_adapter.decode_latent(latent)
    diff = np.abs(np.array(original_image) - np.array(decoded_image))
    median_err = np.median(diff)
    assert median_err == 4


def test_next_step(sd_adapter, image_prompt):
    assert False


def test_prev_step(sd_adapter, image_prompt):
    assert False


def test_get_noise_pred(sd_adapter, image_prompt):
    image, prompt = image_prompt
    prompt_embedding = sd_adapter.encode_text(prompt)
    latent = sd_adapter.encode_image(image)
    with torch.no_grad():
        noise_at_0 = sd_adapter.get_noise_pred(latent, 0, prompt_embedding)
        latent_at_0 = sd_adapter.next_step(noise_at_0, 0, latent)
        noise_at_1 = sd_adapter.get_noise_pred(latent, 100, prompt_embedding)
        latent_at_1 = sd_adapter.next_step(noise_at_1, 100, latent)
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].imshow(sd_adapter.decode_latent(noise_at_0))
    axs[1].imshow(sd_adapter.decode_latent(noise_at_1))
    fig.savefig("test_get_noise_pred.pdf", bbox_inches="tight")


def test_ddim_inversion(sd_adapter, image_prompt):
    idxs = [
        0,
        int(sd_adapter.ddim_steps / 5),
        int(sd_adapter.ddim_steps * 2/5),
        int(sd_adapter.ddim_steps * 3/5),
        -1,
    ]

    def decode_and_plot_latent(ax, latent, title = ""):
        ax.set_axis_off()
        ax.imshow(sd_adapter.decode_latent(latent))
        ax.set_title(title)

    latents = ddim_inversion(sd_adapter, *image_prompt)
    latent_T = latents[-1].clone()
    assert len(latents) == sd_adapter.ddim_steps + 1

    fig, axs = plt.subplots(nrows=1, ncols=len(idxs), figsize=(50, 10))
    for i in range(len(idxs)):
        decode_and_plot_latent(axs[i], latents[idxs[i]])
    fig.savefig("test_ddim_inversion.pdf", bbox_inches="tight")

    def cfg_with_guidance(guidance_scale):
        latents = classifier_free_guidance(sd_adapter, latent_T, image_prompt[1], guidance_scale)
        assert len(latents) == sd_adapter.ddim_steps + 1

        fig, axs = plt.subplots(nrows=1, ncols=len(idxs), figsize=(50, 10))
        for i in range(len(idxs)):
            decode_and_plot_latent(axs[i], latents[idxs[i]])
        fig.savefig(
            f"test_classifier_free_guidance_gs_{str(guidance_scale).replace('.', '')}.pdf",
            bbox_inches="tight",
        )

        return sd_adapter.decode_latent(latents[-1])

    gen_no_guidance = cfg_with_guidance(1)
    gen_medium_guidance = cfg_with_guidance(4)
    gen_high_guidance = cfg_with_guidance(7.5)

    assert True
