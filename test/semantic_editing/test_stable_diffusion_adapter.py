import os
import pytest

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, DDIMScheduler

from semantic_editing.diffusion import StableDiffusionAdapter, ddim_inversion, classifier_free_guidance
from semantic_editing.utils import plot_image_on_axis, save_figure


def test_decode_encode_image(sd_adapter_fixture, image_prompt):
    original_image, _ = image_prompt
    latent = sd_adapter_fixture.encode_image(original_image)
    decoded_image = sd_adapter_fixture.decode_latent(latent)
    diff = np.abs(np.array(original_image) - np.array(decoded_image))
    median_err = np.median(diff)
    assert median_err == 4


def test_get_noise_pred(sd_adapter_fixture, image_prompt, fig_dir):
    image, prompt = image_prompt
    prompt_embedding = sd_adapter_fixture.encode_text(prompt)
    latent = sd_adapter_fixture.encode_image(image)
    with torch.no_grad():
        noise_at_0 = sd_adapter_fixture.get_noise_pred(latent, 0, prompt_embedding)
        latent_at_0 = sd_adapter_fixture.next_step(noise_at_0, 0, latent)
        noise_at_1 = sd_adapter_fixture.get_noise_pred(latent, 100, prompt_embedding)
        latent_at_1 = sd_adapter_fixture.next_step(noise_at_1, 100, latent)
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    plot_image_on_axis(axs[0], sd_adapter_fixture.decode_latent(noise_at_0))
    plot_image_on_axis(axs[1], sd_adapter_fixture.decode_latent(noise_at_1))
    save_figure(fig, os.path.join(fig_dir, "test_get_noise_pred.pdf"))


def test_ddim_inversion(sd_adapter_fixture, image_prompt, fig_dir):
    # TODO: This is not working as expected! We are not seeing the varying
    # steps in the denoising process
    idxs = [
        0,
        int(sd_adapter_fixture.ddim_steps / 5),
        int(sd_adapter_fixture.ddim_steps * 2/5),
        int(sd_adapter_fixture.ddim_steps * 3/5),
        -1,
    ]

    def decode_and_plot_latent(ax, latent, title = ""):
        ax.set_axis_off()
        ax.imshow(sd_adapter_fixture.decode_latent(latent))
        ax.set_title(title)

    n_timesteps = len(sd_adapter_fixture.get_timesteps())
    latents = ddim_inversion(sd_adapter_fixture, *image_prompt)
    latent_T = latents[-1].clone()
    n_latents = len(latents)
    assert n_latents == sd_adapter_fixture.ddim_steps + 1 == n_timesteps + 1 

    fig, axs = plt.subplots(nrows=1, ncols=len(idxs), figsize=(50, 10))
    for i in range(len(idxs)):
        decode_and_plot_latent(axs[i], latents[idxs[i]])
    save_figure(fig, os.path.join(fig_dir, "test_ddim_inversion.pdf"))

    def cfg_with_guidance(guidance_scale):
        latents = classifier_free_guidance(sd_adapter_fixture, latent_T, image_prompt[1], guidance_scale)
        assert len(latents) == sd_adapter_fixture.ddim_steps + 1

        fig, axs = plt.subplots(nrows=1, ncols=len(idxs), figsize=(50, 10))
        for i in range(len(idxs)):
            decode_and_plot_latent(axs[i], latents[idxs[i]])
        fig_name = f"test_classifier_free_guidance_gs_{str(guidance_scale).replace('.', '')}.pdf"
        save_figure(fig, os.path.join(fig_dir, fig_name))

        return sd_adapter_fixture.decode_latent(latents[-1])

    gen_no_guidance = cfg_with_guidance(1)
    gen_medium_guidance = cfg_with_guidance(4)
    gen_high_guidance = cfg_with_guidance(7.5)


def test_tokenisation(sd_adapter_fixture, image_prompt):
    image, prompt = image_prompt
    words = [word.lower() for word in prompt.split(" ")]
    n_words = len(words)
    tokens = sd_adapter_fixture.tokenise_text(prompt, string=True)
    assert tokens[0] == "<|startoftext|>"
    assert all([
        token.endswith("</w>") and token.startswith(word)
        for token, word in zip(tokens[1:n_words + 1], words)
    ])
    assert all([token == "<|endoftext|>" for token in tokens[n_words + 1:]])
    assert len(tokens) == sd_adapter_fixture.tokenizer.model_max_length

