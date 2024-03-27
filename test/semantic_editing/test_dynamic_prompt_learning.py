import numpy as np
import matplotlib.pyplot as plt
import pytest
import torch
from PIL import Image
from sklearn.decomposition import PCA

from semantic_editing.tools import attention_map_pca, attention_map_cluster, attention_map_upsample, find_noun_indices, localise_nouns, normalise_image, stable_diffusion_tokens
from semantic_editing.utils import plot_image_on_axis, save_figure, seed_everything


def _generate(gen, image, prompt, file_name):
    attn_maps = gen.fit(image, prompt)
    image = gen.generate(prompt)
    image.save(file_name)
    return attn_maps


def _visualise_ca_maps(gen, cross_attn_avg: torch.FloatTensor, prompt: str, file_name: str):
    tokens = stable_diffusion_tokens(gen.model, prompt, include_separators=True)
    n_tokens = len(tokens)
    attn_res = gen.attention_resolution

    fig, axes = plt.subplots(nrows=1, ncols=n_tokens, figsize=(15, 5))
    # shape = (res, res, n_tokens)
    cross_attn_avgs = cross_attn_avg[:, :, :n_tokens].cpu().detach()
    # shape = (size, size, n_tokens)
    cross_attn_avgs_upsampled = attention_map_upsample(cross_attn_avgs, attn_res ** 2, "bilinear")

    for i, token in enumerate(tokens):
        # shape = (size, size, 1)
        attn_map = cross_attn_avgs_upsampled[:, :, i].unsqueeze(-1).numpy()
        assert attn_map.shape == (attn_res ** 2, attn_res ** 2, 1)
        norm_attn_map = normalise_image(attn_map)
        plot_image_on_axis(axes[i], norm_attn_map, token, fontsize=10)
    save_figure(fig, file_name)


def _test(model, image_and_prompt, recon_name, cross_attn_name):
    image, prompt = image_and_prompt
    attn_maps = _generate(model, image, prompt, recon_name)
    attn_maps_avg = torch.cat([attn_map.unsqueeze(0) for attn_map in attn_maps], dim=0).mean(0)
    _visualise_ca_maps(model, attn_maps_avg, prompt, cross_attn_name)


def test_dpl_cross_attn_visualisation(
    dpl_3,
    image_prompt_cat_and_dog,
    jet_cmap,
):
    _test(dpl_3, image_prompt_cat_and_dog, "dpl_reconstruction.pdf", "dpl_avg_cross_attention_maps.pdf")


def test_nti_cross_attn_visualisation(
    nti,
    image_prompt_cat_and_dog,
    jet_cmap,
):
    _test(nti, image_prompt_cat_and_dog, "nti_reconstruction.pdf", "nti_avg_cross_attention_maps.pdf")


def test_dpl_losses_cross_attn_visualisation(
    nti, dpl_1, dpl_2, dpl_3, image_prompt_cat_and_dog, jet_cmap, seed,
):
    models = [nti, dpl_1, dpl_2, dpl_3]
    image, prompt = image_prompt_cat_and_dog

    assert all(dpl.model == models[0].model for dpl in models)

    # Get tokens
    tokens = stable_diffusion_tokens(dpl_1.model, prompt, include_separators=True)
    n_tokens = len(tokens)

    fig, axes = plt.subplots(nrows=len(models), ncols=n_tokens, figsize=(15, 5))
    for i, dpl in enumerate(models):
        seed_everything(seed)
        attn_res = dpl.attention_resolution
        # Fit model and get attention maps
        cross_attn_avgs = dpl.fit(image, prompt)
        cross_attn_avgs = torch.cat([attn_map.unsqueeze(0) for attn_map in cross_attn_avgs], dim=0).mean(0)
        # shape = (res, res, n_tokens)
        cross_attn_avgs = cross_attn_avgs[:, :, :n_tokens].cpu().detach()
        # shape = (size, size, n_tokens)
        cross_attn_avgs_upsampled = attention_map_upsample(cross_attn_avgs, attn_res ** 2, "bilinear")
        for j, token in enumerate(tokens):
            # shape = (size, size, 1)
            attn_map = cross_attn_avgs_upsampled[:, :, j].unsqueeze(-1).numpy()
            norm_attn_map = normalise_image(attn_map)
            title = token if i == 0 else None
            plot_image_on_axis(axes[i][j], norm_attn_map, title, fontsize=10)
    save_figure(fig, "dpl_losses_cross_attention_comparison.pdf")

