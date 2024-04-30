import os
import pytest
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pytest
import torch
from PIL import Image
from sklearn.decomposition import PCA

from semantic_editing.dynamic_prompt_learning import DynamicPromptOptimisation
from semantic_editing.tools import attention_map_upsample, normalise_image, stable_diffusion_tokens
from semantic_editing.utils import plot_image_on_axis, save_figure, seed_everything, device_availability


def _fit(gen, image, prompt, tokens, index_noun_pairs, weights_location, seed):
    attn_maps = gen.fit(image, prompt, tokens=tokens, index_noun_pairs=index_noun_pairs, seed=seed)
    gen.save(weights_location)
    return attn_maps


def _edit(gen, swaps, weights, name, fig_dir, local, seed, steps):
    edit = gen.generate(
        swaps=swaps,
        weights=weights,
        cross_replace_steps=steps["cross_replace_steps"],
        self_replace_steps=steps["self_replace_steps"],
        local=local,
        seed=seed,
    )
    swaps_str = "_".join([f"{k}_{v}" for k, v in swaps.items()]) + \
        "_" + \
        "_".join([f"{k}_{v}" for k, v in weights.items()])
    edit.save(os.path.join(fig_dir, f"{name}_edit_{swaps_str}.pdf"))


def _visualise_ca_maps(gen, cross_attn_avg: torch.FloatTensor, tokens: List[str], file_name: str):
    tokens = [gen.model.tokenizer.bos_token] + tokens + [gen.model.tokenizer.eos_token]
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


def _test(model, image_and_prompt, cross_attn_name, weight_location_name, seed, to_cpu=True):
    image, prompt, tokens, index_noun_pairs = image_and_prompt
    attn_maps = _fit(model, image, prompt, tokens, index_noun_pairs, weight_location_name, seed)
    attn_maps_avg = torch.cat([attn_map.unsqueeze(0) for attn_map in attn_maps], dim=0).mean(0)
    _visualise_ca_maps(model, attn_maps_avg, tokens, cross_attn_name)
    # Need to move the model to cpu so that the downstream loading + editing
    # tests have enough GPU memory available to reload the model.
    # TODO: Write a method that sets the device of the SDAdapter
    if to_cpu:
        model.model.model.to("cpu")


@pytest.mark.dependency(name="dpl_3")
def test_dpl_cross_attn_visualisation(
    dpl_3,
    image_prompt_cat_and_dog,
    jet_cmap,
    fig_dir,
    weights_dir,
    seed,
):
    cross_attn_name = os.path.join(fig_dir, "dpl_3_avg_cross_attention_maps.pdf")
    dpl_3_weights_dir = os.path.join(weights_dir, "dpl_3")
    _test(dpl_3, image_prompt_cat_and_dog, cross_attn_name, dpl_3_weights_dir, seed)


@pytest.mark.dependency(name="dpl_recon", depends=["dpl_3"])
def test_dpl_reconstruction(fig_dir, weights_dir):
    name = "dpl_3"
    dpl_3_weights_dir = os.path.join(weights_dir, name)
    model = DynamicPromptOptimisation.load(dpl_3_weights_dir, device_availability())
    recon = model.generate()
    fig_dir_dpl = os.path.join(fig_dir, f"{name}_edit")
    os.makedirs(fig_dir_dpl, exist_ok=True)
    recon.save(os.path.join(fig_dir_dpl, "dpl_3_reconstruction.pdf"))
    # TODO: Add an MSE threshold


@pytest.mark.dependency(name="dpl_replace", depends=["dpl_3"])
def test_dpl_editing(fig_dir, weights_dir, seed, steps):
    name = "dpl_3"
    dpl_3_weights_dir = os.path.join(weights_dir, name)
    model = DynamicPromptOptimisation.load(dpl_3_weights_dir, device_availability())
    local = False
    fig_dir_dpl = os.path.join(fig_dir, f"{name}_edit")
    os.makedirs(fig_dir_dpl, exist_ok=True)
    _edit(model, {"dog": "lion"}, {}, name, fig_dir_dpl, local, seed, steps)
    _edit(model, {"cat": "wolf"}, {}, name, fig_dir_dpl, local, seed, steps)
    _edit(model, {"dog": "lion", "cat": "zebra"}, {}, name, fig_dir_dpl, local, seed, steps)


@pytest.mark.dependency(name="nti")
def test_nti_cross_attn_visualisation(
    nti,
    image_prompt_cat_and_dog,
    jet_cmap,
    fig_dir,
    weights_dir,
    seed,
):
    cross_attn_name = os.path.join(fig_dir, "nti_avg_cross_attention_maps.pdf")
    nti_weights_dir = os.path.join(weights_dir, "nti")
    _test(nti, image_prompt_cat_and_dog, cross_attn_name, nti_weights_dir, seed)


@pytest.mark.dependency(name="nti_recon", depends=["nti"])
def test_nti_reconstruction(fig_dir, weights_dir):
    name = "nti"
    nti_weights_dir = os.path.join(weights_dir, name)
    model = DynamicPromptOptimisation.load(nti_weights_dir, device_availability())
    recon = model.generate()
    fig_dir_nti = os.path.join(fig_dir, f"{name}_edit")
    os.makedirs(fig_dir_nti, exist_ok=True)
    recon.save(os.path.join(fig_dir_nti, "nti_reconstruction.pdf"))
    # TODO: Add an MSE threshold


@pytest.mark.dependency(name="nti_editing", depends=["nti"])
def test_nti_editing(fig_dir, weights_dir, seed, steps):
    name = "nti"
    nti_weights_dir = os.path.join(weights_dir, name)
    model = DynamicPromptOptimisation.load(nti_weights_dir, device_availability())
    local = False
    fig_dir_nti = os.path.join(fig_dir, f"{name}_edit")
    os.makedirs(fig_dir_nti, exist_ok=True)
    _edit(model, {"dog": "lion"}, {}, name, fig_dir_nti, local, seed, steps)
    _edit(model, {"cat": "wolf"}, {}, name, fig_dir_nti, local, seed, steps)
    _edit(model, {"dog": "lion", "cat": "zebra"}, {}, name, fig_dir_nti, local, seed, steps)


@pytest.mark.dependency(depends=["nti_editing", "dpl_editing"])
@pytest.mark.slow
def test_dpl_losses_cross_attn_visualisation(
    nti, dpl_1, dpl_2, dpl_3, image_prompt_cat_and_dog, jet_cmap, seed, fig_dir,
):
    models = [nti, dpl_1, dpl_2, dpl_3]
    image, prompt, _, _ = image_prompt_cat_and_dog

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
    save_figure(fig, os.path.join(fig_dir, "dpl_losses_cross_attention_comparison.pdf"))


@pytest.mark.dependency(depends=["nti_editing", "dpl_editing"])
@pytest.mark.slow
def test_dpl_sd_versions_cross_attn_visualisation(
    dpl_3_14, dpl_3_15, dpl_3_20, dpl_3_21, image_prompt_cat_and_dog, jet_cmap, seed, fig_dir,
):
    models = [dpl_3_14, dpl_3_15, dpl_3_20, dpl_3_21]
    image, prompt, _, _ = image_prompt_cat_and_dog

    # Get tokens
    tokens = stable_diffusion_tokens(dpl_3_14.model, prompt, include_separators=True)
    n_tokens = len(tokens)

    fig, axes = plt.subplots(nrows=len(models), ncols=n_tokens, figsize=(15, 5))
    for i, dpl in enumerate(models):
        dpl.model.model.to("cuda")
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
        dpl.model.model.to("cpu")
    save_figure(fig, os.path.join(fig_dir, "dpl_sd_versions_cross_attention_comparison.pdf"))


def _editing_dirs(editing_dir, model_name, exp_name, makedirs=False):
    store_dir = os.path.join(editing_dir, exp_name)
    model_dir = os.path.join(store_dir, model_name)
    weights_dir = os.path.join(model_dir, "weights")
    images_dir = os.path.join(model_dir, "images")
    if makedirs:
        os.makedirs(store_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
    return store_dir, model_dir, weights_dir, images_dir


@pytest.mark.parametrize(
    "name", 
    [
        ("pear_and_apple"),
        ("sports_equipment"),
        ("horse_and_sheep"),
        ("book_clock_bottle"),
        ("football_on_bench"),
        ("cake_on_plate"),
        ("cat_bird_stitching"),
        ("cat_bird_painting"),
        ("cat_dog_watercolour"),
        ("cat_dog_flowers"),
    ]
)
@pytest.mark.dependency()
@pytest.mark.slow
def test_fitting_more_models(name, request, more_editing_dir, dpl_3, nti, seed):
    weights_dir = os.path.join(more_editing_dir, "weights")
    image_and_prompt = request.getfixturevalue(f"image_prompt_{name}")
    for model_name, model in [("nti", nti), ("dpl", dpl_3)]:
        store_dir, model_dir, weights_dir, images_dir = _editing_dirs(more_editing_dir, model_name, name, True)
        cross_attn_name = os.path.join(images_dir, "cross_attention_maps.pdf")
        _test(model, image_and_prompt, cross_attn_name, weights_dir, seed, False)


@pytest.mark.parametrize(
    "name, swaps_list", 
    [
        ("pear_and_apple", [{"pear": "watermelon"}, {"apple": "pineapple"}]),
        ("sports_equipment", [{"football": "balloon"}, {"basketball": "wheel"}]),
        ("horse_and_sheep", [{"horse": "donkey"}]),
        ("book_clock_bottle", [{"clock": "wheel"}]),
        ("cat_dog_flowers", [{"cat": "tiger"}, {"dog": "lion"}]),
    ]
)
@pytest.mark.dependency(depends=["test_fitting_more_models"])
@pytest.mark.slow
def test_editing_with_more_models(name, swaps_list, request, more_editing_dir, dpl_3, nti, steps, seed):
    local = False
    for model_name, model in [("nti", nti), ("dpl", dpl_3)]:
        for swaps in swaps_list:
            store_dir, model_dir, weights_dir, images_dir = _editing_dirs(more_editing_dir, model_name, name)
            model = DynamicPromptOptimisation.load(weights_dir, device_availability())
            _edit(model, swaps, {}, "", images_dir, local, seed, steps)

