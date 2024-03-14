import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.decomposition import PCA

from semantic_editing.tools import attention_map_pca, attention_map_cluster, attention_map_upsample, find_noun_indices, localise_nouns, normalise_image, stable_diffusion_tokens
from semantic_editing.utils import plot_image_on_axis, save_figure


def _generate(gen, image, prompt, file_name):
    attn_maps = gen.fit(image, prompt)
    image = gen.generate(prompt)
    image.save(file_name)
    return attn_maps


def _visualise_ca_maps(gen, cross_attn_avg: torch.FloatTensor, prompt: str, ca_res: int, file_name: str):
    # cross_attn_avg = attention_store.aggregate_attention(
    #     places_in_unet=["up", "down", "mid"],
    #     is_cross=True,
    #     res=ca_res,
    #     element_name="attn",
    # )
    tokens = stable_diffusion_tokens(gen.model, prompt, include_separators=True)
    n_tokens = len(tokens)

    fig, axes = plt.subplots(nrows=1, ncols=n_tokens, figsize=(15, 5))
    # shape = (res, res, n_tokens)
    cross_attn_avgs = cross_attn_avg[:, :, :n_tokens].cpu().detach()
    # shape = (size, size, n_tokens)
    cross_attn_avgs_upsampled = attention_map_upsample(cross_attn_avgs, ca_res ** 2, "bilinear")

    for i, token in enumerate(tokens):
        # shape = (size, size, 1)
        attn_map = cross_attn_avgs_upsampled[:, :, i].unsqueeze(-1).numpy()
        assert attn_map.shape == (ca_res ** 2, ca_res ** 2, 1)
        norm_attn_map = normalise_image(attn_map)
        plot_image_on_axis(axes[i], norm_attn_map, token, fontsize=10)
    save_figure(fig, file_name)


def test_dpl_generation_visualisation(
    dpl,
    image_prompt_cat_and_dog,
    attention_store,
):
    image, prompt = image_prompt_cat_and_dog
    attn_maps = _generate(dpl, image, prompt, "dpl_reconstruction.pdf")
    attn_maps_avg = torch.cat([attn_map.unsqueeze(0) for attn_map in attn_maps], dim=0).mean(0)
    _visualise_ca_maps(dpl, attn_maps_avg, prompt, 32, "dpl_avg_cross_attention_maps.pdf")
    assert False


def test_nti_generation_visualisation(
    nti,
    image_prompt_cat_and_dog,
    attention_store,
):
    image, prompt = image_prompt_cat_and_dog
    _generate(nti, image, prompt, "nti_reconstruction.pdf")
    _visualise_ca_maps(nti, attention_store, prompt, 32, "nti_avg_cross_attention_maps.pdf")
    assert False

