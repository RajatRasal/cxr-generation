import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA

from semantic_editing.tools import attention_map_pca, attention_map_cluster, find_noun_indices, localise_nouns
from semantic_editing.utils import plot_image_on_axis


def test_visualise_self_attention_maps_pca(
    cfg_ddim,
    image_prompt_cat_and_dog,
    attention_store,
):
    image, prompt = image_prompt_cat_and_dog
    cfg_ddim.fit(image, prompt)

    fig, axes = plt.subplots(nrows=3, ncols=1)

    # PCA for attention maps from inversion
    for i, res in enumerate([16, 32, 64]):
        attn_avg = attention_store.aggregate_attention(
            places_in_unet=["up", "down", "mid"],
            is_cross=False,
            res=res,
            element_name="attn",
        )
        proj = attention_map_pca(attn_avg, n_components=3, normalise=True)
        proj_img = Image.fromarray((proj * 255).astype(np.uint8)) 
        proj_img = proj_img.resize((512, 512))
        plot_image_on_axis(axes[i], proj_img, f"{res} x {res}")
    fig.savefig("ddim_inversion_avg_self_attention_proj.pdf")

    assert False


def test_visualise_self_attention_maps_clustering(
    cfg_ddim,
    image_prompt_cat_and_dog,
    attention_store,
):
    image, prompt = image_prompt_cat_and_dog
    cfg_ddim.fit(image, prompt)

    def _plot(alg: str, n_clusters: int, name: str):
        fig, axes = plt.subplots(nrows=3, ncols=1)
        for i, res in enumerate([16, 32, 64]):
            attn_avg = attention_store.aggregate_attention(
                places_in_unet=["up", "down", "mid"],
                is_cross=False,
                res=res,
                element_name="attn",
            )
            clusters = attention_map_cluster(attn_avg, algorithm=alg, n_clusters=n_clusters)
            plot_image_on_axis(axes[i], clusters, f"{res} x {res}")
        fig.savefig(name)

    for alg in ["kmeans", "gmm", "bgmm"]:
        _plot(alg, 10, f"ddim_inversion_avg_self_attention_clusters_{alg}.pdf")


def test_visualise_cross_attention_maps_nouns_clustering(
    cfg_ddim,
    image_prompt_cat_and_dog,
    attention_store,
):
    image, prompt = image_prompt_cat_and_dog
    cfg_ddim.fit(image, prompt)

    noun_indices = find_noun_indices(cfg_ddim.model, prompt)

    self_attn_avg = attention_store.aggregate_attention(
        places_in_unet=["up", "down", "mid"],
        is_cross=False,
        res=32,
        element_name="attn",
    )
    clusters = attention_map_cluster(self_attn_avg, "kmeans", n_clusters=10)
    cross_avg = attention_store.aggregate_attention(
        places_in_unet=["up", "down", "mid"],
        is_cross=True,
        res=16,
        element_name="attn",
    )
    masks = localise_nouns(clusters, cross_avg.cpu().numpy(), noun_indices, 0.3)
    masks = {k: Image.fromarray((v * 255).astype(np.uint8)) for k, v in masks.items()}

    fig, axes = plt.subplots(nrows=1, ncols=len(masks), figsize=(15, 5))
    plot_image_on_axis(axes[0], masks["BG"], "Background")
    del masks["BG"]
    for i, (k, v) in enumerate(masks.items()):
        plot_image_on_axis(axes[i + 1], v, k.capitalize())
    fig.savefig("ddim_inversion_object_masks.pdf")


def test_visualise_prompt_localisation(
    sd_adapter_with_attn_excite,
    cfg_ddim,
    image_prompt_cat_and_dog,
    attention_store,
):
    image, prompt = image_prompt_cat_and_dog
    cfg_ddim.fit(image, prompt)

    noun_indices = find_noun_indices(cfg_ddim.model, prompt)

    resolutions = [16, 32, 64]
    fig, axes = plt.subplots(nrows=len(resolutions), ncols=len(noun_indices))
    for i, res in enumerate(resolutions):
        attn_avg = attention_store.aggregate_attention(
            places_in_unet=["up", "down", "mid"],
            is_cross=True,
            res=res,
            element_name="attn",
        )
        for j, (index, noun) in enumerate(noun_indices):
            attn_map = attn_avg[:, :, index].cpu().numpy()
            title = noun if i == 0 else ""
            plot_image_on_axis(axes[i, j], attn_map, title)

    fig.savefig("ddim_inversion_avg_cross_attention_maps_nouns.pdf")

