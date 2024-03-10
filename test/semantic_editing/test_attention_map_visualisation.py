import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA

from semantic_editing.tools import attention_map_pca, attention_map_cluster, find_noun_indices, localise_nouns
from semantic_editing.utils import plot_image_on_axis


def test_visualise_cross_attention_maps_pca(
    cfg_ddim,
    image_prompt_cat_and_dog,
    attention_store,
):
    image, prompt = image_prompt_cat_and_dog
    cfg_ddim.fit(image, prompt)
    cfg_ddim.generate(prompt)

    # TODO: Is this the denoising PCA or the inversion PCA?
    resolution = 32
    attn_avg = attention_store.aggregate_attention(
        places_in_unet=["up", "down", "mid"],
        is_cross=False,
        res=resolution,
        element_name="attn",
    )

    # TODO: Make separate unittests for pca and clustering scripts
    proj = attention_map_pca(attn_avg, n_components=3, normalise=True)
    proj_img = Image.fromarray((proj * 255).astype(np.uint8)) 
    proj_img = proj_img.resize((512, 512))
    proj_img.save("ddim_inversion_avg_cross_attention_proj.pdf")


def test_visualise_cross_attention_maps_clustering(
    cfg_ddim,
    image_prompt_cat_and_dog,
    attention_store,
):
    image, prompt = image_prompt_cat_and_dog
    cfg_ddim.fit(image, prompt)
    cfg_ddim.generate(prompt)

    # TODO: Is this the denoising PCA or the inversion PCA?
    attn_avg = attention_store.aggregate_attention(
        places_in_unet=["up", "down", "mid"],
        is_cross=True,
        res=16,
        element_name="attn",
    )

    clusters = attention_map_cluster(attn_avg, gmm=False, n_clusters=5)
    plt.imshow(clusters)
    plt.axis("off")
    plt.savefig("ddim_inversion_avg_cross_attention_clusters_kmeans.pdf", bbox_inches="tight", pad_inches=0)

    clusters = attention_map_cluster(attn_avg, gmm=True, n_clusters=10)
    plt.imshow(clusters)
    plt.axis("off")
    plt.savefig("ddim_inversion_avg_cross_attention_clusters_gmm.pdf", bbox_inches="tight", pad_inches=0)


def test_visualise_cross_attention_maps_nouns_clustering(
    sd_adapter_with_attn_excite,
    cfg_ddim,
    image_prompt_cat_and_dog,
    attention_store,
):
    image, prompt = image_prompt_cat_and_dog
    cfg_ddim.fit(image, prompt)
    cfg_ddim.generate(prompt)

    noun_indices = find_noun_indices(cfg_ddim.model, prompt)

    attn_avg = attention_store.aggregate_attention(
        places_in_unet=["up", "down", "mid"],
        is_cross=False,
        res=32,
        element_name="attn",
    )
    clusters = attention_map_cluster(attn_avg, gmm=False, n_clusters=5)

    cross_avg = attention_store.aggregate_attention(
        places_in_unet=["up", "down", "mid"],
        is_cross=True,
        res=16,
        element_name="attn",
    )
    masks = localise_nouns(clusters, cross_avg.cpu().numpy(), noun_indices)
    masks = {k: Image.fromarray((v * 255).astype(np.uint8)) for k, v in masks.items()}

    fig, axes = plt.subplots(nrows=1, ncols=len(masks), figsize=(15, 5))
    plot_image_on_axis(axes[0], masks["BG"], "Background")
    del masks["BG"]
    for i, (k, v) in enumerate(masks.items()):
        plot_image_on_axis(axes[i + 1], v, k.capitalize())
    fig.savefig("masks.pdf")

    assert False


def test_visualise_prompt_localisation(
    sd_adapter_with_attn_excite,
    cfg_ddim,
    image_prompt_cat_and_dog,
    attention_store,
):
    image, prompt = image_prompt_cat_and_dog
    cfg_ddim.fit(image, prompt)
    cfg_ddim.generate(prompt)

    noun_indices = find_noun_indices(cfg_ddim.model, prompt)
    print(noun_indices)

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

    fig.savefig("ca_maps.pdf")

    assert False

