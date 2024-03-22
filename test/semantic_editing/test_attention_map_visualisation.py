import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from semantic_editing.tools import attention_map_pca, attention_map_cluster, find_noun_indices, localise_nouns
from semantic_editing.utils import plot_image_on_axis


def test_visualise_attention_maps_pca(
    cfg_ddim,
    image_prompt_cat_and_dog,
    attention_store_accumulate,
):
    image, prompt = image_prompt_cat_and_dog
    cfg_ddim.fit(image, prompt)

    fig1, axes1 = plt.subplots(nrows=3, ncols=4)
    fig2, axes2 = plt.subplots(nrows=3, ncols=4)
    fig3, axes3 = plt.subplots(nrows=3, ncols=4)

    col_titles = ["Self-Attention", "Query", "Key", "Value"]
    element_names = ["attn", "query", "key", "value"]
    res_names = [16, 32, 64]
    for j, element_name in enumerate(element_names):
        for i, res in enumerate(res_names):
            attn_avg = attention_store_accumulate.aggregate_attention(
                places_in_unet=["up", "down", "mid"],
                is_cross=False,
                res=res,
                element_name=element_name,
            )
            proj = attention_map_pca(attn_avg, n_components=3, normalise=True)
            clusters_kmeans = attention_map_cluster(attn_avg, algorithm="kmeans", n_clusters=5)
            clusters_gmm = attention_map_cluster(attn_avg, algorithm="bgmm", n_clusters=10)
            proj_img = Image.fromarray((proj * 255).astype(np.uint8)) 
            proj_img = proj_img.resize((512, 512))
            title = col_titles[j] if i == 0 else None
            plot_image_on_axis(axes1[i][j], proj_img, title)
            plot_image_on_axis(axes2[i][j], clusters_kmeans, title)
            plot_image_on_axis(axes3[i][j], clusters_gmm, title)

    for ax1, res_name in zip(axes1[:, 0], res_names):
        kwargs = {"x": -100, "y": 400, "s": f"{res_name} x {res_name}", "rotation": 90}
        ax1.text(**kwargs)

    for ax2, ax3, res_name in zip(axes2[:, 0], axes3[:, 0], res_names):
        kwargs = {"s": f"{res_name} x {res_name}", "rotation": 90}
        ax2.text(**kwargs)
        ax3.text(**kwargs)

    fig1.savefig("ddim_inversion_attn_pca_proj.pdf", bbox_inches="tight")
    fig2.savefig("ddim_inversion_attn_kmeans.pdf", bbox_inches="tight")
    fig3.savefig("ddim_inversion_attn_bgmm.pdf", bbox_inches="tight")


def test_visualise_cross_attention_maps_nouns_clustering(
    cfg_ddim,
    image_prompt_cat_and_dog,
    attention_store_accumulate,
):
    image, prompt = image_prompt_cat_and_dog
    cfg_ddim.fit(image, prompt)

    noun_indices = find_noun_indices(cfg_ddim.model, prompt)

    self_attn_avg = attention_store_accumulate.aggregate_attention(
        places_in_unet=["up", "down", "mid"],
        is_cross=False,
        res=32,
        element_name="attn",
    )
    clusters = attention_map_cluster(self_attn_avg, "kmeans", n_clusters=10)
    cross_avg = attention_store_accumulate.aggregate_attention(
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
    attention_store_accumulate,
):
    image, prompt = image_prompt_cat_and_dog
    cfg_ddim.fit(image, prompt)

    noun_indices = find_noun_indices(cfg_ddim.model, prompt)

    resolutions = [16, 32, 64]
    fig, axes = plt.subplots(nrows=len(resolutions), ncols=len(noun_indices))
    for i, res in enumerate(resolutions):
        attn_avg = attention_store_accumulate.aggregate_attention(
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

