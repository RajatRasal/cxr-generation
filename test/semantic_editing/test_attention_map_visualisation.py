import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from semantic_editing.tools import attention_map_pca, attention_map_cluster, find_masks, find_noun_indices, localise_nouns
from semantic_editing.utils import plot_image_on_axis, save_figure


def test_visualise_attention_maps_pca(
    cfg_ddim,
    image_prompt_cat_and_dog,
    attention_store_accumulate,
    jet_cmap,
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
        kwargs = {"x": -100, "y": 400, "s": f"{res_name} x {res_name}", "rotation": 90}
        ax2.text(**kwargs)
        ax3.text(**kwargs)

    save_figure(fig1, "ddim_inversion_attn_pca_proj.pdf")
    save_figure(fig2, "ddim_inversion_attn_kmeans.pdf")
    save_figure(fig3, "ddim_inversion_attn_bgmm.pdf")


def test_visualise_cross_attention_maps_nouns_clustering(
    cfg_ddim,
    image_prompt_cat_and_dog,
    attention_store_accumulate,
):
    # TODO: Make this test display figure 4 from the paper
    image, prompt = image_prompt_cat_and_dog
    cfg_ddim.fit(image, prompt)

    # Self-Attention
    self_attn_avg = attention_store_accumulate.aggregate_attention(
        places_in_unet=["up", "down", "mid"],
        is_cross=False,
        res=32,
        element_name="attn",
    )
    self_attn_avg_proj = attention_map_pca(self_attn_avg, n_components=3, normalise=True)
    self_attn_avg_clusters = attention_map_cluster(self_attn_avg, algorithm="kmeans", n_clusters=5)

    # Cross-Attention
    cross_attn_avg = attention_store_accumulate.aggregate_attention(
        places_in_unet=["up", "down", "mid"],
        is_cross=True,
        res=16,
        element_name="attn",
    )

    # Object masks
    noun_indices = find_noun_indices(cfg_ddim.model, prompt)
    masks = find_masks(
        attention_store_accumulate,
        noun_indices,
    )
    masks = {k: Image.fromarray((v.cpu().numpy() * 255).astype(np.uint8)) for k, v in masks.items()}

    # (original, self-attn, self-attn cluster, cross-attn cat, cross-attn dog, background, foreground cat, foreground dog)
    fig, axes = plt.subplots(nrows=1, ncols=8, figsize=(15, 5))

    plot_image_on_axis(axes[0], image, "Original")
    plot_image_on_axis(axes[1], self_attn_avg_proj, "Self-Attn")
    plot_image_on_axis(axes[2], self_attn_avg_clusters, "Self-Attn Clusters")

    noun_base_index = 3
    for i, (noun_index, noun) in enumerate(noun_indices):
        plot_image_on_axis(axes[noun_base_index + i], cross_attn_avg[:, :, noun_index].cpu().numpy(), f"Cross-Attn: {noun.capitalize()}")

    background_index = 5
    plot_image_on_axis(axes[background_index], masks["BG"], "Background")
    del masks["BG"]
    for i, (k, v) in enumerate(masks.items()):
        plot_image_on_axis(axes[background_index + i + 1], v, f"Foreground: {k.capitalize()}")

    save_figure(fig, "ddim_inversion_attn_maps.pdf")

