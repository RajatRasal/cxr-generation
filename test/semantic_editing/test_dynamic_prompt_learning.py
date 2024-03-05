import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

from semantic_editing.tools import attention_map_pca, attention_map_cluster, find_noun_indices, localise_nouns


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
    clusters = attention_map_cluster(attn_avg, gmm=False, n_clusters=6)

    cross_avg = attention_store.aggregate_attention(
        places_in_unet=["up", "down", "mid"],
        is_cross=True,
        res=16,
        element_name="attn",
    )
    interpretation, masks = localise_nouns(clusters, cross_avg.cpu().numpy(), noun_indices)
    print(interpretation)

    assert False

