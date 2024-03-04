import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

from semantic_editing.tools import attention_map_pca, attention_map_cluster


def test_visualise_cross_attention_maps(
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

    seed = 42

    # TODO: Make separate unittests for pca and clustering scripts
    proj = attention_map_pca(attn_avg, n_components=3, normalise=True, seed=seed)
    proj_img = Image.fromarray((proj * 255).astype(np.uint8)) 
    proj_img = proj_img.resize((512, 512))
    proj_img.save("ddim_inversion_avg_cross_attention_proj.pdf")

    clusters = attention_map_cluster(attn_avg, n_clusters=5, seed=seed)
    plt.imshow(clusters)
    plt.axis("off")
    plt.savefig("ddim_inversion_avg_cross_attention_clusters.pdf", bbox_inches="tight", pad_inches=0)

