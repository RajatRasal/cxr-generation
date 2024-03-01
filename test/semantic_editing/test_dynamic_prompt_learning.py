from PIL import Image

import numpy as np
from sklearn.decomposition import PCA


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
    attn_avg_flattened = attn_avg.reshape(resolution ** 2, -1).cpu().numpy()

    pca = PCA(n_components=3)
    proj = pca.fit_transform(attn_avg_flattened)
    proj = proj.reshape(resolution, resolution, 3)
    proj_min = proj.min(axis=(0, 1))
    proj_max = proj.max(axis=(0, 1))
    proj_normalised = (proj - proj_min) / (proj_max - proj_min)

    proj_img = Image.fromarray((proj_normalised * 255).astype(np.uint8))
    proj_img = proj_img.resize((512, 512))
    proj_img.save("ddim_inversion_avg_cross_attention_proj.pdf")

