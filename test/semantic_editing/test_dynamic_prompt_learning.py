import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA

from semantic_editing.tools import attention_map_pca, attention_map_cluster, find_noun_indices, localise_nouns, stable_diffusion_tokens
from semantic_editing.utils import plot_image_on_axis


def test_dynamic_prompt_losses(
    dpl,
    image_prompt_cat_and_dog,
    attention_store,
):
    image, prompt = image_prompt_cat_and_dog
    dpl.fit(*image_prompt_cat_and_dog)
    image = dpl.generate(prompt)
    image.save("dpl_reconstruction.pdf")

    cross_attn_avg = attention_store.aggregate_attention(
        places_in_unet=["up", "down", "mid"],
        is_cross=True,
        res=32,
        element_name="attn",
    )
    tokens = stable_diffusion_tokens(dpl.model, prompt)
    n_tokens = len(tokens)

    fig, axes = plt.subplots(nrows=1, ncols=n_tokens)
    for i, token in zip(range(1, n_tokens + 1), tokens):
        attn_map = cross_attn_avg[:, :, i].cpu().numpy()
        plot_image_on_axis(axes[i - 1], attn_map, token)

    fig.savefig("dpl_avg_cross_attention_maps.pdf")
    assert False

