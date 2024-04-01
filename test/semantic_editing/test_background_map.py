import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as F_vision
from PIL import Image

from semantic_editing.diffusion import ddim_inversion
from semantic_editing.tools import background_mask, find_noun_indices
from semantic_editing.utils import plot_image_on_axis, save_figure


def test_background_map(
    sd_adapter_with_attn_accumulate,
    image_prompt_cat_and_dog,
    seed,
):
    image_size = 512
    image, prompt = image_prompt_cat_and_dog
    image = image.resize((image_size, image_size))
    _ = ddim_inversion(
        sd_adapter_with_attn_accumulate,
        image,
        prompt,
    )

    noun_indices = find_noun_indices(
        sd_adapter_with_attn_accumulate,
        prompt,
    )
    bg_map = background_mask(
        sd_adapter_with_attn_accumulate.attention_store,
        noun_indices,
        background_threshold=0.3,
        algorithm="kmeans",
        n_clusters=5,
        cluster_random_state=seed,
        attention_resolution=16,
        upscale_size=image_size,
    )

    target_bg_map = torch.load("test/semantic_editing/bg_maps_tensor")
    print(bg_map)
    print(target_bg_map)
    assert torch.equal(bg_map, target_bg_map) or (~(bg_map == target_bg_map).bool()).sum() <= 3

