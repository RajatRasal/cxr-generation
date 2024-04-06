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
    cfg_ddim,
    image_prompt_cat_and_dog,
    seed,
    image_size,
    background_map_hps,
):
    image, prompt = image_prompt_cat_and_dog
    cfg_ddim.fit(image, prompt)
    attention_store_accumulate = cfg_ddim.model.get_attention_store()

    noun_indices = find_noun_indices(
        cfg_ddim.model,
        prompt,
    )
    bg_map = background_mask(
        attention_store_accumulate,
        noun_indices,
        background_threshold=background_map_hps["background_threshold"],
        algorithm=background_map_hps["algorithm"],
        n_clusters=background_map_hps["n_clusters"],
        cluster_random_state=seed,
        attention_resolution=16,
        upscale_size=image_size,
    )

	# TODO: Figure out a location for this tensor file
    target_bg_map = torch.load("test/semantic_editing/bg_maps_tensor")

    print("Predicted map:\n", bg_map)
    print("Target map:\n", target_bg_map)
    assert torch.equal(bg_map, target_bg_map) or (~(bg_map == target_bg_map).bool()).sum() <= 3

