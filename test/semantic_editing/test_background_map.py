import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as F_vision
from PIL import Image
from collections import Counter

from semantic_editing.attention import AttentionStoreAccumulate, AttnProcessorWithAttentionStore
from semantic_editing.diffusion import ddim_inversion
from semantic_editing.tools import background_mask, attention_map_cluster, find_tokens_and_noun_indices
from semantic_editing.utils import plot_image_on_axis, save_figure


def test_find_masks(
    sd_adapter_fixture,
    image_prompt_cat_and_dog,
    seed,
    image_size,
    background_map_hps,
):
    # Setup
    image, prompt, _, _ = image_prompt_cat_and_dog
    sd_adapter_fixture.register_attention_store(
        AttentionStoreAccumulate(),
        AttnProcessorWithAttentionStore,
    )
    attention_store_accumulate = sd_adapter_fixture.get_attention_store()
    ddim_inversion(
        sd_adapter_fixture,
        image.convert("RGB").resize((image_size, image_size)),
        prompt,
        torch.manual_seed(seed),
    )

    # Calculate average self-attention maps from DDIM inversion
    self_attn_map = attention_store_accumulate.aggregate_attention(
        places_in_unet=["up", "down", "mid"],
        is_cross=False,
        res=32,
        element_name="attn",
    )

    # Run clusters
    clusters = attention_map_cluster(
        self_attn_map,
        algorithm=background_map_hps["algorithm"],
        n_clusters=background_map_hps["n_clusters"],
        **background_map_hps["kwargs"],
    )

    # Verify the distribution of clusters compared to reference implementation
    assert {**Counter(clusters.flatten().tolist())} == {3: 232, 1: 216, 2: 212, 0: 193, 4: 171}


def test_background_map(
    sd_adapter_fixture,
    image_prompt_cat_and_dog,
    seed,
    image_size,
    background_map_hps,
):
    # Setup
    image, prompt, _, index_noun_pairs = image_prompt_cat_and_dog
    sd_adapter_fixture.register_attention_store(
        AttentionStoreAccumulate(),
        AttnProcessorWithAttentionStore,
    )
    attention_store_accumulate = sd_adapter_fixture.get_attention_store()
    ddim_inversion(
        sd_adapter_fixture,
        image.convert("RGB").resize((image_size, image_size)),
        prompt,
        torch.manual_seed(seed),
    )

    # Calculate background mask
    bg_map = background_mask(
        attention_store_accumulate,
        index_noun_pairs,
        background_threshold=background_map_hps["background_threshold"],
        algorithm=background_map_hps["algorithm"],
        n_clusters=background_map_hps["n_clusters"],
        attention_resolution=16,
        upscale_size=image_size,
        **background_map_hps["kwargs"],
    )

    # TODO: Use bg_map from reference implementation
    assert bg_map.sum() == 188

