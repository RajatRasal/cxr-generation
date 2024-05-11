from typing import Dict, List, Literal, Tuple

import nltk
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as F_vision
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from semantic_editing.attention import AttentionStore
from semantic_editing.diffusion import PretrainedStableDiffusionAdapter
from semantic_editing.validate import _validate_attn_map


CLUSTERING_ALGORITHM = Literal["kmeans", "gmm", "bgmm"]

def normalise_image(image: np.ndarray) -> np.ndarray:
    shape = image.shape
    assert len(shape) == 3 and shape[0] == shape[1]

    image_min = image.min(axis=(0, 1))
    image_max = image.max(axis=(0, 1))
    image_normalised = (image - image_min) / (image_max - image_min)

    return image_normalised


def attention_map_upsample(
    attn_map: torch.FloatTensor,
    size: int = 512,
    mode: Literal["bilinear"] = "bilinear",
):
    # TODO: Write unittest for this method
    # shape = (res, res, n_tokens)
    res, features = _validate_attn_map(attn_map)
    # shape = (n_tokens, res, res)
    attn_map = attn_map.permute(2, 0, 1)
    # shape = (n_tokens, 1, res, res)
    attn_map = attn_map.unsqueeze(1)
    # shape = (n_tokens, 1, size, size)
    attn_map = F.interpolate(attn_map, size=size, mode=mode)
    # shape = (n_tokens, size, size)
    attn_map = attn_map.squeeze(1)
    # shape = (size, size, n_tokens)
    attn_map = attn_map.permute(1, 2, 0)
    return attn_map


def attention_map_pca(
    attn_map: torch.FloatTensor,
    n_components: int = 3,
    normalise: bool = True,
    seed: int = 42,
    **pca_kwargs,
) -> np.ndarray:
    res, features = _validate_attn_map(attn_map)

    attn_map_flat = attn_map.cpu().numpy().reshape(res ** 2, -1)
    
    pca = PCA(n_components=n_components, random_state=seed, **pca_kwargs)
    proj = pca.fit_transform(attn_map_flat).reshape(res, res, n_components)
    proj = normalise_image(proj) if normalise else proj

    return proj


def attention_map_cluster(
    attn_map: torch.FloatTensor,
    algorithm: CLUSTERING_ALGORITHM,
    n_clusters: int = 5,
    **clustering_kwargs,
) -> np.ndarray:
    res, features = _validate_attn_map(attn_map)
    attn_map_flat = attn_map.cpu().numpy().reshape(res ** 2, features)

    if algorithm == "gmm":
        model = GaussianMixture(
            n_components=n_clusters,
            **clustering_kwargs,
        )
    elif algorithm == "bgmm":
        model = BayesianGaussianMixture(
            n_components=n_clusters,
            **clustering_kwargs,
        )
    elif algorithm == "kmeans":
        model = KMeans(
            n_clusters=n_clusters,
            **clustering_kwargs,
        )
    else:
        raise ValueError(f"Algorithm {algorithm} not valid")
    clusters = model.fit_predict(attn_map_flat)
    clusters = clusters.reshape(res, res)

    # TODO: Print this with logging
    from collections import Counter
    print(Counter(list(clusters.flatten())))

    return clusters


def stable_diffusion_tokens(
    model: PretrainedStableDiffusionAdapter,
    prompt: str,
    include_separators: bool = False,
) -> List[str]:
    word_end = "</w>"
    separator_start = model.tokenizer.bos_token
    separator_end = model.tokenizer.eos_token

    # TODO: These assertions are all unittests for tokenise_text.
    # Do not include assertions here. Instead, write unittests for 
    # tokenise_text method.
    tokens = model.tokenise_text(prompt, True)
    n_tokens = len(tokens)
    assert tokens[0] == separator_start
    separator_end_index = tokens.index(separator_end)
    assert all([tok not in [separator_start, separator_end] for tok in tokens[1:separator_end_index]])

    suffix_len = len(word_end)
    tokens = [token for token in tokens[1:separator_end_index]]
    if include_separators:
        tokens = [separator_start] + tokens + [separator_end] * (n_tokens - separator_end_index)
    return tokens


def find_tokens_and_noun_indices(model: PretrainedStableDiffusionAdapter, prompt: str) -> Tuple[List[str], List[Tuple[int, str]]]:
    tokens = stable_diffusion_tokens(model, prompt, True)
    pos_tags = nltk.pos_tag(tokens[1:tokens.index(model.tokenizer.eos_token)])
    index_noun_pairs = [
        (i + 1, token)
        for i, (token, pos_tag) in enumerate(pos_tags)
        if pos_tag[:2] == "NN"
    ]
    return tokens, index_noun_pairs


def localise_nouns(
    clusters: np.ndarray,
    cross_attention: torch.FloatTensor,
    index_noun_pairs: List[Tuple[int, str]],
    background_threshold: int = 0.2,
) -> Dict[str, torch.FloatTensor]:
    scale = clusters.shape[0] / cross_attention.shape[0]
    if scale < 1:
        raise ValueError(f"Dimensionality of clusters must be greater than cross_attention (dim(clusters) = {clusters.shape[0]} < dim(cross_attention) = {cross_attention.shape[0]})")
    if cross_attention.shape[0] % clusters.shape[0] == 0:
        raise ValueError(f"dim(cross_attention) = {cross_attention.shape[0]} must be a scale factor of dim(clusters) = {clusters.shape[0]}")

    device = cross_attention.device
    cross_attention = cross_attention.cpu().numpy()

    noun_names = [n for _, n in index_noun_pairs]
    noun_indices = [i for i, _ in index_noun_pairs]
    noun_ca_maps = cross_attention[:, :, noun_indices]

    # Normalise the noun cross-attention maps.
    normalised_noun_ca_maps = np.zeros_like(noun_ca_maps) \
        .repeat(scale, axis=0) \
        .repeat(scale, axis=1)
    for i in range(noun_ca_maps.shape[2]):
        noun_ca_map = noun_ca_maps[:, :, i].repeat(scale, axis=0).repeat(scale, axis=1)
        normalised_noun_ca_maps[:, :, i] = (noun_ca_map - np.abs(noun_ca_map.min())) / noun_ca_map.max()

    masks = {}
    n_clusters = len(np.unique(clusters))
    for c in range(n_clusters):
        cluster_mask = np.zeros_like(clusters)
        cluster_mask[clusters == c] = 1
        # Agreement score between each cluster in the self-attention maps
        # and the cross-attention map for each noun shows how much of the
        # cluster_mask overlaps the cross-attention maps.
        agreement_score_maps = [
            cluster_mask * normalised_noun_ca_maps[:, :, i]
            for i in range(normalised_noun_ca_maps.shape[2])
        ]
        agreement_scores = [
            agreement_score_map.sum() / cluster_mask.sum()
            for agreement_score_map in agreement_score_maps
        ]
        # If none of the agreement scores exceed the threshold, then none
        # of the noun cross-attention maps are sufficiently overlapping
        # the cluster c. Thus, cluster c forms part of the background.
        if max(agreement_scores) > background_threshold:
            c = noun_names[np.argmax(np.array(agreement_scores))]
        else:
            c = "BG"

        if c not in masks:
            masks[c] = cluster_mask
        else:
            masks[c] += cluster_mask

    if "BG" not in masks:
        raise ValueError("Background not found in masks, retry with another different hyperparameters or use a different algorithm.")

    return {label: torch.tensor(mask).to(device) for label, mask in masks.items()}


def find_masks(
    attention_store: AttentionStore,
    index_noun_pairs: List[Tuple[int, str]],
    background_threshold: float = 0.2,
    algorithm: CLUSTERING_ALGORITHM = "kmeans",
    n_clusters: int = 5,
    **clustering_kwargs,
) -> Dict[str, torch.FloatTensor]:
    attn_avg = attention_store.aggregate_attention(
        places_in_unet=["up", "down", "mid"],
        is_cross=False,
        res=32,
        element_name="attn",
    )
    device = attn_avg.device
    clusters = attention_map_cluster(
        attn_avg,
        algorithm=algorithm,
        n_clusters=n_clusters,
        **clustering_kwargs,
    )
    cross_avg = attention_store.aggregate_attention(
        places_in_unet=["up", "down", "mid"],
        is_cross=True,
        res=16,
        element_name="attn",
    )
    masks = localise_nouns(
        clusters,
        cross_avg,
        index_noun_pairs,
        background_threshold,
    )
    return {k: v.to(device) for k, v in masks.items()}


def background_mask(
    attention_store: AttentionStore,
    index_noun_pairs: List[Tuple[int, str]],
    background_threshold: float = 0.2,
    algorithm: CLUSTERING_ALGORITHM = "kmeans",
    n_clusters: int = 5,
    attention_resolution: int = 16,
    upscale_size: int = 512,
    **clustering_kwargs,
) -> torch.FloatTensor:
    masks = find_masks(
        attention_store,
        index_noun_pairs,
        background_threshold,
        algorithm,
        n_clusters,
        **clustering_kwargs,
    )
    bg_map = masks["BG"]
    device = bg_map.device

    # TODO: Directly resize background map without converting to image first
    bg_map = F.interpolate(
        bg_map.float().unsqueeze(0).unsqueeze(0),
        size=(upscale_size, upscale_size),
        mode="nearest",
    ).round().bool().float().squeeze(0).squeeze(0)
    bg_map = F_vision.to_pil_image(bg_map)
    bg_map = bg_map.resize((attention_resolution, attention_resolution))
    bg_map = F_vision.pil_to_tensor(bg_map).bool().float().to(device)
    return bg_map


def center_crop(img: Image.Image) -> Image.Image:
    dim = min(img.size)
    img = F_vision.center_crop(img, output_size=[dim, dim])
    return img 

