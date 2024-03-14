from typing import Dict, List, Literal, Tuple

import nltk
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from semantic_editing.diffusion import StableDiffusionAdapter
from semantic_editing.attention import AttentionStore


def normalise_image(image: np.ndarray) -> np.ndarray:
    shape = image.shape
    assert len(shape) == 3 and shape[0] == shape[1]

    image_min = image.min(axis=(0, 1))
    image_max = image.max(axis=(0, 1))
    image_normalised = (image - image_min) / (image_max - image_min)

    return image_normalised


def _validate_attn_map(attn_map: torch.FloatTensor):
    shape = attn_map.shape
    if len(shape) != 3:
        raise ValueError(f"Invalid attention map. Must have 3 dimensions not {shape}")
    if shape[0] != shape[1]:
        raise ValueError(f"Invalid attention map. Dim 1 ({shape[0]}) != Dim 2 ({shape[1]})")
    res, _, features = shape
    return res, features


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
    algorithm: Literal["kmeans", "gmm", "bgmm"],
    n_clusters: int = 5,
    **clustering_kwargs,
) -> np.ndarray:
    res, features = _validate_attn_map(attn_map)

    attn_map_flat = attn_map.cpu().numpy().reshape(res ** 2, -1)

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

    return clusters


def stable_diffusion_tokens(
    model: StableDiffusionAdapter,
    prompt: str,
    include_separators: bool = False,
) -> List[str]:
    word_end = "</w>"
    separator_start = "<|startoftext|>"
    separator_end = "<|endoftext|>"

    # TODO: These assertions are all unittests for tokenise_text.
    # Do not include assertions here. Instead, write unittests for 
    # tokenise_text method.
    tokens = model.tokenise_text(prompt, True)
    assert tokens[0] == separator_start
    separator_end_index = tokens.index(separator_end)
    assert all([tok not in [separator_start, separator_end] for tok in tokens[1:separator_end_index]])

    suffix_len = len(word_end)
    tokens = [token[:-suffix_len] for token in tokens[1:separator_end_index]]
    if include_separators:
        tokens = [separator_start] + tokens + [separator_end]
    return tokens


def find_noun_indices(model: StableDiffusionAdapter, prompt: str) -> List[Tuple[int, str]]:
    tokens = stable_diffusion_tokens(model, prompt)
    pos_tags = nltk.pos_tag(tokens)
    return [
        (i + 1, token)
        for i, (token, pos_tag) in enumerate(pos_tags)
        if pos_tag[:2] == "NN"
    ]


def localise_nouns(
    clusters: np.ndarray,
    cross_attention: np.ndarray,
    index_noun_pairs: List[Tuple[int, str]],
    background_threshold: int = 0.2,
) -> Dict[str, np.ndarray]:
    scale = clusters.shape[0] / cross_attention.shape[0]
    assert scale > 1

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

    return masks


def background_mask(
    attention_store: AttentionStore,
    index_noun_pairs: List[Tuple[int, str]],
    background_threshold: float = 0.2,
) -> torch.FloatTensor:
    attn_avg = attention_store.aggregate_attention(
        places_in_unet=["up", "down", "mid"],
        is_cross=False,
        res=32,
        element_name="attn",
    )
    clusters = attention_map_cluster(attn_avg, algorithm="kmeans", n_clusters=5)

    cross_avg = attention_store.aggregate_attention(
        places_in_unet=["up", "down", "mid"],
        is_cross=True,
        res=16,
        element_name="attn",
    )
    masks = localise_nouns(
        clusters,
        cross_avg.cpu().numpy(),
        index_noun_pairs,
        background_threshold,
    )
    return masks["BG"]
