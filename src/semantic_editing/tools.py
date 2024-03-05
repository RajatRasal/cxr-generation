from typing import Dict, List, Tuple

import nltk
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from semantic_editing.diffusion import StableDiffusionAdapter


def normalise_image(image: np.ndarray) -> np.ndarray:
    shape = image.shape
    assert len(shape) == 3 and shape[0] == shape[1]

    image_min = image.min(axis=(0, 1))
    image_max = image.max(axis=(0, 1))
    image_normalised = (image - image_min) / (image_max - image_min)

    return image_normalised


def _validate_attn_map(attn_map: torch.FloatTensor):
    if len(attn_map.shape) != 3:
        raise ValueError(f"Invalid attention map. Must have 3 dimensions not {attn_map.shape}")
    res1, res2, features = attn_map.shape[0], attn_map.shape[1], attn_map.shape[2]
    assert res1 == res2
    return res1, res2, features


def attention_map_pca(
    attn_map: torch.FloatTensor,
    n_components: int = 3,
    normalise: bool = True,
    seed: int = 42,
    **pca_kwargs,
) -> np.ndarray:
    res1, res2, features = _validate_attn_map(attn_map)

    attn_map_flat = attn_map.cpu().numpy().reshape(res1 ** 2, -1)
    
    pca = PCA(n_components=n_components, random_state=seed, **pca_kwargs)
    proj = pca.fit_transform(attn_map_flat).reshape(res1, res1, n_components)
    proj = normalise_image(proj) if normalise else proj

    return proj


def attention_map_cluster(
    attn_map: torch.FloatTensor,
    gmm: bool = False,
    n_clusters: int = 5,
    **clustering_kwargs,
) -> np.ndarray:
    res1, res2, features = _validate_attn_map(attn_map)

    attn_map_flat = attn_map.cpu().numpy().reshape(res1 ** 2, -1)

    if gmm:
        model = GaussianMixture(
            n_components=n_clusters,
            **clustering_kwargs,
        )
    else:
        model = KMeans(
            n_clusters=n_clusters,
            **clustering_kwargs,
        )
    clusters = model.fit_predict(attn_map_flat)
    clusters = clusters.reshape(res1, res2)

    return clusters


def find_noun_indices(model: StableDiffusionAdapter, prompt: str) -> List[Tuple[int, str]]:
    tokens = model.tokenise_text(prompt, True)
    suffix_len = len("</w>")
    tokens = [token[:-suffix_len] for token in tokens[1:tokens.index("<|endoftext|>")]]
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

    normalised_noun_ca_maps = np.zeros_like(noun_ca_maps).repeat(scale, axis=0).repeat(scale, axis=1)

    for i in range(noun_ca_maps.shape[2]):
        noun_ca_map = noun_ca_maps[:, :, i].repeat(scale, axis=0).repeat(scale, axis=1)
        normalised_noun_ca_maps[:, :, i] = (noun_ca_map - np.abs(noun_ca_map.min())) / noun_ca_map.max()

    cluster_interpretation = {}
    cluster_masks = {}
    n_clusters = len(np.unique(clusters))
    for c in range(n_clusters):
        cluster_mask = np.zeros_like(clusters)
        cluster_mask[clusters == c] = 1
        score_maps = [cluster_mask * normalised_noun_ca_maps[:, :, i] for i in range(len(noun_indices))]
        scores = [score_map.sum() / cluster_mask.sum() for score_map in score_maps]
        cluster_interpretation[c] = noun_names[np.argmax(np.array(scores))] if max(scores) > background_threshold else "BG"
        cluster_masks[c] = cluster_mask

    return cluster_interpretation, cluster_mask

