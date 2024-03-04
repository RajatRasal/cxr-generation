from typing import Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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
    n_clusters: int = 5,
    seed: int = 42,
    **kmeans_kwargs,
) -> np.ndarray:
    res1, res2, features = _validate_attn_map(attn_map)

    attn_map_flat = attn_map.cpu().numpy().reshape(res1 ** 2, -1)

    kmeans = KMeans(n_clusters=5, random_state=seed, **kmeans_kwargs).fit(attn_map_flat)
    clusters = kmeans.labels_
    clusters = clusters.reshape(res1, res2)

    return clusters


