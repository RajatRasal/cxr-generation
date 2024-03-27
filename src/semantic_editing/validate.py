import torch

def _validate_attn_map(attn_map: torch.FloatTensor):
    shape = attn_map.shape
    if len(shape) != 3:
        raise ValueError(f"Invalid attention map. Must have 3 dimensions not {shape}")
    if shape[0] != shape[1]:
        raise ValueError(f"Invalid attention map. Dim 0 ({shape[0]}) != Dim 1 ({shape[1]})")
    res, _, features = shape
    return res, features

