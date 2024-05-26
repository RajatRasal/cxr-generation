from typing import Literal

import torch


def get_device() -> Literal["cuda", "cpu"]:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device)
    generator.manual_seed(seed)
    return generator
