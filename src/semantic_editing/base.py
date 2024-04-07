from abc import ABC, abstractmethod, abstractclassmethod
from typing import List

import torch
from PIL import Image


NULL_STRING = ""


class CFGOptimisation(ABC):

    # @property
    # @abstractmethod
    # def model(self) -> StableDiffusionAdapter:
    #     raise NotImplementedError

    # @property
    # @abstractmethod
    # def guidance_scale(self) -> int:

    @abstractmethod
    def fit(self, image: Image.Image, prompt: str) -> List[torch.FloatTensor]:
        """
        Returns attention maps computed at the end of each fitting step.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str) -> Image.Image:
        raise NotImplementedError

    @abstractmethod
    def save(self, dirname: str):
        raise NotImplementedError

    @abstractclassmethod
    def load(cls, dirname: str) -> "CFGOptimisation":
        raise NotImplementedError

