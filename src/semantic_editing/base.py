from abc import ABC, abstractmethod

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
    def fit(self, image: Image.Image, prompt: str):
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str) -> Image.Image:
        pass

