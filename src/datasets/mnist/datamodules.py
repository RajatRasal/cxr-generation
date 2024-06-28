import os
from typing import Literal, Tuple

import numpy as np
import lightning as L
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2

from datasets.mnist.datasets import MNIST
from ddpm.utils import get_generator


class MNISTDataModule(L.LightningDataModule):

    def __init__(
        self,
        data_dir: str = "src/datasets/mnist/files/raw",
        seed: int = 42,
        split_ratio: Tuple[float, float] = (0.9, 0.1),
        batch_size: int = 32,
        num_workers: int = 0,
        one_hot: bool = True,
        normalise: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.seed = seed
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.one_hot = one_hot
        self.normalise = normalise

    def setup(self, stage: Literal["predict", "test", "train"]):
        transforms_list = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
        if self.normalise:
            # Known mean and std for MNIST
            transforms_list.append(v2.Normalize(mean=(0.1307,), std=(0.3081,)))
        self.transform = v2.Compose(transforms_list)
        if self.one_hot:
            self.target_transform = v2.Lambda(lambda x: F.one_hot(torch.tensor(x), num_classes=10))
        else:
            self.target_transform = v2.Lambda(lambda x: torch.tensor(x).unsqueeze(0).long())

        if stage == "fit":
            mnist_full = MNIST(
                self.data_dir,
                train=True,
                transform=self.transform,
                target_transform=self.target_transform,
            )
            self.mnist_train, self.mnist_val = random_split(
                mnist_full,
                self.split_ratio,
                generator=get_generator(self.seed, "cpu"),
            )
        elif stage == "test":
            self.mnist_test = MNIST(
                self.data_dir,
                train=False,
                transform=self.transform,
                target_transform=self.target_transform,
            )
        elif stage == "predict":
            self.mnist_predict = MNIST(
                self.data_dir,
                train=False,
                transform=self.transform,
                target_transform=self.target_transform,
            )
        else:
            raise ValueError(f"stage can only be `fit`, `test` or `predict`, not {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, shuffle=False)


def get_mnist_variant(
    variant: Literal["grey", "colour"],
    seed: int,
    batch_size: int,
    num_workers: int,
) -> MNISTDataModule:
    kwargs = {
        "seed": seed,
        "split_ratio": (0.9, 0.1),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "one_hot": False,
        "normalise": False,  # True,
    }
    if variant == "grey":
        dm = MNISTDataModule(
            data_dir="src/datasets/mnist/files/raw",
            # data_dir="/vol/biomedic3/rrr2417/cxr-generation/src/datasets/mnist/files/morphomnist_thick_thin",
            **kwargs,
        )
    elif variant == "colour":
        dm = MNISTDataModule(
            data_dir="src/datasets/mnist/files/colour_with_fixed_saturation",
            **kwargs,
        )
    else:
        raise ValueError(f"mnist_variant must be {'grey', 'colour'}, not {mnist_variant}")
    return dm
