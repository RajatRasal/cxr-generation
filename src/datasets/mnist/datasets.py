import os
from pathlib import Path
from typing import Union, Optional, Callable, Tuple, Any, Dict

import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity

from datasets.mnist.io import load_idx


class MNIST(VisionDataset):

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
    ]

    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        self.data, self.targets = self._load_data()

    @property
    def train_labels(self):
        return self.targets

    @property
    def test_labels(self):
        return self.targets

    @property
    def train_data(self):
        return self.data

    @property
    def test_data(self):
        return self.data

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte.gz"
        data = load_idx(os.path.join(self.root, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte.gz"
        targets = load_idx(os.path.join(self.root, label_file))

        return data, targets
        
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"image": img, "label": target}

    def __len__(self) -> int:
        return len(self.data)

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(self.root, os.path.splitext(os.path.basename(url))[0]))
            for url, _ in self.resources
        )

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


class ColourMNIST(MNIST):

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        hue_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.hues = self._load_hue()
        self.hue_transform = hue_transform

    def _load_hue(self):
        _file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte.gz"
        hues = load_idx(os.path.join(self.root, label_file))
        return hues

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = super().__getitem__(index)

        hue = self.hues[index]
        if self.hue_transform is not None:
            hue = self.hue_transform(hue)

        return {"hue": hue, **data}
