import argparse
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from tqdm import tqdm

from datasets.mnist.io import load_idx, save_idx


def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    assert len(image.shape) == 2
    assert image.shape[0] == image.shape[1]
    image = np.expand_dims(image, axis=0)
    image = np.repeat(image, repeats=3, axis=0)
    return image


def set_hs(image: np.ndarray, hue: float, saturation: float) -> np.ndarray:
    image = np.moveaxis(image, 0, 2)
    image = rgb_to_hsv(image)
    # (HSV, height, width)
    H, S, V = 0, 1, 2
    mask = image[:, :, V] != 0
    image[:, :, H][mask] = hue
    image[:, :, S][mask] = saturation
    image = hsv_to_rgb(image)
    image = np.moveaxis(image, 2, 0)
    return image


def colour_mnist_with_fixed_saturation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=str, default="src/datasets/mnist/files/raw")
    parser.add_argument("--output_path", type=str, default="src/datasets/mnist/files/colour_with_fixed_saturation")
    parser.add_argument("--saturation", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Seed for sampling hues
    np.random.seed(args.seed)

    # Make output path and overwrite existing location
    output_path = os.path.join(args.output_path, f"{str(args.seed)}_{str(args.saturation)}")
    os.makedirs(output_path, exist_ok=True)

    def _helper(in_file, out_file, sample_name, out_file_hue):
        images = load_idx(in_file)

        # Sample hues
        hues = np.random.uniform(low=0.0, high=1.0, size=len(images))
        # Change hue of each image and save to new array
        shape = images.shape
        perturbed_images = np.empty((shape[0], 3, shape[1], shape[2]), dtype=images.dtype)
        for i in tqdm(range(len(images))):
            img_rgb = grayscale_to_rgb(images[i])
            img_colour = set_hs(img_rgb, hues[i], args.saturation)
            perturbed_images[i] = img_colour

        # Plot some images
        fig, axes = plt.subplots(nrows=2, ncols=5)
        for img, hue, ax in zip(perturbed_images, hues, axes.flatten()):
            ax.set_title(f"Hue: {round(hue, 2)}")
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_xticks([])
            ax.imshow(np.moveaxis(img, 0, 2))
        fig.savefig(sample_name, bbox_inches="tight")

        # Save dataset to raw files
        save_idx(perturbed_images, out_file)
        save_idx(hues, out_file_hue)

    # Load dataset from raw files
    train_name = "train-images-idx3-ubyte.gz"
    train_name_hue = "train-hues-idx1-ubyte.gz"
    train_name_label = "train-labels-idx1-ubyte.gz"
    test_name = "t10k-images-idx3-ubyte.gz"
    test_name_hue = "t10k-hues-idx1-ubyte.gz"
    test_name_label = "t10k-labels-idx1-ubyte.gz"

    train_path = os.path.join(args.raw_path, train_name)
    test_path = os.path.join(args.raw_path, test_name)

    train_path_out = os.path.join(output_path, train_name)
    test_path_out = os.path.join(output_path, test_name)

    train_path_out_hue = os.path.join(output_path, train_name_hue)
    test_path_out_hue = os.path.join(output_path, test_name_hue)

    train_path_out_label = os.path.join(output_path, train_name_label)
    test_path_out_label = os.path.join(output_path, test_name_label)

    _helper(
        train_path,
        train_path_out,
        os.path.join(output_path, "train_sample.pdf"),
        train_path_out_hue,
    )
    _helper(
        test_path,
        test_path_out,
        os.path.join(output_path, "test_sample.pdf"),
        test_path_out_hue,
    )

    for path in [train_name_label, test_name_label]:
        shutil.copy2(
            os.path.join(args.raw_path, path),
            os.path.join(output_path, path),
        )
