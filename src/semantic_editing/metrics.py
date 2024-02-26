import os

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
# from torchmetrics.functional.image import learned_perceptual_image_patch_similarity
# from torchmetrics.functional.image import frechet_inception_distance

from semantic_editing.utils import plot_image_on_axis


def plot_counterfactuals_by_image(
    output_path: str,
    idx: int,
):
    output_dir = os.path.join(output_path, str(idx))
    df = pd.read_csv(os.path.join(output_dir, "metadata.csv"))

    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(40, 40))
    for i, (ax, output_path) in enumerate(zip(axs.flatten(), df.output_path)):
        plot_image_on_axis(ax, Image.open(output_path), "")

    for i, caption in enumerate(df.original_caption.unique()):
        axs[0, i].set_title(caption)
        # axs[i, 0].set_title(caption, rotation="vertical")
    
    fig.savefig(os.path.join(output_dir, "grid.pdf"), bbox_inches="tight")


def main():
    for i in range(35, 36):
        plot_counterfactuals_by_image(
            "/vol/biomedic3/rrr2417/cxr-generation/Flicker8k_Counterfactuals_full_ddim/",
            4,
        )
