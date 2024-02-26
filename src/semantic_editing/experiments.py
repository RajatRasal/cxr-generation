import argparse
import os
from collections import defaultdict
from typing import List, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.datasets import Flickr8k
from tqdm import tqdm

from semantic_editing.utils import init_stable_diffusion, seed_everything, plot_image_on_axis
from semantic_editing.diffusion import StableDiffusionAdapter
from semantic_editing.null_text_inversion import CFGOptimisation, NullTokenOptimisation, PromptTokenOptimisation


def image_counterfactuals_by_caption(
    model: StableDiffusionAdapter,
    optimiser: Type[CFGOptimisation],
    image: Image.Image,
    captions: List[str],
    **cfg_kwargs,
) -> List[Tuple[str, str, Image.Image]]:
    counterfactuals = []
    for start_caption in captions:
        cfg = optimiser(model, **cfg_kwargs)
        cfg.fit(image, start_caption)
        for target_caption in captions:
            counterfactual = cfg.generate(target_caption)
            counterfactuals.append((start_caption, target_caption, counterfactual))
    return counterfactuals


# def multiple_captions_plot():
#     fig, axs = plt.subplots(nrows=len(captions), ncols=len(captions), figsize=(40, 40))
#     for row, caption in enumerate(captions):
#         ax_row = axs[row, :]
#         nti = NullTokenOptimisation(model, **nti_kwargs)
#         nti.fit(image, caption)
#         for caption, ax in zip(captions, ax_row):
#             recon = nti.generate(caption)
# 
#             loss_fn_alex = lpips.LPIPS(net="alex")
#             image_tensor = F.pil_to_tensor(image.resize(recon.size)).unsqueeze(0)
#             recon_tensor = F.pil_to_tensor(recon).unsqueeze(0)
#             d = loss_fn_alex(image_tensor, recon_tensor)
#             mse = ((np.array(image.resize(recon.size)) - np.array(recon)) ** 2).mean()
# 
#             plot_image_on_axis(ax, recon.resize(image.size), f"MSE: {mse} | LPIPS: {d.item()}") # \n{caption}")
#     fig.savefig(file_name, bbox_inches="tight")


def load_flickr_captions(ann_file: str, images_path: str) -> pd.DataFrame:
    df = pd.read_csv(ann_file, sep="\t", header=None, names=["image", "caption"])
    df.image = df.image.str[:-2]
    df = df.groupby("image")["caption"].apply(list).reset_index(name="captions")
    df["image_path"] = df.image.apply(lambda name: os.path.join(images_path, name))
    return df


def flickr8k_counterfactuals_by_image(
    model: StableDiffusionAdapter,
    optimiser: Type[CFGOptimisation],
    df: pd.DataFrame,
    image_ids: List[int],
    output_path: str,
    **cfg_kwargs,
):
    for idx, row in df.iloc[image_ids].iterrows():
        output_dir = os.path.join(output_path, str(idx))
        os.makedirs(output_dir, exist_ok=True)

        image = Image.open(row.image_path)
        image.save(os.path.join(output_dir, row.image))
        cfs = image_counterfactuals_by_caption(
            model,
            optimiser,
            image,
            row.captions,
            **cfg_kwargs,
        )

        df_content = []
        for i, (source, target, image) in enumerate(cfs):
            image_output_path = os.path.join(output_dir, str(i) + ".png")
            df_content.append((source, target, image_output_path))
            image.save(image_output_path, "PNG")

        df = pd.DataFrame(df_content, columns=["original_caption", "target_caption", "output_path"])
        df.to_csv(os.path.join(output_dir, "metadata.csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_idxs", nargs="+", type=int)
    parser.add_argument("--optimiser", default="nti", choices=["nti", "pti"])
    parser.add_argument("--ddim_steps", default=50, type=int)
    args = parser.parse_args()

    # TODO: Store flickr dataset on S3 bucket and download it from there
    seed = 88
    seed_everything(seed)

    model = StableDiffusionAdapter(init_stable_diffusion(), args.ddim_steps)
    cfg_kwargs = {"guidance_scale": 7.5, "image_size": 512, "num_inner_steps": 20, "epsilon": 1e-5}

    if args.optimiser == "nti":
        output_path = "/vol/biomedic3/rrr2417/cxr-generation/Flicker8k_Counterfactuals_full_ddim/"
    else:
        output_path = "/vol/biomedic3/rrr2417/cxr-generation/Flicker8k_Counterfactuals_pti/"
    os.makedirs(output_path, exist_ok=True)

    if args.optimiser == "nti":
        optimiser = NullTokenOptimisation
    else:
        optimiser = PromptTokenOptimisation

    images_path = "/vol/biomedic3/rrr2417/cxr-generation/Flicker8k_Dataset/"
    ann_file = "/vol/biomedic3/rrr2417/cxr-generation/Flickr8k.token.txt"
    df = load_flickr_captions(ann_file, images_path)

    flickr8k_counterfactuals_by_image(model, optimiser, df, args.image_idxs, output_path, **cfg_kwargs)
