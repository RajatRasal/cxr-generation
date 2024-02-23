from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from semantic_editing.utils import init_stable_diffusion, seed_everything, plot_image_on_axis
from semantic_editing.diffusion import StableDiffusionAdapter
from semantic_editing.null_text_inversion import CFGOptimisation, NullTokenOptimisation


def multiple_captions(model: StableDiffusionAdapter, image: Image.Image, captions: List[str], epsilon: float, **nti_kwargs):
    fig, axs = plt.subplots(nrows=1, ncols=len(captions) + 1, figsize=(40, 10))
    plot_image_on_axis(axs[0], image, "Original")
    for caption, ax in zip(captions, axs[1:]):
        nti = NullTokenOptimisation(model, **nti_kwargs)
        nti.fit(image, caption, epsilon)
        recon = nti.generate(caption).resize(image.size)
        mse = ((np.array(image) - np.array(recon)) ** 2).mean()
        plot_image_on_axis(ax, recon, f"MSE: {mse}\n{caption}")
    fig.savefig("multiple_captions.pdf", bbox_inches="tight")


def main():
    image = Image.open("101654506_8eb26cfb60.jpg")
    captions = [
        "A brown and white dog is running through the snow .",
        "A dog is running in the snow",
        "A dog running through snow .",
        "a white and brown dog is running through a snow covered field .",
        "The white and brown dog is running over the surface of the snow .",
    ]

    seed = 88
    seed_everything(seed)

    epsilon = 1e-5
    model = StableDiffusionAdapter(init_stable_diffusion(), 50)
    multiple_captions(model, image, captions, epsilon, **{"guidance_scale": 7.5, "image_size": 512, "num_inner_steps": 20})
