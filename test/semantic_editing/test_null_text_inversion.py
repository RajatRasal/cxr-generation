import random

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from semantic_editing.utils import plot_image_on_axis


def test_reconstruction(nti, pti, cfg_ddim, image_prompt):
    image, prompt = image_prompt

    nti.fit(*image_prompt, 1e-5)
    pti.fit(*image_prompt, 1e-5)
    cfg_ddim.fit(*image_prompt)

    nti_recon = nti.generate(prompt)
    assert nti_recon.size == (nti.image_size, nti.image_size)
    pti_recon = pti.generate(prompt, 0)
    assert pti_recon.size == (pti.image_size, pti.image_size)
    cfg_recon = cfg_ddim.generate(prompt)
    assert cfg_recon.size == image.size

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(25, 10))
    plot_image_on_axis(axs[0], image, "Original")
    plot_image_on_axis(axs[1], cfg_recon, "DDIM + CFG Reconstruction")
    plot_image_on_axis(axs[2], nti_recon.resize(image.size), "NTI Reconstruction")
    plot_image_on_axis(axs[3], pti_recon.resize(image.size), "PTI Reconstruction")
    fig.savefig("test_reconstructions.pdf", bbox_inches="tight")

    image_arr = np.array(image.resize((nti.image_size, nti.image_size)))
    nti_recon_err = ((image_arr - np.array(nti_recon)) ** 2).mean()
    pti_recon_err = ((image_arr - np.array(pti_recon)) ** 2).mean()
    cfg_recon_err = ((np.array(image) - np.array(cfg_recon)) ** 2).mean()

    assert 35 <= nti_recon_err <= 36
    assert pti_recon_err <= 35
    assert pti_recon_err < nti_recon_err < cfg_recon_err
