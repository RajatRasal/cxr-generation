import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from improved_diffusion.improved_diffusion.script_util import add_dict_to_argparser
from improved_diffusion.improved_diffusion import dist_util, logger


def main():
    args = create_argparser().parse_args()

    imgs = np.load(args.samples_path)

    for i in range(imgs["arr_0"].shape[0]):
        plt.imsave(os.path.join(args.output_path, str(i)) + ".png", imgs["arr_0"][i])
        plt.show()


def create_argparser():
    defaults = dict(
        samples_path="",
        output_path="",
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
