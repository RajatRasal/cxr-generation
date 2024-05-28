import random
from typing import Dict, Literal, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.axes import Axes
from sklearn.neighbors import KernelDensity


def softmax(x: np.array):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def _density_plot_1d(
    ax: Axes,
    data: np.ndarray,
    y_lims: Tuple[float, float],
    linspace: int,
    title: str,
    y_label: Optional[str],
    y_ticks: Literal["left", "right", "none"],
    kde_bandwidth: float,
    true_data: Optional[np.ndarray] = None, 
):
    # TODO: overlay KDEs at the end with true data
    # Sampling intervals
    _ys = np.linspace(*y_lims, linspace)
    ax.set_ylim(*y_lims)

    # Background colour
    ax.set_facecolor("white")

    # Titles and labels
    ax.set_title(title)
    if y_label is not None:
        ax.set_ylabel(y_label, rotation=0)
    ax.set_xticks([])

    # Location of ticks
    if y_ticks == "left":
        pass
    elif y_ticks == "right":
        ax.tick_params(labelleft=False)
        ax.tick_params(labelleft=True)
        ax.yaxis.set_ticks_position("right")
    elif y_ticks == "none":
        ax.tick_params(labelleft=False)
        ax.yaxis.set_ticks_position("none")
    else:
        raise ValueError(f"{y_ticks} must be 'left', 'right', or 'none'")

    # Remove the first and last label
    if y_ticks != "none":
        y_labels = ax.get_yticklabels()
        y_labels[0].set_text("")
        y_labels[-1].set_text("")
        ax.set_yticklabels(y_labels)

    # Black border for the kde plot
    for spine_name in ["bottom", "top", "right", "left"]:
        ax.spines[spine_name].set_color("black")

    # Fit a KDE to data
    kde = KernelDensity(bandwidth=kde_bandwidth)
    kde.fit(data[:, None])
    density = kde.score_samples(_ys[:, None])
    # Plot KDE using softmax to create more contrast between peaks and troughs
    ax.plot(softmax(density), _ys, color="black", label="Sample")

    if true_data is not None:
        # Fit a KDE to the true trajectories
        kde = KernelDensity(bandwidth=kde_bandwidth)
        kde.fit(true_data[:, None])
        true_density = kde.score_samples(_ys[:, None])
        ax.plot(softmax(true_density), _ys, color="grey", ls="--", label="True")


def _trajectory_densities(
    ax: Axes,
    trajectories: np.ndarray,
    title: str,
    forward: bool,
    y_lims: Tuple[float, float],
    fast: bool = True,
):
    # Set background colour for area not covered by KDE
    if fast:
        ax.set_facecolor("white")
    else:
        cmap = "magma"
        colour_palette = sns.mpl_palette(cmap, n_colors=100)
        ax.set_facecolor(colour_palette[0])

    # title
    ax.set_title(title)

    # setting y-axis direction
    ax.set_ylim(*y_lims)

    # setting axis direction
    T = trajectories[0].shape[0]
    timesteps = np.arange(0, T)
    if forward:
        ax.set_xlim(0, T)
    else:
        timesteps = timesteps[::-1]
        ax.set_xlim(T, 0)
    x_labels = ax.get_xticklabels()[:-1]
    ax.set_xticklabels(x_labels)

    # KDEplot normalised per timestep
    if fast:
        for i, traj in enumerate(trajectories[:500]):
            ax.plot(timesteps, traj, color="#3f2b4f", lw=0.5)
    else:
        data = [
            (t - 1, x)
            for traj in trajectories
            for t, x in zip(timesteps, traj)
        ]
        data = random.sample(data, k=1000)
        df = pd.DataFrame(data, columns=["t", "x"])
        sns.kdeplot(
            data=df,
            x="t", y="x",
            ax=ax,
            fill=True,
            thresh=0,
            levels=50,
            cmap=cmap,
            common_norm=False,
        )

    # No labels or ticks
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(labelleft=False)
    ax.yaxis.set_ticks_position("none")

    n_colours = 5
    traj_colour_palette = sns.mpl_palette("tab10", n_colors=n_colours)
    for i, traj in enumerate(trajectories[:n_colours]):
        ax.plot(timesteps, traj, ls="--", color=traj_colour_palette[i], lw=3.0)


def trajectory_plot_1d(
    trajectories: List[np.ndarray],
    T: int,
    y_lims: Tuple[float, float],
    save_path: str,
    kde_bandwidth: float,
    output_type: str,
    true_data: Optional[np.ndarray] = None,
    fast: bool = True,
):
    with plt.style.context("bmh"):
        fig, axes = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(8, 3),
            gridspec_kw={"wspace": 0.05, "width_ratios": [1, 10, 1]},
        )

        # Axes
        init_density_ax = axes[0]
        traj_ax = axes[1]
        final_density_ax = axes[2]

        # TODO: We fit a KDE plot to each timestep, currently the KDE plot is normalised across time
        title = r"$\longrightarrow \text{Reverse process} \longrightarrow$"
        init = 0
        final = -1
        init_density_title = "$z_T$"
        final_density_title = "$z_0$"

        _trajectory_densities(
            traj_ax,
            trajectories,
            title,
            forward=True,
            y_lims=y_lims,
            fast=fast,
        )

        # Beginning and end KDE plots 
        _density_plot_1d(
            ax=init_density_ax,
            data=np.array([traj[init] for traj in trajectories]),
            y_lims=y_lims,
            linspace=100,
            title=init_density_title,
            y_label="$x$",
            y_ticks="left",
            kde_bandwidth=kde_bandwidth,
        )
        _density_plot_1d(
            ax=final_density_ax,
            data=np.array([traj[final] for traj in trajectories]),
            y_lims=y_lims,
            linspace=100,
            title=final_density_title,
            y_label=None,
            y_ticks="right",
            kde_bandwidth=kde_bandwidth,
            true_data=true_data,
        )

        # Save plot
        image_name = f"{save_path}.{output_type}"
        fig.savefig(image_name, format=output_type, bbox_inches="tight")
        return image_name 


def trajectory_plot_1d_with_inverse(
    trajectories: List[np.ndarray],
    inverse_trajectories: List[np.ndarray],
    T: int,
    y_lims: Tuple[float, float],
    save_path: str,
    kde_bandwidth: float,
    output_type: str,
    true_data: Optional[np.ndarray] = None,
    fast: bool = True,
):
    with plt.style.context("bmh"):
        fig, axes = plt.subplots(
            nrows=1,
            ncols=5,
            figsize=(16, 3),
            gridspec_kw={"wspace": 0.05, "width_ratios": [1, 10, 1, 10, 1]},
        )

        # Axes
        init_density_ax = axes[0]
        inverse_traj_ax = axes[1]
        intermediate_density_ax = axes[2]
        traj_ax = axes[3]
        final_density_ax = axes[4]

        # TODO: We fit a KDE plot to each timestep, currently the KDE plot is normalised across time
        title = r"$\longrightarrow \text{Classifier-Free Guidance} \longrightarrow$"
        title_inv = r"$\longrightarrow \text{DDIM Inversion} \longrightarrow$"
        init = 0
        final = -1
        init_density_title = "$z_T$"
        final_density_title = "$z_0$"

        # TODO: assertion to check final inverse and first traj
        check = np.array([traj[final] for traj in inverse_trajectories]) == np.array([traj[init] for traj in trajectories])
        assert all(check.tolist())

        _trajectory_densities(
            inverse_traj_ax,
            inverse_trajectories,
            title_inv,
            forward=False,
            y_lims=y_lims,
            fast=fast,
        )
        _trajectory_densities(
            traj_ax,
            trajectories,
            title,
            forward=True,
            y_lims=y_lims,
            fast=fast,
        )

        # Beginning, intermediate and end KDE plots 
        _density_plot_1d(
            ax=init_density_ax,
            data=np.array([traj[init] for traj in inverse_trajectories]),
            y_lims=y_lims,
            linspace=100,
            title=final_density_title,
            y_label="$x$",
            y_ticks="left",
            kde_bandwidth=kde_bandwidth,
        )
        _density_plot_1d(
            ax=intermediate_density_ax,
            data=np.array([traj[init] for traj in trajectories]),
            y_lims=y_lims,
            linspace=100,
            title=init_density_title,
            y_label="",
            y_ticks="none",
            kde_bandwidth=kde_bandwidth,
        )
        _density_plot_1d(
            ax=final_density_ax,
            data=np.array([traj[final] for traj in trajectories]),
            y_lims=y_lims,
            linspace=100,
            title=final_density_title,
            y_label=None,
            y_ticks="right",
            kde_bandwidth=kde_bandwidth,
            # true_data=true_data,
        )

        # Save plot
        image_name = f"{save_path}.{output_type}"
        fig.savefig(image_name, format=output_type, bbox_inches="tight")
        return image_name 


def visualise_gmm(
    original: torch.FloatTensor,
    center_box: Tuple[float, float],
    keypoints: torch.FloatTensor,
    save_path: str,
    output_type: Literal["pdf", "png"],
):
    with plt.style.context("bmh"):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
        fig.set_facecolor("white")
        ax.set_xlim((center_box[0] - 2, center_box[1] + 2))
        ax.set_ylim((center_box[0] - 2, center_box[1] + 2))
        ax.set_facecolor("white")

        original_df = pd.DataFrame(original, columns=["x", "y"])
        sns.kdeplot(data=original_df, x="x", y="y", ax=ax, fill=True, thresh=0.2, levels=5, cmap="Purples")

        for i, keypoint in enumerate(keypoints):
            ax.scatter(keypoint[0], keypoint[1], marker="x", c="#4A993A", s=0.5, linewidths=0.5)  # "#4A993A")
            ax.annotate(str(i), (keypoint[0] + 0.1, keypoint[1] + 0.1))

        ax.set_xlabel("")
        ax.set_ylabel("")

        image_name = f"{save_path}.{output_type}"
        fig.savefig(image_name, format=output_type, bbox_inches="tight")
