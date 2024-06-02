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
    no_ticks: bool = False,
    n_examples: int = 5,
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
            ax.plot(timesteps, traj, color="dimgray", lw=0.1, alpha=0.75)
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
    if no_ticks:
        ax.tick_params(labelbottom=False)
        ax.xaxis.set_ticks_position("none")
    ax.tick_params(labelleft=False)
    ax.yaxis.set_ticks_position("none")

    traj_colour_palette = sns.mpl_palette("tab10", n_colors=n_examples)
    for i, traj in enumerate(trajectories[:n_examples]):
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


def trajectory_plots_1d_with_inverse(
    trajectories_x: List[np.ndarray],
    inverse_trajectories_x: List[np.ndarray],
    trajectories_y: List[np.ndarray],
    inverse_trajectories_y: List[np.ndarray],
    recons: np.ndarray,
    T: int,
    y_lims: Tuple[float, float],
    save_path: str,
    kde_bandwidth: float,
    output_type: str,
    true_data: Optional[np.ndarray] = None,
    fast: bool = True,
    title: str = r"$\longrightarrow \text{Classifier-Free Guidance} \longrightarrow$",
    title_inv: str = r"$\longrightarrow \text{DDIM Inversion} \longrightarrow$",
):
    with plt.style.context("bmh"):
        fig, axes = plt.subplots(
            nrows=2,
            ncols=8,
            figsize=(25, 6),
            gridspec_kw={
                "wspace": 0.05,
                "width_ratios": [1, 10, 1, 10, 1, 1, 3, 3],
                "hspace": 0.1,
                "height_ratios": [1, 1],
            },
        )

        gs = axes[0, 6].get_gridspec()
        for ax in axes[:, 5:].flatten():
            ax.remove()
        ax_edits = fig.add_subplot(gs[:, 6:])

        init = 0
        final = -1
        init_density_title = "$z_T$"
        final_density_title = "$z_0$"

        check = np.array([traj[final] for traj in inverse_trajectories_x]) == np.array([traj[init] for traj in trajectories_x])
        assert all(check.tolist())
        check = np.array([traj[final] for traj in inverse_trajectories_y]) == np.array([traj[init] for traj in trajectories_y])
        assert all(check.tolist())

        ### Edits
        ax_edits.set_title(r"$ \text{Edits} $")
        ax_edits.set_xlim((y_lims[0] + 1, y_lims[1] - 1))
        ax_edits.set_ylim((y_lims[0] + 1, y_lims[1] - 1))
        ax_edits.set_facecolor("white")
        original_df = pd.DataFrame(true_data, columns=[r"$x$", r"$y$"])
        # edit_df = pd.DataFrame([
        #     (x_traj[-1], y_traj[-1])
        #     for x_traj, y_traj in zip(trajectories_x, trajectories_y)
        # ], columns=[r"$x$", r"$y$"])
        sns.kdeplot(data=original_df, x=r"$x$", y=r"$y$", ax=ax_edits, fill=True, thresh=0.2, levels=5, cmap="Greys")
        n_examples = 5
        palette = sns.mpl_palette("tab10", n_colors=n_examples)
        for traj_x, traj_y, c in zip(trajectories_x[:n_examples], trajectories_y[:n_examples], palette):
            ax_edits.plot(traj_x[-1], traj_y[-1], "x", color=c)
        for traj_x, traj_y, c in zip(inverse_trajectories_x[:n_examples], inverse_trajectories_y[:n_examples], palette):
            ax_edits.plot(traj_x[0], traj_y[0], "o", color=c)
        for point, c in zip(recons, palette):
            ax_edits.plot(point[0, 0], point[0, 1], "^", color=c)
        ax_edits.margins(2)
        ax_edits.tick_params(labelleft=False)
        ax_edits.tick_params(labelright=True)
        ax_edits.yaxis.set_ticks_position("right")
        ax_edits.set_ylabel(ax_edits.get_ylabel(), rotation="horizontal")
        ax_edits.yaxis.set_label_position("right")
        xticks = ax_edits.get_xticklabels()
        xticks[0] = ""
        xticks[-1] = ""
        yticks = ax_edits.get_yticklabels()
        yticks[0] = ""
        yticks[-1] = ""
        ax_edits.set_xticklabels(xticks)
        ax_edits.set_yticklabels(yticks)

        ### Top row 
        top_row = axes[0]

        _trajectory_densities(
            top_row[1],
            inverse_trajectories_x,
            title_inv,
            forward=False,
            y_lims=y_lims,
            fast=fast,
            no_ticks=True,
        )
        _trajectory_densities(
            top_row[3],
            trajectories_x,
            title,
            forward=True,
            y_lims=y_lims,
            fast=fast,
            no_ticks=True,
        )

        _density_plot_1d(
            ax=top_row[0],
            data=np.array([traj[init] for traj in inverse_trajectories_x]),
            y_lims=y_lims,
            linspace=100,
            title=final_density_title,
            y_label="$x$",
            y_ticks="left",
            kde_bandwidth=kde_bandwidth,
        )
        _density_plot_1d(
            ax=top_row[2],
            data=np.array([traj[init] for traj in trajectories_x]),
            y_lims=y_lims,
            linspace=100,
            title=init_density_title,
            y_label="",
            y_ticks="none",
            kde_bandwidth=kde_bandwidth,
        )
        _density_plot_1d(
            ax=top_row[4],
            data=np.array([traj[final] for traj in trajectories_x]),
            y_lims=y_lims,
            linspace=100,
            title=final_density_title,
            y_label=None,
            y_ticks="right",
            kde_bandwidth=kde_bandwidth,
        )

        ### Bottom row
        bottom_row = axes[1]

        _trajectory_densities(
            bottom_row[1],
            inverse_trajectories_y,
            "",
            forward=False,
            y_lims=y_lims,
            fast=fast,
        )
        _trajectory_densities(
            bottom_row[3],
            trajectories_y,
            "",
            forward=True,
            y_lims=y_lims,
            fast=fast,
        )

        _density_plot_1d(
            ax=bottom_row[0],
            data=np.array([traj[init] for traj in inverse_trajectories_y]),
            y_lims=y_lims,
            linspace=100,
            title="",
            y_label="$y$",
            y_ticks="left",
            kde_bandwidth=kde_bandwidth,
        )
        _density_plot_1d(
            ax=bottom_row[2],
            data=np.array([traj[init] for traj in trajectories_y]),
            y_lims=y_lims,
            linspace=100,
            title="",
            y_label="",
            y_ticks="none",
            kde_bandwidth=kde_bandwidth,
        )
        _density_plot_1d(
            ax=bottom_row[4],
            data=np.array([traj[final] for traj in trajectories_y]),
            y_lims=y_lims,
            linspace=100,
            title="",
            y_label=None,
            y_ticks="right",
            kde_bandwidth=kde_bandwidth,
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
        sns.kdeplot(data=original_df, x="x", y="y", ax=ax, fill=True, thresh=0.2, levels=5, cmap="Greys")

        for i, keypoint in enumerate(keypoints):
            ax.scatter(keypoint[0], keypoint[1], marker="x", c="#4A993A", s=0.5, linewidths=0.5)  # "#4A993A")
            ax.annotate(str(i), (keypoint[0] + 0.1, keypoint[1] + 0.1))

        ax.set_xlabel("")
        ax.set_ylabel("")

        image_name = f"{save_path}.{output_type}"
        fig.savefig(image_name, format=output_type, bbox_inches="tight")
