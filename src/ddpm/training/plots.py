import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KernelDensity


def softmax(x: np.array):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def trajectory_plot(timesteps, trajectories, T: int, y_lim: int, n_trajectories: int, save_path: str):
    # plot forward diffusion
    # TODO: Install cm-super so that latex can be used in matplotlib
    # plt.rc("text", usetex=True)
    # plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

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

        y_lims = (-y_lim, y_lim)

        # Trajectory density
        # TODO: Draw arrows in latex using matplotlib
        traj_ax.set_title(r"$\longleftarrow p_t(x) \longrightarrow$")
        traj_ax.set_xlim(0, T)
        traj_ax.set_ylim(*y_lims)
        traj_ax.tick_params(labelleft=False)
        traj_ax.yaxis.set_ticks_position("none")
        dfs = pd.concat([
            pd.DataFrame(np.array([timesteps, traj]).T, columns=["t", "x"])
            for traj in trajectories
        ])
        colour_palette = sns.mpl_palette("magma", n_colors=100)
        traj_ax.set_facecolor(colour_palette[0])
        # Some good visualisations in this one: https://www.imperial.ac.uk/media/imperial-college/faculty-of-engineering/computing/public/2223-ug-projects/Investigating-numerical-methods-in-score-based-models.pdf
        # TODO: We fit a KDE plot to each timestep, currently the KDE plot is normalised across time
        sns.kdeplot(data=dfs, x="t", y="x", ax=traj_ax, fill=True, thresh=0, levels=100, cmap="magma")
        # sns.displot(data=dfs, x="t", y="x", hue="x", ax=traj_ax, kind="kde", thresh=0, levels=100, cmap="magma", common_norm=False, fill=True)
        traj_ax.set_ylabel("")
        traj_ax.set_xlabel("")

        # Trajectory plot
        for traj in trajectories[:n_trajectories]:
            traj_ax.plot(timesteps, traj)

        _ys = np.linspace(*y_lims, 100)
        # Init density plot
        init_density_ax.set_facecolor("white")
        init_density_ax.set_title("$p_0(x)$")
        init_density_ax.set_ylim(*y_lims)
        init_density_ax.set_ylabel("$x$", rotation=0)
        init_density_ax.set_xticks([])
        init_density_ax_y_labels = init_density_ax.get_yticklabels()
        init_density_ax_y_labels[0].set_text("")
        init_density_ax_y_labels[-1].set_text("")
        init_density_ax.set_yticklabels(init_density_ax_y_labels)
        for spine in ["bottom", "top", "right", "left"]:
            init_density_ax.spines[spine].set_color("black")
        init_kde = KernelDensity()
        init_kde.fit(np.array([traj[0] for traj in trajectories])[:, None])
        init_density = init_kde.score_samples(_ys[:, None])
        init_density_ax.plot(softmax(init_density), _ys, color="black")

        # Final density plot
        final_density_ax.set_facecolor("white")
        final_density_ax.set_title("$p_T(x)$")
        final_density_ax.set_ylim(*y_lims)
        final_density_ax.set_xticks([])
        final_density_ax.tick_params(labelleft=False)
        final_density_ax.tick_params(labelright=True)
        final_density_ax.yaxis.set_ticks_position("right")
        final_density_ax_y_labels = final_density_ax.get_yticklabels()
        final_density_ax_y_labels[0].set_text("")
        final_density_ax_y_labels[-1].set_text("")
        final_density_ax.set_yticklabels(final_density_ax_y_labels)
        for spine in ["bottom", "top", "right", "left"]:
            final_density_ax.spines[spine].set_color("black")
        final_kde = KernelDensity()
        final_kde.fit(np.array([traj[-1] for traj in trajectories])[:, None])
        final_density = final_kde.score_samples(_ys[:, None])
        final_density_ax.plot(softmax(final_density), _ys, color="black")

        # Save plot
        fig.tight_layout()
        fig.savefig(save_path)
