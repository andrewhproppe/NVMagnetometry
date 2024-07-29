import numpy
import numpy as np
import seaborn as sns
from src.visualization.fig_utils import dress_fig, set_font_size, make_fig
from matplotlib import pyplot as plt, pyplot

"""
visualize.py

Include routines here that will visualize parts of your analysis.

This can include taking serialized models and seeing how a trained
model "sees" inputs at each layer, as well as just making figures
for talks and writeups.
"""


def plot_mulitple_VAE_output(
    xdata,
    yraw,
    ytest,
    ytrue,
    spacer,
    alpha=0.5,
    color_fit=None,
    plot_mean=True,
    plot_raw_all=False,
    plot_fit_all=False,
    colors=None,
    figsize=(4, 4),
    labels=None,
):
    if color_fit is None:
        color_fit = [0.5, 0.5, 0.5]

    if colors is None:
        colors = np.array(sns.color_palette("icefire", 12))

    if labels is None:
        labels = ["Data", "True", "NN fit", "NN mean fit"]

    fig = plt.figure(dpi=150, figsize=figsize)
    for ctr, y in enumerate(ytest):
        # Plot raw data
        if ctr == 0 or plot_raw_all:
            plt.scatter(
                xdata,
                yraw + spacer * ctr,
                marker="s",
                color=colors[1],
                s=2,
                alpha=0.5,
                label=labels[0],
            )

        # Plot true distribution
        if ctr == 0 or plot_fit_all:
            plt.plot(
                xdata,
                ytrue + spacer * ctr,
                "--",
                color="k",
                ms=1,
                lw=1,
                label=labels[1],
            )

        # Plot fits
        if ctr == 0:
            plt.plot(
                xdata, y + spacer * ctr, color=color_fit, alpha=alpha, label=labels[2]
            )
        else:
            plt.plot(xdata, y + spacer * ctr, color=color_fit, alpha=alpha)

    if plot_mean:
        plt.plot(
            xdata, ytest.mean(0) + spacer * ctr, color=[0.6, 0.2, 0.2], label=labels[3]
        )

    dress_fig(xlabel="$\\tau$ ($\mu$s)", ylabel="$g^{(2)}(\\tau)$", legend=True)


def plot_ensemble_variance(
    x,
    g2_ens,
    g2_true=None,
    input=None,
    fill=True,
    nstd=2,
    idx=0,
    cidxs=np.array([3, 8, 1, 10]),
    figsize=(4, 4),
    labels=None,
    colors=None,
    fontsize=7,
):
    if labels is None:
        labels = [
            "$\mu$",
            "$\mu$ + %d$\sigma$" % nstd,
            "$\mu$ - %d$\sigma$" % nstd,
            "Input",
        ]

    if colors is None:
        colors = np.array(sns.color_palette("icefire", 12))

    mean = g2_ens[0][idx]
    var = g2_ens[1][idx]

    set_font_size(fontsize)
    fig, ax = make_fig(figsize)

    if not input is None:
        # plt.scatter(x, input[idx], marker='s', s=2, alpha=0.5, label=labels[-1], color=colors[cidxs[0]])
        # plt.plot(x, input[idx], alpha=0.7, lw=0.5, label=labels[-1], color=colors[cidxs[0]])
        plt.plot(x, input, alpha=0.5, lw=0.5, label=labels[-1], color=colors[cidxs[0]])

    if fill:
        # fill_col = colors[cidxs[0]]
        fill_col = [0, 0, 0]
        plt.fill_between(
            x,
            mean + nstd * np.sqrt(var),
            mean - nstd * np.sqrt(var),
            linewidth=0.0,
            color=fill_col,
            alpha=0.3,
            label="$\mu$ +/- %d$\sigma$" % nstd,
        )
        # plt.plot(x, mean + nstd * np.sqrt(var), lw=0.2, color=fill_col, alpha=.5)
        # plt.plot(x, mean - nstd * np.sqrt(var), lw=0.2, color=fill_col, alpha=.5)

    if not g2_true is None:
        plt.plot(x, g2_true, "--", label="True", color="k")
    plt.plot(x, mean, label=labels[0], color="k")

    if not fill:
        plt.plot(x, mean + nstd * np.sqrt(var), label=labels[1], color=colors[cidxs[1]])
        plt.plot(x, mean - nstd * np.sqrt(var), label=labels[2], color=colors[cidxs[2]])

    dress_fig(
        xlabel="$\\tau$ ($\mu$s)", ylabel="$g^{(2)}(\\tau)$", legend=True, tight=False
    )

    return fig


def plot_ensemble_submodels(
    x, g2_ens, spacer=0, alpha=1, idx=0, figsize=(4, 4), colors=None, fontsize=7
):
    if colors is None:
        colors = np.array(sns.color_palette("icefire", 12))

    set_font_size(fontsize)
    fig, ax = make_fig(figsize)
    for i, s in enumerate(g2_ens[2][:, idx, :]):
        plt.plot(x, s[0 : int(len(s) / 2)] + i * spacer, color=colors[i], alpha=alpha)

    dress_fig(xlabel="$\\tau$ (ps)", ylabel="$g^{(2)}(\\tau)$", tight=False)

    return fig


def plot_MCMC_variance(
    x,
    y,
    y_MCMC,
    ppc,
    y_true=None,
    cidxs=np.array([3, 8, 1, 10]),
    figsize=(2, 2),
    labels=None,
    colors=None,
    ppc_alpha=0.5,
    ppc_color=None,
    fontsize=7,
):
    if labels is None:
        # labels = ['$\mu$', '$\mu$ + %d$\sigma$' % nstd, '$\mu$ - %d$\sigma$' % nstd, 'Input']
        labels = ["$\mu$ HMC"]

    if ppc_color is None:
        ppc_color = [0.5, 0.5, 0.5]

    if colors is None:
        colors = np.array(sns.color_palette("icefire", 12))

    set_font_size(fontsize)
    fig, ax = make_fig(figsize[0], figsize[1])

    for ctr, p in enumerate(ppc):
        if ctr == 0:
            plt.plot(
                x, p, lw=0.5, alpha=ppc_alpha, color=ppc_color, label="$\sigma$ HMC"
            )
        else:
            plt.plot(x, p, lw=0.5, alpha=ppc_alpha, color=ppc_color)

    plt.plot(x, y, alpha=0.5, lw=0.5, label="Input", color=colors[cidxs[0]])
    if not y_true is None:
        plt.plot(x, y_true, "--", lw=0.5, label="True", color="k")
    plt.plot(x, y_MCMC, lw=1, label="$\mu$ HMC", color="k")

    dress_fig(
        xlabel="$\\tau$ ($\mu$s)",
        ylabel="$g^{(2)}(\\tau)$",
        legend=True,
        tight=False,
        lgd_loc="upper left",
    )

    return fig


def plot_adj_matrix(weights, ax=None):
    indices = np.tril_indices(8)

    # Initialize an 8x8 matrix with zeros
    adj = np.zeros((8, 8))

    # Assign the lower triangle elements to the matrix
    adj[indices] = weights

    ax.matshow(adj)

    # plt.show()


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    weights = np.random.rand(36)

    plot_adj_matrix(weights, ax=ax)
