import os
import sys
import mne
import igraph
import numpy as np
import pylab as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable


def imshow_plot(data, ax=None, fname='f', cmap='afmhot',
                figsize=(5, 5),
                interpolation="None",
                vmax=None,
                vmin=None,
                title=None,
                xticks=None,
                yticks=None):

    save_fig = False
    if ax is None:
        fig, ax = pl.subplots(1, figsize=figsize)
        save_fig = True

    im = ax.imshow(data,
                   interpolation=interpolation,
                   cmap=cmap,
                   vmax=vmax,
                   vmin=vmin,
                   origin="lower")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    cbar = pl.colorbar(im, cax=cax, ax=ax)
    cbar.ax.tick_params(labelsize=12)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if title:
        ax.set_title(title, fontsize=14)

    if save_fig:
        pl.savefig(fname)
        pl.close()


def fill_symmetric(a):
    n = a.shape[0]
    i_upper = np.triu_indices(n, 1)
    a[i_upper] = a.T[i_upper]
    return a


def plot_scatter(x, y,
                 ax,
                 xlabel=None,
                 ylabel=None,
                 xlim=None,
                 ylim=None,
                 color="k",
                 alpha=0.4,
                 markersize=10,
                 labelsize=14,
                 fontsize=14,
                 title=None):
    """
    scatter plot.

    :param x: [np.array] values on x axis.
    :param y: [np.array] values on y axis, with save size of x values.
    :param ax: axis of figure to plot the figure.
    :param xlabel: if given, label of x axis.
    :param ylabel: if given, label of y axis.
    :param xlim: [float, float] limit of x values.
    :param ylim: [float, float] limit of y values.
    :param color: [string] color of markers.
    :param alpha: [float] opacity of markers in [0, 1].
    :param markersize: [int] size of marker.
    :param title: title of fiure.
    :return: axis with plot
    """

    assert (len(x) == len(y))

    ax.scatter(x, y, s=markersize,
               color=color, alpha=alpha)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)

    ax.tick_params(labelsize=labelsize)

    if (xlim is None) and (ylim is None):
        ax.margins(x=0.02, y=0.02)

    if title:
        ax.set_title(title, labelsize)

    return ax


# ---------------------------------------------------------------------- #
def scale_distances(x, broad=150):

    a = np.min(x)
    b = np.max(x)
    d = b - a
    xp = (x - a) / d * broad
    return xp


def walktrap(adj, steps=5):
    """
    Community detection algorithm of Latapy & Pons, based on random walks. The basic idea of the algorithm is that short random walks tend to stay in the same community. The result of the clustering will be represented as a dendrogram.

    :param adj: name of an edge attribute or a list containing edge weights
    :param steps: length of random walks to perform
    :return: a L{VertexDendrogram} object, initially cut at the maximum  modularity.

    :Reference: 

    -  Pascal Pons, Matthieu Latapy: Computing communities in large networks using random walks, U{http://arxiv.org/abs/physics/0512106}.

    >>> adj = np.random.rand(20, 20)
    >>> print(walktrap(adj))
    >>> # Clustering with 20 elements and 2 clusters
    >>> # [0] 0, 5, 8, 10, 19
    >>> # [1] 1, 2, 3, 4, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18

    """

    conn_indices = np.where(adj)
    weights = adj[conn_indices]
    edges = list(zip(*conn_indices))
    G = igraph.Graph(edges=edges, directed=False)
    comm = G.community_walktrap(weights, steps=steps)
    communities = comm.as_clustering()

    # print comm
    # print("%s number of clusters = %d " % (
    #     label, len(communities)))
    # print "optimal count : ", comm.optimal_count

    return communities
# ---------------------------------------------------------------------- #


def reorder_nodes(C, communities):

    # reordering the nodes:
    N = C.shape[0]
    n_comm = len(communities)

    nc = []
    for i in range(n_comm):
        nc.append(len(communities[i]))

    # new indices of nodes-------------------------------------------

    newindices = []
    for i in range(n_comm):
        newindices += communities[i]
    # --------------------------------------------------------------

    reordered = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            reordered[i, j] = C[newindices[i], newindices[j]]

    return reordered
# ---------------------------------------------------------------------- #


# epochs = make_fixed_length_epochs(raw,
#                                   preload=True,
#                                   duration=epochs_duration)
# print(info)

# raw.plot(duration=60,
#              order=mag_channels,
#              n_channels=10,
#              remove_dc=False,
#              )
# ===================================================================
# Overview of artifact detection
# https://mne.tools/stable/auto_tutorials/preprocessing/plot_10_preprocessing_overview.html#sphx-glr-auto-tutorials-preprocessing-plot-10-preprocessing-overview-py

# power line noise ==================================================
# fig, axes = pl.subplots(3)
# raw.plot_psd(tmax=np.inf, fmax=250, average=True, ax=axes, show=False)
# # add some arrows at 50 Hz and its harmonics:
# for ax in axes[:2]:
#     freqs = ax.lines[-1].get_xdata()
#     psds = ax.lines[-1].get_ydata()
#     for freq in [50, 100, 150, 200]:
#         idx = np.searchsorted(freqs, freq)
#         print(idx, freqs[idx], psds[idx])
#         ax.arrow(x=freqs[idx], y=psds[idx] + 18, dx=0, dy=-12, color='red',
#                     width=0.1, head_width=3, length_includes_head=True)
# pl.savefig("f.png")

# Heartbeat artifacts (ECG) =========================================
# ecg_epochs = create_ecg_epochs(raw)
# ecg_epochs.plot_image(combine='mean')

# Ocular artifacts (EOG)=============================================
# eog_epochs = mne.preprocessing.create_eog_epochs(raw, baseline=(-0.5, -0.2))
# eog_epochs.plot_image(combine='mean')
# eog_epochs.average().plot_joint()
# eog_epochs = mne.preprocessing.create_eog_epochs(raw,
#                                                  picks="grad",
#                                                  ch_name="MEG0113",
#                                                  baseline=(-0.5, -0.2),
#                                                  )
# eog_epochs.plot_image(combine='mean')
# eog_epochs.average().plot_joint()

# Removing power-line noise with notch filtering=====================
# raw.plot_psd(tmax=100, fmax=250, average=False,
#              show=True, area_mode='range', picks="mag")
# raw.notch_filter(np.arange(50, 300, 50),
#                  picks="mag",
#                  filter_length='auto',
#                  phase='zero')
# raw.plot_psd(tmax=100, fmax=250, average=False,
#              show=True, area_mode='range', picks="mag")

# Removing artifact by low pass filter
# raw.filter(None, 70.,
#            h_trans_bandwidth='auto',
#            filter_length='auto',
#            phase='zero')
# raw.plot_psd(tmax=100, fmax=250, average=False,
#              show=True, area_mode='range', picks="mag")


# raw.plot(n_channels=10, duration=40, start=20)
# print(info["ch_names"])

# mag_channels = pick_types(raw.info, meg='mag')

# ica = ICA(n_components=15, random_state=99)
# ica.fit(raw)
# ica.plot_sources(raw, start=tmin)
# ica.exclude = []