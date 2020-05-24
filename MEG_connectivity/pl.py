import os
import sys
import mne
import lib
import numpy as np
import pylab as pl
from main import *


plot_time_series = 1

if __name__ == "__main__":

    if channel_type == "grad":
        space = 5e-11
    if channel_type == "mag":
        space = 3e-12

    for method in methods:
        d = 'data/figs/{}'.format(method)
        if not os.path.exists(d):
            os.makedirs(d)

    for sub_ID in subject_IDs:

        if plot_time_series:
            fig, ax = pl.subplots(1, figsize=(10, 4))
            raw_fname = "MAD-CN-{:04d}_tsss_mc_raw".format(sub_ID)
            raw = io.read_raw_fif(os.path.join(
                subject_dir, raw_fname + ".fif"))
            raw.crop(tmin=20,
                     tmax=30).load_data().filter(l_freq=0.4,
                                                 h_freq=100)
            data, times = raw.get_data(return_times=True,
                                       picks=channel_type,
                                       verbose=True)
            for i in range(10):
                ax.plot(times, data[i, :] + i * space, lw=0.5, color="k")
            ax.margins(x=0.01, y=0.01)
            ax.axis("off")
            pl.savefig("data/figs/raw_{:02d}_{:s}.png".format(
                    sub_ID, channel_type))
            pl.close()
            exit(0)

        for freq_band in freq_dict:
            for method in methods:
                file_name = "{:02d}_{:s}_{:s}".format(sub_ID,
                                                      freq_band,
                                                      method)
                data = np.load("data/npz/" + file_name + ".npz")

                fig, ax = pl.subplots(nrows=1, ncols=2, figsize=(8, 3))
                pl.subplots_adjust(wspace=0.2)
                con = data["c"]
                x = data["x"]
                y = data["y"]

                lib.imshow_plot(con,
                                ax=ax[0],
                                xticks=[],
                                yticks=[],
                                title="",
                                vmax=1,  # np.max(con),
                                vmin=0)

                # comm = lib.walktrap(con, steps=5)
                # r_con = lib.reorder_nodes(con, comm)
                # lib.imshow_plot(r_con,
                #                 ax=ax[1],
                #                 xticks=[],
                #                 yticks=[],
                #                 title="",
                #                 vmax=1,  # np.max(con),
                #                 vmin=0)
                if plot_distance_matrix:
                    lib.imshow_plot(dist_matrix,
                                    fname="data/figs/d_{}.png".format(sub_ID),
                                    # xticks=[], yticks=[],
                                    title="distances",
                                    figsize=(5, 5),
                                    )
                    plot_distance_matrix = 0

                lib.plot_scatter(x, y, ax[1],
                                 xlabel="distances [mm]",
                                 ylabel=method,
                                 alpha=0.2,
                                 fontsize=14)
                pl.tight_layout()
                ax[1].axis([0, 150, -1, 1])

                pl.savefig("data/figs/{:s}/fc_{:02d}_{:s}_{:s}.png".format(
                    method,
                    sub_ID,
                    freq_band,
                    method))
                pl.close()
