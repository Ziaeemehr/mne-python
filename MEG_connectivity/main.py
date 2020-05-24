import os
import sys
import mne
import lib
import numpy as np
import pylab as pl
from mne.connectivity import spectral_connectivity
from sklearn.metrics.pairwise import euclidean_distances
from mne import (make_fixed_length_events,
                 make_fixed_length_epochs,
                 pick_types,
                 Epochs,
                 io,
                 )
from mne.preprocessing import (create_ecg_epochs,
                               create_eog_epochs,
                               corrmap,
                               ICA,
                               )

# ------------------------------------------------------------------#

subject_IDs = list(range(30))
subject_dir = "dataset"

freq_dict = {
    "delta": (0.4, 4),
    "theta": (4, 8),
    "alpha1": (8, 10),
    "alpha2": (10, 13),
    "beta1": (13, 20),
    "beta2": (20, 30),
    "gamma1": (30, 45),
    "gamma2": (45, 70),
}

tmin = 0.0  # [s]
tmax = 60.0  # [s]
event_id, event_tmin = 1, -0.5
channel_type = "mag"  # "mag"  # grad
# methods = ['pli', 'coh', 'imcoh',
#            'plv', 'ciplv', 'ppc',
#            'wpli']

methods = ['coh', 'plv', 'ppc']

plot_distance_matrix = 0  # if true, plot once
mode = "multitaper"  # multitaper, fourier, cwt_morlet.

# ica_excludes = [[4], [], [0], [12], [], [], [], [], [], []]
# ------------------------------------------------------------------#

if __name__ == "__main__":

    for sub_ID in subject_IDs:
        raw_fname = "MAD-CN-{:04d}_tsss_mc_raw".format(sub_ID)
        raw = io.read_raw_fif(os.path.join(subject_dir, raw_fname + ".fif"))
        raw.crop(tmin=tmin, tmax=tmax)  # .load_data()
        raw.load_data().filter(l_freq=0.4, h_freq=100)
        sfreq = raw.info['sfreq']  # the sampling frequency

        for freq_band in freq_dict:

            fmin, fmax = freq_dict[freq_band]
            event_tmax = duration = 1.0 / fmin * 7 - \
                event_tmin          # considering 7 cycles

            events = mne.make_fixed_length_events(raw,
                                                  start=10,
                                                  stop=tmax,
                                                  duration=duration)

            picks = pick_types(raw.info,
                               meg=channel_type,
                               eeg=False,
                               stim=False,
                               eog=False,
                               exclude='bads')

            epochs = Epochs(raw,
                            events,
                            event_id,
                            event_tmin,
                            event_tmax,
                            picks=picks,
                            baseline=None,
                            )

            # print(info['projs'])

            epochs.load_data().pick_types(meg=channel_type)
            sens_loc = [raw.info['chs'][k]['loc'][:3] for k in picks]
            sens_loc = np.array(sens_loc)
            dist_matrix = euclidean_distances(sens_loc)

            # con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
            for method in methods:
                con, _, _, _, _ = spectral_connectivity(
                    epochs,
                    method=['coh', 'plv'], #method,
                    mode=mode,
                    sfreq=sfreq,
                    fmin=fmin,
                    fmax=fmax,
                    faverage=True,
                    tmin=0.0,  # exclude the baseline period
                    mt_adaptive=False,
                    n_jobs=4)
                
                print(len(con), con[0].shape)
                exit(0)

                con = lib.fill_symmetric(con[:, :, 0])
                N = con.shape[0]

                upper_ind = np.triu_indices(N, 1)
                x_scatter = dist_matrix[upper_ind]
                x_scatter = lib.scale_distances(x_scatter)
                y_scatter = con[upper_ind]

                file_name = "{:02d}_{:s}_{:s}".format(sub_ID,
                                                      freq_band,
                                                      method)
                np.savez("data/npz/" + file_name,
                         c=con,
                         x=x_scatter,
                         y=y_scatter,
                         d=dist_matrix)

    # print(sens_loc.shape)
    # print(dist_matrix.shape)
