#========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2021-09-27 08:28:14
# Name       : data_loader.py
# Version    : V1.0
# Description: .
#========================================
import mne
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from neo.rawio import BlackrockRawIO as br
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)

# --------------------------------------------------------
# Load data
data_root = '/home/jxu/File/Data/Invasive_BCI/test_data/'
#
filename = ['20171206-154207-001.ns6', '20191016-113006-002.ns6']
fileid = 1
reader = br(filename=data_root+filename[fileid])
reader.parse_header()

data = reader.get_analogsignal_chunk(block_index=0, seg_index=0)
float_data = reader.rescale_signal_raw_to_float(raw_signal=data)

n_channels = 96
sampling_freq = 30000  # in Hertz

ds_ratio = 4
downsample_data = float_data.T[:, ::ds_ratio]


# --------------------------------------------------------
# Channel rejection list
reject_chn = True
if reject_chn:
    excl_list = [59, 51, 91, 58, 84, 48, 43, 49, 78, 42, 46, 86, 13, 56, 39, 5, 9,
                 47, 55, 2, 11, 20, 90, 16, 52, 7, 85, 57, 8, 65, 27, 4, 36, 15,
                 22, 1, 12, 29, 59, 37, 28, 62, 19, 25, 91, 53, 54, 45, 31, 23, 87,
                 30, 50, 6, 59, 83, 67, 69, 59, 60, 75, 63, 94, 92, 61, 73, 89,
                 32, 44, 40, 18, 41, 33, 76, 14, 80, 3]
    unique_excl_list = np.unique(excl_list)
    downsample_data = np.delete(downsample_data, unique_excl_list-1, axis=0)
    ch_list = np.setdiff1d(np.arange(1, 97), unique_excl_list)
else:
    ch_list = np.arange(1, 97)
import pdb;pdb.set_trace()

# --------------------------------------------------------
# list of ICs to include during repairing the data set via ICA
ic_incl_list = [24, 27, 34, 35, 36, 37, 38, 41, 44, 49, 50, 51, 52, 54, 55, 57,
                58, 59, 63,64, 67, 68, 72, 76, 79, 80, 81, 82, 83, 84,85, 86,
                87, 89, 90, 91, 92, 94]
ic_excl_list =  np.setdiff1d(np.arange(95), ic_incl_list)


# --------------------------------------------------------
# Create data object to do further preprocessing
ch_names = [f'ECoG_{n:03}' for n in ch_list]
ch_types = ['ecog'] * len(ch_list)
info = mne.create_info(ch_names, sfreq=sampling_freq/ds_ratio, ch_types=ch_types)

simulated_raw = mne.io.RawArray(downsample_data*1e-6, info)
simulated_raw.filter(l_freq=1.0, h_freq=250.0, phase='zero')
simulated_raw.notch_filter(freqs=np.arange(50, 351, 50))
# re-reference based on common average
simulated_raw.set_eeg_reference(ref_channels='average', ch_type='ecog')


# --------------------------------------------------------
# Fit ICA object
ica = ICA(n_components=0.999999, max_iter='auto', random_state=97)
ica.fit(simulated_raw)

# ica.plot_sources(simulated_raw, show_scrollbars=False)
# src_act = ica.get_sources(simulated_raw).get_data()
# chn_act = simulated_raw.get_data()
repair_data = True
if repair_data:
    ica.exclude = ic_excl_list
    raw_cp = simulated_raw.copy()
    ica.apply(raw_cp)
    cleaned_data = raw_cp.get_data()

# --------------------------------------------------------
# Plot the spatial pattern
hm_plot = False
if hm_plot:
    all_mixing_mat = np.dot(ica.mixing_matrix_.T, ica.pca_components_[:ica.n_components_])
    # each row is each mixing vector/spatial pattern

    # Copied from mne official doc to ensure the computation of spatial pattern
    # is correct
    # data = np.dot(ica.mixing_matrix_[:, picks].T,
    #                   ica.pca_components_[:ica.n_components_])
    from matplotlib import gridspec
    width_ratios = [5, 0.5] * 8
    height_ratios = [5] * 12

    len_w = len(width_ratios)
    len_h = len(height_ratios)

    fig_ica = plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))
    gs = gridspec.GridSpec(len(height_ratios), len(width_ratios),
                        height_ratios=height_ratios, width_ratios=width_ratios)
    ax_ica = []
    cax_ica = []
    for i in range(len_h):
        for j in range(len_w):
            if j % 2 == 0:
                ax_ica.append(fig_ica.add_subplot(gs[i, j]))
            elif j % 2 == 1:
                cax_ica.append(fig_ica.add_subplot(gs[i, j]))

    pad_ind_list = np.delete(np.arange(100), [0, 9, 90, 99])
    if reject_chn:
        pad_ind_list = np.delete(pad_ind_list, unique_excl_list-1)

    annot_array = np.zeros((100,))
    annot_array[pad_ind_list] = ch_list  # replace with the spatial pattern vector
    annot_array = annot_array[::-1].reshape((10, 10), order='F')

    for id_sa, data_ in enumerate(all_mixing_mat):
        empty_array = np.zeros((100,))
        empty_array[pad_ind_list] = data_  #  np.arange(1, 97)  # replace with the spatial pattern vector
        sns.heatmap(empty_array[::-1].reshape((10, 10), order='F'),
                    annot=annot_array, ax=ax_ica[id_sa], cbar_ax=cax_ica[id_sa])
        ax_ica[id_sa].set_title(f'IC{id_sa:02}')
    import pdb;pdb.set_trace()
    fig_ica.tight_layout()

    fig_ica.savefig(f'ICA{filename[fileid]}_after_notch.jpg')

