#============================================================================
# Code: Receiving and visualizing data stream via LSL
# Author: Jiachen XU <jiachen.xu.94@gmail.com>
#
# Last Update: 2019-11-29
#============================================================================


import numpy as np
from pylsl import StreamInlet, resolve_stream
from mne import create_info
from mne.io import RawArray
from mne.channels import read_montage
from mne.io.pick import _picks_to_idx, pick_info
import matplotlib.pyplot as plt
from mne.epochs import EpochsArray
from mne.io import read_raw_fif
import matplotlib.gridspec as gridspec
from mne.realtime import LSLClient, MockLSLStream
import copy
from mne import EvokedArray


fs = 512
eeg_ch_names = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7',
                'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
                'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
                'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
                'FPz', 'FP2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4',
                'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
                'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
                'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
emg_ch_names = ['EMG1', 'EMG2', 'EMG3', 'EMG4']
ch_names = eeg_ch_names + emg_ch_names + ['Stim']
ch_types = ['eeg'] * 64 + ['emg'] * 4 + ['stim']
montage = read_montage('standard_1005')
info = create_info(ch_names=ch_names, ch_types=ch_types,
                   sfreq=fs, montage=montage)

info_plot = copy.deepcopy(info)
info_plot['sfreq'] = 1

print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
pick_time = np.int64(fs * np.arange(0.1, 1.1, 0.2)) 

inlet = StreamInlet(streams[0])
plt.ion()
fig_ = plt.figure(figsize=[10, 16])
gs = gridspec.GridSpec(8, 6)
plt_cbar = False
unit_size = 3
nr_comp = 5
if plt_cbar == True:
    n_row, n_col = 8, nr_comp + 1
    w_ratio = [4] * (n_col - 1)
    w_ratio.append(1)
    fig_ = plt.figure(
        figsize=[unit_size * n_col -  unit_size +  unit_size / 4, unit_size * n_row])
    gs = gridspec.GridSpec(
        n_row, n_col, width_ratios=w_ratio)

    ax_pat = []
    for ax_col in range(nr_comp + 1):
        ax_pat.append(fig_.add_subplot(gs[0, ax_col]))
    cbar_axes = ax_pat[-1]
    pat_axes = ax_pat[:-1]
else:
    n_row, n_col = 5 , nr_comp
    w_ratio = [4] * n_col
    fig_ = plt.figure(
        figsize=[(unit_size + 1)  * n_col, unit_size * n_row])
    gs = gridspec.GridSpec(
        n_row, n_col, width_ratios=w_ratio)

    ax_pat = []
    for ax_col in range(nr_comp):
        ax_pat.append(fig_.add_subplot(gs[0, ax_col]))
    cbar_axes = None 
    pat_axes = ax_pat
ax_eeg = fig_.add_subplot(gs[1:, :])

gain = 5e-4
offset = np.tile(np.expand_dims(np.arange(69), axis=1)*gain, fs)
#  import pdb
#  pdb.set_trace()
while True:
    # get a new sample     
    events = np.expand_dims(np.array([0, 1, 1]), axis=0)
    picks = _picks_to_idx(info, None, 'all', exclude=())
    pinfo = pick_info(info, picks)
    joint_kwargs = dict(ts_args=dict(time_unit='s'),
                    topomap_args=dict(time_unit='s'))

    for ii in range(10):
        ax_eeg.clear()
        [ax_pat[ind].clear() for ind in range(len(ax_pat))] 

        sample, timestamp = inlet.pull_chunk(max_samples=fs, timeout=2)
         
        raw_data = np.vstack(sample).T
        mean = np.tile(np.expand_dims(raw_data.mean(axis=1), axis=1), fs)
        data = raw_data - mean + offset
        epoch = EpochsArray(data[picks][np.newaxis], pinfo, events)
        epoch.average().plot(axes=ax_eeg, titles=None)
        pat = EvokedArray(raw_data[:, pick_time], info_plot, tmin=0)

        fig_pat, images, contours_ = pat.plot_topomap(times=range(nr_comp), time_unit='s', ch_type=None,
                layout=None, vmin=None, vmax=None, cmap='RdBu_r',
                colorbar=plt_cbar, res=64, cbar_fmt='%3.1f',
                sensors=True, scalings=1, units='Value',
                time_format=None, size=1.5,
                show_names=False,
                title='', mask_params=None, mask=None,
                outlines='head',
                contours=6, image_interp='bilinear', show=False,
                average=None, head_pos=None, axes=pat_axes)
        if plt_cbar: 
            cbar = fig.colorbar(images[-1], ax=cbar_axes, cax=cbar_axes, format='%3.1f', orientation='vertical')
            cbar.set_ticks(contours_[-1].levels)
            cbar.ax.tick_params(labelsize=7)
        plt.tight_layout()
        plt.pause(1)
    plt.draw()

