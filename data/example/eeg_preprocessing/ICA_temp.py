#========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2020-11-27 13:40:41
# Name       : ICA_temp.py
# Version    : V1.0
# Description: .
#========================================


import mne
from mne.preprocessing import ICA
import os
import numpy as np
# read data

filename = "/home/jxu/File/Data/NIBS/Stage_one/EEG/Exp/TES/Session_0/TES_seg_0.vhdr"

full_raw = mne.io.read_raw_brainvision(filename, preload=True)

import pdb;pdb.set_trace()
raw = full_raw.crop(tmin=0.0, tmax=2.0)
# plot raw data

# raw.plot()
# raw.plot_psd(fmin=0, fmax=45)
# determine bad channels
raw.info['bads'] = ['Audio', 'tACS', 'EOG151', 'EOG152', 'FC3', 'FFC5h', 'FCC5h', 'FC5', 'FTT7h', 'CP5']
# common average reference
raw.set_eeg_reference('average', projection=False)
# filter data
raw.notch_filter(np.arange(50, 400, 50))
raw.filter(l_freq=1, h_freq=None)
# get montage

montage = mne.channels.read_custom_montage(
    '/home/jxu/anaconda3/lib/python3.7/site-packages/jxu/data/BC-TMS-128.bvef')
raw.set_montage(montage, on_missing='warn')
# raw.plot_sensors()  # plot the channel location in a 2D topomap
# run ica
ica = ICA(n_components=120)
ica.fit(raw)
source = ica.get_sources(raw)
import pdb;pdb.set_trace()
# img_s = source.plot(show=False)
# img_prop = ica.plot_properties(raw, picks=18)
img_comp = ica.plot_components(show=False)

import pdb;pdb.set_trace()
