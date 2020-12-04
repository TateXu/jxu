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
from jxu.data.loader import vhdr_load
from jxu.data.preprocess import NIBSEEG
from mne.time_frequency import psd_welch, psd_multitaper
import numpy as np
import pickle


subj = 1
ses = 0
ses_eeg = NIBSEEG(subject=subj, session=ses)
ses_eeg.raw_load()
ses_eeg.set_montage()
ses_eeg.data_concat(cp_flag=True)

pre_stim = ses_eeg.raw_data_clean.copy()
pre_stim = pre_stim.crop(tmax=2086.010)
pre_stim.info['bads'] = ['Audio', 'tACS', 'EOG151', 'EOG152', 'FC3', 'FFC5h', 'FCC5h', 'FC5', 'FTT7h', 'CP5']
# common average reference
pre_stim.set_eeg_reference('average', projection=False)
# filter data
pre_stim.notch_filter(np.arange(50, 400, 50))
pre_stim.filter(l_freq=1, h_freq=None)
# get montage

montage = mne.channels.read_custom_montage(
    '/home/jxu/anaconda3/lib/python3.7/site-packages/jxu/data/BC-TMS-128.bvef')
pre_stim.set_montage(montage, on_missing='warn')
# raw.plot_sensors()  # plot the channel location in a 2D topomap
# run ica
ica = ICA(n_components=120)
ica.fit(pre_stim)
s = ica.get_sources(pre_stim)
s.plot()
ica.plot_properties(pre_stim, picks=18)
ica.plot_components()

import pdb;pdb.set_trace()
