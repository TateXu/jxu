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
from jxu.data.eeg_process import NIBSEEG
from mne.time_frequency import psd_welch, psd_multitaper
import numpy as np
import pickle

from jxu.basiccmd.mycmd import create_folder


subj = 2
ses = 0
ind = 0
ses_eeg = NIBSEEG(subject=subj, session=ses)
ses_eeg.raw_load()
ses_eeg.raw_data[ind].info['bads'] = ['FC3', 'FFC5h', 'FCC5h', 'FFC3h', 'FCC3h', 'T7', 'FTT7h', 'TTP7h']
ses_eeg.raw_data[ind].filter(l_freq=1.0, h_freq=70.0)

events, event_id = mne.events_from_annotations(ses_eeg.raw_data[ind])
rs_flag = events[np.where(events[:,2] == 34)[0][0]][0]/1000
ses_eeg.raw_data[ind].plot(n_channels=32, start=rs_flag, scalings=65e-6, bad_color=(1.0, 0.0, 0.0))
import pdb;pdb.set_trace()
