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


subj = 5
ses = 2
ind = 0
ses_eeg = NIBSEEG(subject=subj, session=ses)
ses_eeg.raw_load()
ses_eeg.set_montage()
ses_eeg.data_concat(cp_flag=True)

ses_eeg.set_channels(reset=True)
ses_eeg.rereference('average')
ses_eeg.raw_filter(bands=[(1.0, 70.0)], notch=True)

import pdb;pdb.set_trace()
ses_eeg.data[0].save('split_eeg/nibs_raw.fif', split_size='1GB')
ses_eeg.plot_montage(axes='2D')

seg = 'post'   # whole , pre, stim, post'
raw = ses_eeg.data_seg(name=seg, filtered_seg=0)

ICA_folder = ses_eeg.root + "ICA/" + ses_eeg.eeg_folder


import pdb;pdb.set_trace()
for i in [-14, -10, -1]:
    create_folder(ICA_folder[:i])


try:
    with open(ICA_folder + seg + "_ICA_obj.pkl", 'rb') as f:
        ica = pickle.load(f)
except FileNotFoundError:
    ica = ICA(n_components=None)
    ica.fit(raw)

    with open(ICA_folder + seg + "_ICA_obj.pkl", 'wb') as f:
        pickle.dump(ica, f)

import pdb;pdb.set_trace()
img_comp = ica.plot_components(show=False)
create_folder(ICA_folder + "Pattern/")
for id_ig, ig in enumerate(img_comp):
    ig.savefig('{0}Pattern/{1}_fig_{2}.png'.format(
        ICA_folder, seg, str(id_ig)))

import pdb;pdb.set_trace()
# img_prop = ica.plot_properties(raw, picks=18)
# create_folder(ICA_folder + "Properties/")
# for id_ig, ig in enumerate(img_comp):
    # ig.savefig('{0}Properties/{1}_fig_{2}.png'.format(
        # ICA_folder, seg, str(id_ig)))
# import pdb;pdb.set_trace()

sources = ica.get_sources(raw)
img_s = sources.plot(show=False)
