#========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2020-11-26 13:23:30
# Name       : RS_plot.py
# Version    : V1.0 # Description: .  #========================================
from numpy.fft import fft, fftfreq
from scipy import signal
import matplotlib.pyplot as plt
from itertools import product
from mne.viz import plot_epochs_image

import mne

import pdb

from jxu.data.loader import vhdr_load
from jxu.data.eeg_process import NIBSEEG
from jxu.data.utils import nibs_event_dict

from mne.time_frequency import psd_welch, psd_multitaper
import numpy as np
import pickle

fig_list = []
ax_list = []
for i in range(6):
    fig, ax = plt.subplots(8, 4, figsize=(24, 16), num=i)
    fig_list.append(fig)
    ax_list.append(ax)

when_list = ['pre', 'stim1', 'stim2', 'post', 'all']
task_list = ['RS_close', 'RS_open',
             'QA_trial', 'QA_audio', 'QA_ans', 'QA_rec', 'QA_cen_word',
             'Arti_trial', 'Arti_action', 'Arti_rec']

rs_list = ['open', 'close']
stg_list = ['pre', 'post', 'stim_1', 'stim_2', 'all']


task = 'RS_close'

_, evt_dict, _, _ = nibs_event_dict()
try:
    evt_id, _, tmin, tmax = evt_dict[task][0]
except IndexError:
    evt_id, _ = evt_dict[task][0]
    print('==================================================================')
    print('tmin and tmax are not available for currect task: ' + task)
    print('==================================================================')

for id_sj, subj in enumerate([1, 2, 3, 5, 6, 7, 8, 10]):
    for ses in range(4):

        subj = 2
        ses = 0

        ses_eeg = NIBSEEG(subject=subj, session=ses)
        ses_eeg.raw_load()
        ses_eeg.set_montage()
        ses_eeg.data_concat(cp_flag=True)
        events, event_id = mne.events_from_annotations(ses_eeg.raw_data_clean)
        import pdb;pdb.set_trace()

#         ses_eeg.epoch(task=['RS_close', 'RS_open', 'QA_question', 'QA_answer'],
#                       when=['all', 'pre', 'stim1', 'stim2', 'post'])
        rs_open = mne.Epochs(ses_eeg.raw_data_clean, events, event_id=[evt_id],
                             tmin=tmin, tmax=tmax, picks=['Fz', 'Cz', 'Pz'],
                             baseline=None)
        rs_close = mne.Epochs(ses_eeg.raw_data_clean, events, event_id=[34],
                              tmin=tmin, tmax=tmax, picks=['Fz', 'Cz', 'Pz'],
                              baseline=None)

        for id_stg, stage in enumerate(stg_list):
            if stage == 'pre':
                stg = 0
            elif stage == 'post':
                stg = 3
            elif stage == 'pre&post':
                stg = [0, 3]
            fmax = 225.0

            rs_open[stg].plot_psd(fmax=fmax, color=True, show=False,
                                  ax=ax_list[id_stg][id_sj, ses])
            ax_list[id_stg][id_sj, ses].set_title(
                'S{0}_Ses{1}_RS_Open_{2}'.format(str(subj), str(ses), stage))

#             ax_list[id_stg][id_sj, ses].plot(nf_freq, nf_psd[0].T, 'gray')
            # ax_list[id_stg][id_sj, ses].plot(bp_freq, bp_psd[0].T, 'black')

            rs_close[stg].plot_psd(fmax=fmax, color=True, show=False,
                                   ax=ax_list[id_stg+3][id_sj, ses])
            ax_list[id_stg+3][id_sj, ses].set_title(
                'S{0}_Ses{1}_RS_Close_{2}'.format(str(subj), str(ses), stage))
#             ax_list[id_stg+3][id_sj, ses].plot(nf_freq, nf_psd[0].T, 'gray')
            # ax_list[id_stg+3][id_sj, ses].plot(bp_freq, bp_psd[0].T, 'black')

import pdb;pdb.set_trace()
for i in range(6):
    rs, st = divmod(i, 3)

    fig_list[i].tight_layout()
    fig_list[i].savefig('RS_{0}_{1}.jpg'.format(
        rs_list[rs], stg_list[st]))

import pdb;pdb.set_trace()
ig, ax = plt.subplots(1,1)
ses_eeg.raw_data_clean.plot_psd( tmin=1844.248 , tmax=2026.620 ,fmax = 100.0, fmin=0.1, picks=["Cz"], show=False, ax=ax)
ses_eeg.raw_data_clean.plot_psd( tmin=1844.248 , tmax=2026.620 ,fmax = 100.0, fmin=0.1, picks=["Pz"], show=False, ax=ax)
bipolar.plot_psd(fmax = 100.0, fmin=0.1, show=False, ax=ax)

noise_floor.plot_psd(fmax = 100.0, fmin=0.1, show=False, ax=ax)
plt.show()
import pdb;pdb.set_trace()





import pdb;pdb.set_trace()
ses_eeg.raw_filter()
ses_eeg.set_bad_channels()
import pdb;pdb.set_trace()

