#========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2020-11-26 13:23:30
# Name       : RS_plot.py
# Version    : V1.0
# Description: .
#========================================


from numpy.fft import fft, fftfreq
from scipy import signal
import matplotlib.pyplot as plt
from itertools import product
from mne.viz import plot_epochs_image
import platform

import mne

import pdb

from jxu.data.loader import vhdr_load
from jxu.data.preprocess import NIBSEEG
from mne.time_frequency import psd_welch, psd_multitaper
import numpy as np
import pickle

bipolar_1 = vhdr_load("/home/jxu/Group_share/projects/z010_noise_measurement/Hoerlgasse/noise/recordings/bipolar_1_1_a.vhdr")
bipolar_2 = vhdr_load("/home/jxu/Group_share/projects/z010_noise_measurement/Hoerlgasse/noise/recordings/bipolar_1_1_b.vhdr")
bipolar_3 = vhdr_load("/home/jxu/Group_share/projects/z010_noise_measurement/Hoerlgasse/noise/recordings/bipolar_1_2.vhdr")
bipolar = mne.concatenate_raws([bipolar_1, bipolar_2, bipolar_3])

noise_floor_1 = vhdr_load("/home/jxu/Group_share/projects/z010_noise_measurement/Hoerlgasse/noise/recordings/noiseFloor_2.vhdr")
noise_floor_2 = vhdr_load("/home/jxu/Group_share/projects/z010_noise_measurement/Hoerlgasse/noise/recordings/noiseFloor_3.vhdr")
noise_floor = mne.concatenate_raws([noise_floor_1, noise_floor_2])

bipolar.annotations.delete([1, 2, 3, 4, 5, 6])
events_bp, events_id_bp = mne.events_from_annotations(bipolar)
bp = mne.Epochs(bipolar, events_bp, event_id=99999, tmin=0.0, tmax=2438.7,
                baseline=None, reject=None)

noise_floor.annotations.delete([1, 2, 3])
events_nf, events_id_nf = mne.events_from_annotations(noise_floor)
nf_ep = mne.Epochs(noise_floor, events_nf, event_id=99999, tmin=0.0, tmax=921.2,
                   baseline=None, picks=['Cz'], reject=None)

fig, ax = plt.subplots(1, 1, figsize=(6, 2))
#  aaa = nf_ep.plot_psd(fmax=225.0, color=False, show=False,ax=ax)
import pdb;pdb.set_trace()
# psd_list_3, freq_3 = psd_multitaper(bipolar, fmax=225.0, fmin=0.1 )
# psd_list_4, freq_4 = psd_multitaper(noise_floor, fmax=225.0, fmin=0.1)
# plt.plot(freq_3, np.log10(psd_list_3.T)*10, 'yellow')
# plt.plot(freq_4, np.log10(psd_list_4[0].T)*10, 'red')
# plt.legend([Bipolar', 'Noise_saline'])
# import pdb;pdb.set_trace()

with open('/home/jxu/anaconda3/lib/python3.7/site-packages/jxu/data/example/eeg_preprocessing/nf_psd.pkl', 'rb') as f_out:
    nf_psd, nf_freq = pickle.load(f_out)

with open('/home/jxu/anaconda3/lib/python3.7/site-packages/jxu/data/example/eeg_preprocessing/bp_psd.pkl', 'rb') as f_out:
    bp_psd, bp_freq = pickle.load(f_out)


fig_list = []
ax_list = []
for i in range(6):
    fig, ax = plt.subplots(8, 4, figsize=(24, 16), num=i)
    fig_list.append(fig)
    ax_list.append(ax)

rs_list = ['open', 'close']
stg_list = ['pre', 'post', 'pre&post']

for id_sj, subj in enumerate([1, 2, 3, 5, 6, 7, 8, 10]):
    for ses in range(4):

        ses_eeg = NIBSEEG(subject=subj, session=ses)
        ses_eeg.raw_load()
        ses_eeg.set_montage()
        ses_eeg.data_concat(cp_flag=True)
        events, event_id = mne.events_from_annotations(ses_eeg.raw_data_clean)
        rs_open = mne.Epochs(ses_eeg.raw_data_clean, events, event_id=[32],
                             tmin=0, tmax=180, picks=['Fz', 'Cz', 'Pz'],
                             baseline=None)
        rs_close = mne.Epochs(ses_eeg.raw_data_clean, events, event_id=[34],
                              tmin=0, tmax=180, picks=['Fz', 'Cz', 'Pz'],
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

