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
from mne.time_frequency import psd_welch
ses_eeg = NIBSEEG(subject=2, session=1)

ses_eeg.raw_load()

ses_eeg.data_concat(cp_flag=True)
import numpy as np


bipolar_1 = vhdr_load("/home/jxu/Group_share/projects/z010_noise_measurement/Hoerlgasse/noise/recordings/bipolar_1_1_a.vhdr")
bipolar_2 = vhdr_load("/home/jxu/Group_share/projects/z010_noise_measurement/Hoerlgasse/noise/recordings/bipolar_1_1_b.vhdr")
bipolar_3 = vhdr_load("/home/jxu/Group_share/projects/z010_noise_measurement/Hoerlgasse/noise/recordings/bipolar_1_2.vhdr")
bipolar = mne.concatenate_raws([bipolar_1, bipolar_2, bipolar_3])


noise_floor_1 = vhdr_load("/home/jxu/Group_share/projects/z010_noise_measurement/Hoerlgasse/noise/recordings/noiseFloor_2.vhdr")
noise_floor_2 = vhdr_load("/home/jxu/Group_share/projects/z010_noise_measurement/Hoerlgasse/noise/recordings/noiseFloor_3.vhdr")
noise_floor = noise_floor_1   # mne.concatenate_raws([noise_floor_1, noise_floor_2])
picks = ['Pz', 'Cz', 'TP9', 'TP10']

psd_list_1, freq_1 = psd_welch(ses_eeg.raw_data_clean, fmax = 300.0, fmin=0.1, picks=picks, n_fft=2048)
psd_list_3, freq_3 = psd_welch(bipolar, fmax=300.0, fmin=0.1, n_fft=1024*5)
psd_list_4, freq_4 = psd_welch(noise_floor, fmax=300.0, fmin=0.1, n_fft=2048, picks=picks, average='mean')


plt.plot(freq_1, np.log10(psd_list_1.T)*10)
plt.plot(freq_3, np.log10(psd_list_3.T)*10)
plt.plot(freq_4, np.log10(psd_list_4.T)*10)
plt.legend(picks + ['Bipolar'] + [i + '_noise_floor' for i in picks])
plt.title('NUK Session 2 (1st Segment, whole) + NF (2nd Segment, whole)')

fig, ax = plt.subplots(1,1)
ses_eeg.raw_data_clean.plot_psd(fmax = 300.0, fmin=0.1, picks=["Cz"], show=False, ax=ax)
ses_eeg.raw_data_clean.plot_psd(fmax = 300.0, fmin=0.1, picks=["Pz"], show=False, ax=ax)
bipolar.plot_psd(fmax = 300.0, fmin=0.1, show=False, ax=ax)

noise_floor.plot_psd(fmax = 300.0, fmin=0.1, show=False, ax=ax)
plt.show()
import pdb;pdb.set_trace()





import pdb;pdb.set_trace()
ses_eeg.raw_filter()
ses_eeg.set_montage()
ses_eeg.set_bad_channels()
import pdb;pdb.set_trace()