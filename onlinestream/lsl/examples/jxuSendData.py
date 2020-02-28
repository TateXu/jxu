#============================================================================
# Code: Sending data stream via LSL
# Author: Jiachen XU <jiachen.xu.94@gmail.com>
#
# Last Update: 2019-11-29
#============================================================================


import numpy as np
import time
from scipy import io as sio
from random import random as rand

from pylsl import StreamInfo, StreamOutlet


fs = 512  
update_time = 0.01

nsample = np.int(fs*update_time)
data = sio.loadmat('data/s01.mat', squeeze_me=True, struct_as_record=False, verify_compressed_data_integrity=False)['eeg']

imagery_left = data.imagery_left - \
    data.imagery_left.mean(axis=1, keepdims=True)
imagery_right = data.imagery_right - \
    data.imagery_right.mean(axis=1, keepdims=True)

eeg_data_l = np.vstack([imagery_left * 1e-6, data.imagery_event])
eeg_data_r = np.vstack([imagery_right * 1e-6,
                        data.imagery_event * 2])
eeg_data = np.hstack([eeg_data_l, np.zeros((eeg_data_l.shape[0], 500)),
                      eeg_data_r])

datamat = eeg_data 
if datamat.ndim != 2:
    raise ValueError("INPUT must be 2-dim!")
len_off_data = datamat.shape[1]
info = StreamInfo('BioSemi', 'EEG', datamat.shape[0],  fs, 'float32', 'myuid34234')
outlet = StreamOutlet(info)
axis = np.arange(len_off_data)
live_index = np.arange(nsample)
print("now sending data...")
counter = 0
while True:
    mysample = datamat[:, counter].ravel().tolist()
    # now send it and wait
    outlet.push_sample(mysample)
    counter = 0 if counter == len_off_data - 1 else counter + 1
    time.sleep(1.0 / fs)
