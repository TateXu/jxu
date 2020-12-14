##=============================================================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2020-10-03 10:38:25
# Name       : audio_std_batch.py
# Version    : V1.0
# Description: Preload and standardise answer audio
##=============================================================================

from jxu.data.audio_process import NIBSAudio
import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import f_oneway


nr_subject = 10  # [1, 5, 6]:
nr_ses = 3
print('=========================================================')
print(' Start Subject {0} Session {1}'.format(nr_subject, nr_ses))
print('=========================================================')


ses_audio = NIBSAudio(subject=nr_subject, session=nr_ses)
# Load stadardised audio files: 44100sps, 16bit, mono channel

ses_audio.audio_load(preload=False, std=False)
ses_audio.audio_std()

ses_audio.audio_load(preload=False, std=True)
ses_audio.audio_to_seg(opt_noise_level=5)
import pdb;pdb.set_trace()
