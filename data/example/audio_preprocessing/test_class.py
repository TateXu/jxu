##=============================================================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2020-08-29 12:55:41
# Name       : test_class.py
# Version    : V1.0
# Description: Minimal example code for the NIBSAudio class
##=============================================================================

from jxu.data.preprocess import NIBSAudio
import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import f_oneway
ses_audio = NIBSAudio(subject=10, session=1)


# Load stadardised audio files: 44100sps, 16bit, mono channel

ses_audio.audio_load(audio_type='question')
try:
    ses_audio.audio_load(preload=False, std=True)
    ses_audio.audio_to_seg(opt_noise_level=5, opt_ET=51.5)
except FileNotFoundError:
    print('Std audio file does not exist.')
    ses_audio.audio_load(preload=False, std=False)
    ses_audio.audio_std()
    ses_audio.audio_load(preload=False, std=True)
    ses_audio.audio_to_seg(opt_noise_level=5, opt_ET=51.5)
# Process answer into segment:
# - 1. Denoise
# - 2. Seg
ses_audio.valid_seg()

ses_audio.audio_load(audio_type='question')

# ses_audio.qa_combine()

# import pdb;pdb.set_trace()

import pdb;pdb.set_trace()
