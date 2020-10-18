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

for nr_subject in [1, 2, 3, 5, 6, 7, 8, 10]:
    for nr_ses in range(4):

        # =============================================================================
        # STEP 1: Load stadardised audio files: 44100sps, 16bit, mono channel
        # =============================================================================
        # For standardising audio file, use audio_std_batch.py

        ses_audio = NIBSAudio(subject=nr_subject, session=nr_ses)

        # =============================================================================
        # STEP 2: Process raw audio files into valid audio segments
        # =============================================================================

        # -> 1. Filtering: optimal level 5
        # -> 2. Segmentation
        # -> 3. Automatically detect segment based on amplitude: optimal ET - 51.5
        # -> 4. Manually detect bad segments
        # -> 5. Manually adjust onset and transcribe segment into text

        # RETURN
        # =============================================================================
        # |    Func    |    File    |    Var    |                Format               |
        # -----------------------------------------------------------------------------
        # |  audio_seg |  audio_folder/Marker/answer.pkl  | self.seg_marker |
        # [#trials] * [#segs, noise_flag, onset, onset+dur, ext_l, ext_r ]   |
        # =============================================================================

        # RETURN

        # -----------------------------------------------------------------------------
        # |    Func    |    File    |    Var    |                Format               |
        # -----------------------------------------------------------------------------
        # |  valid_seg | audio_folder/Marker/valid_answer.pkl | self.valid_seg_marker |
        # [#trials] * [noise_flag, loc of valid seg audio, onset, duration, text]   |
        # -----------------------------------------------------------------------------

        ses_audio.audio_load(audio_type='question')
        try:
            ses_audio.audio_load(preload=False, std=True)  # Step 1
            ses_audio.audio_to_seg(opt_noise_level=5, opt_ET=51.5)  # Step 2: P1-P4
        except FileNotFoundError:
            print('Std audio file does not exist. Following code works for single ' +
                  'session. For batch processing the audio files, please run ' +
                  'audio_std_batch.py first')
            ses_audio.audio_load(preload=False, std=False)
            ses_audio.audio_std()
            ses_audio.audio_load(preload=False, std=True)
            ses_audio.audio_to_seg(opt_noise_level=5, opt_ET=51.5)

        ses_audio.valid_seg()  # Step 2: P5


        ses_audio.metric_extractor()

        # ses_audio.qa_combine()

        # import pdb;pdb.set_trace()

import pdb;pdb.set_trace()
