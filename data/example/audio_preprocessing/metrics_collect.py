##=============================================================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2020-10-18 18:30:25
# Name       : metrics_collect.py
# Version    : V1.0
# Description: collect performance data from session specific pickle file
##=============================================================================

from jxu.data.audio_process import NIBSAudio
import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import f_oneway
import pandas as pd

import seaborn as sns

def extractor(invar, ind):

    if isinstance(invar, list):
        return invar[ind]
    elif ind == 0:
        return invar
    else:
        raise ValueError('Check invar and ind')


col_name = [('META_INFO', 'subject'),
            ('META_INFO', 'session'),
            ('META_INFO', 'block'),
            ('META_INFO', 'q_ind'),
            ('META_INFO', 'a_ind'),
            ('META_INFO', 'q_ind_permanent'),
            ('META_INFO', 'freq'),
            ('QA_INFO', 'q_text'),
            ('QA_INFO', 'a_given_text'),
            ('QA_INFO', 'a_subject_text'),
            ('METRICS', 'onset'),
            ('METRICS', 'fluency'),
            ('METRICS', 'nr_blank'),
            ('SCORE', 'lexical'),
            ('SCORE', 'bert_prob'),
            ('SCORE', 'wordvec'),
            ]

file_root = '/home/jxu/File/Data/NIBS/Stage_one/EEG/Exp/Data/'
dataframe_path = file_root + 'all_metric_df.pkl'

import pdb;pdb.set_trace()
try:
    all_metric_df = pd.read_pickle(dataframe_path)
except FileNotFoundError:
    # all_metric_df = pd.DataFrame(columns=col_name)
    # all_metric_df.columns = pd.MultiIndex.from_tuples(all_metric_df.columns, names=['General','Detail'])
    # all_metric_df.to_pickle(dataframe_path)

    metric_df_list = []

    for nr_subject in [1, 2, 3, 5, 6, 7, 8, 10]:
        for nr_ses in range(4):
            ses_audio = NIBSAudio(subject=nr_subject, session=nr_ses)

            ses_audio.audio_load(audio_type='question')
            ses_audio.audio_load(preload=False, std=True)  # Step 1
            ses_audio.audio_to_seg(opt_noise_level=5, opt_ET=51.5)  # Step 2: P1-P4
            ses_audio.valid_seg()  # Step 2: P5
            ses_audio.metric_extractor()

            val_stim = list(ses_audio.subject_list.values())[nr_subject][nr_ses]
            nr_blank = 50 - ses_audio.metric[:, 0].reshape((4, -1)).sum(axis=1)

            for id_T in range(ses_audio.nr_qa_trial):

                print(str(nr_subject) + ' ' + str(nr_ses)+ ' ' + str(id_T))

                META_Q_INFO = ses_audio.qa_info.iloc[id_T]
                A_INFO = ses_audio.valid_seg_marker[id_T]
                METRIC = ses_audio.metric[id_T]

                id_block = divmod(id_T, 50)[0]

                if isinstance(METRIC[1], list):
                    nr_ans = len(METRIC[1])
                else:
                    nr_ans = 1

                for id_ans in range(nr_ans):

                    if A_INFO[2] != METRIC[1]:
                        import pdb;pdb.set_trace()

                    data = {('META_INFO', 'subject'): [ses_audio.subject],
                            ('META_INFO', 'session'): [ses_audio.session],
                            ('META_INFO', 'block'): [id_block],
                            ('META_INFO', 'q_ind'): [id_T],
                            ('META_INFO', 'a_ind'): [id_ans],
                            ('META_INFO', 'q_ind_permanent'): [META_Q_INFO.SENTENCE_INFO.permanent_index],
                            ('META_INFO', 'freq'): [val_stim],
                            ('QA_INFO', 'q_text'): [META_Q_INFO.SENTENCE_INFO.beeped_sen_content],
                            ('QA_INFO', 'a_given_text'): [META_Q_INFO.SENTENCE_INFO.beeped_word],
                            ('QA_INFO', 'a_subject_text'): [extractor(A_INFO[-1], id_ans)],
                            ('METRICS', 'onset'): [extractor(METRIC[1], id_ans)],
                            ('METRICS', 'fluency'): [extractor(METRIC[2], id_ans)],
                            ('METRICS', 'nr_blank'): [nr_blank[id_block]],
                            ('SCORE', 'lexical'): [0.0],
                            ('SCORE', 'bert_prob'): [0.0],
                            ('SCORE', 'wordvec'): [0.0],
                            }

                    metric_df_list.append(pd.DataFrame(data))


    all_metric_df = pd.concat(metric_df_list, ignore_index=True)

    import pdb;pdb.set_trace()
    # no_duplicate_col_name = [('META_INFO', 'sen_content'), ('SENTENCE_INFO', 'beeped_word')]
    # all_metric_df.drop_duplicates(subset=no_duplicate_col_name, keep='first', inplace=True)
    all_metric_df.columns = pd.MultiIndex.from_tuples(all_metric_df.columns, names=['Caps','Lower'])

    all_metric_df.to_pickle(dataframe_path)

import pdb;pdb.set_trace()
# =============================================================================
# STEP 1: Load stadardised audio files: 44100sps, 16bit, mono channel
# =============================================================================
# For standardising audio file, use audio_std_batch.py


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


import pdb;pdb.set_trace()
