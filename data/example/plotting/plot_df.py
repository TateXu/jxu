##=============================================================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2020-10-18 20:25:44
# Name       : plot_df.py
# Version    : V1.0
# Description: analysis behavior performance using all_metric_df
##=============================================================================

import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import f_oneway
from matplotlib import pyplot as plt
from jxu.data.preprocess import NIBSAudio
"""
Index information of all_metric_df.

MultiIndex([('META_INFO',         'subject'),  # 1, 2, 3, 5, 6, 7, 8, 10
            ('META_INFO',         'session'),  # 0-3
            ('META_INFO',           'block'),  # 0-3: pre, stim_1, stim_2, post
            ('META_INFO',           'q_ind'),  # q ind within session, 0-199
            ('META_INFO',           'a_ind'),  # First/Second/.../Last answer
            ('META_INFO', 'q_ind_permanent'),  # Unmodifiable index, i.e. 0-799
            ('META_INFO',            'freq'),  # session_specific, 0, 4, 10, 40
            (  'QA_INFO',          'q_text'),  # question text
            (  'QA_INFO',    'a_given_text'),  # expected answer
            (  'QA_INFO',  'a_subject_text'),  # subj's answer(s), can be list
            (  'METRICS',           'onset'),  # reaction time
            (  'METRICS',         'fluency'),  # dur(subj's answer)/baseline
            (  'METRICS',        'nr_blank'),  # #blank answer, block specific
            (    'SCORE',         'lexical'),  # scoring via synonym dict.
            (    'SCORE',       'bert_prob'),  # scoring via BERT pred. prob.
            (    'SCORE',         'wordvec')], # scoring via vec. dist.
           names=['Caps', 'Lower'])
"""


def subset_df(input_df, info_dict):

    temp_df = input_df.copy()
    for k, v in info_dict.items():
        temp_df = temp_df.loc[temp_df[k] == v]

    return temp_df


file_root = './'  # '/home/jxu/File/Data/NIBS/Stage_one/EEG/Exp/Data/'
dataframe_path = file_root + 'all_metric_df.pkl'

all_metric_df = pd.read_pickle(dataframe_path)

no_duplicate_col_name = [('META_INFO', 'subject'),
                         ('META_INFO', 'session'),
                         ('META_INFO', 'q_ind')]
# DF related to the first answer, size #subj*#ses*#trial=8*4*200=6400
first_answer_df = all_metric_df.drop_duplicates(
    subset=no_duplicate_col_name, keep='first', inplace=False)

# DF related to the last answer, size #subj*#ses*#trial=8*4*200=6400
last_answer_df = all_metric_df.drop_duplicates(
    subset=no_duplicate_col_name, keep='last', inplace=False)

first_answer_df_ = all_metric_df.loc[all_metric_df[('META_INFO', 'a_ind')] == 0]
assert first_answer_df_.equals(first_answer_df), 'Inconsistend first word dataframes extracted via two different ways'

# Add the info about which subset you want to select, refer to the top dict
subset_info = {('META_INFO', 'block'): 0,
               ('META_INFO', 'subject'): 1,
               }
pre_df = subset_df(first_answer_df, subset_info)

import pdb;pdb.set_trace()
