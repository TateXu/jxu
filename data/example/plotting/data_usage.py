##=============================================================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2020-10-18 20:25:44
# Name       : plot_df_hist.py
# Version    : V1.0
# Description: analysis behavior performance using all_metric_df and hist plot
##=============================================================================

import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import f_oneway
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


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


def drop_col(input_df):
    input_df.columns = input_df.columns.droplevel()
    return input_df

# -----------------------------------------------------------------------------
# Load data containing all behavioral data
# -----------------------------------------------------------------------------
file_root = './'  # '/home/jxu/File/Data/NIBS/Stage_one/EEG/Exp/Data/'
dataframe_path = file_root + 'all_metric_df.pkl'

all_metric_df = pd.read_pickle(dataframe_path)

# -----------------------------------------------------------------------------
# Extract the data related to first or last answers
# -----------------------------------------------------------------------------
no_duplicate_col_name = [('META_INFO', 'subject'),
                         ('META_INFO', 'session'),
                         ('META_INFO', 'q_ind')]
# DF related to the first answer, size #subj*#ses*#trial=8*4*200=6400
first_answer_df = all_metric_df.drop_duplicates(
    subset=no_duplicate_col_name, keep='first', inplace=False)

# DF related to the last answer, size #subj*#ses*#trial=8*4*200=6400
last_answer_df = all_metric_df.drop_duplicates(
    subset=no_duplicate_col_name, keep='last', inplace=False)

# -----------------------------------------------------------------------------
# Example code for extracting interested subset of data
# -----------------------------------------------------------------------------
# Add the info about which subset you want to select, refer to the top dict
subset_info = {('META_INFO', 'block'): 0,
               ('META_INFO', 'subject'): 1,
               }
pre_s1_df = subset_df(first_answer_df, subset_info)

# Example for extracting the data as a numpy array
pre_s1_onset_val = pre_s1_df.METRICS.onset.values
# -----------------------------------------------------------------------------
# Plotting example using seaborn which nicely support dataframe
# -----------------------------------------------------------------------------

# Note that the multiple index key was allowed for seaborn in 0.9.0 but not
# anymore in the newest 0.11.0. Hence add a function to drop the first level

fig_folder = './figures/'
block_name_list = ['pre_stim', 'stim_1', 'stim_2', 'post']
block_val_list = [0, 1, 2, 3]

freq_val_list = [0, 4, 10, 40]
subject_val_list = [0, 1, 2, 3, 5, 6, 7, 8, 10]

metric = 'Onset'  # 'Fluency' 'Fluency'  #

# drop_col will drop the first level columns, i.e., META_INFO, METRICS, etc
hist_ax = sns.histplot(x=metric.lower(), data=drop_col(pre_s1_df), bins=20)
hist_ax.set_xlabel('Time / s')
hist_ax.set_xlim([0, 11])

import pdb;pdb.set_trace()
