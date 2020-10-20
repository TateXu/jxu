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

from jxu.data.preprocess import NIBSAudio

from statannot import add_stat_annotation
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


def subplot_generator(fig, rows, cols):
    gs = GridSpec(rows, cols, figure=fig)

    ax = np.empty((rows, cols), dtype='object')

    for irow in range(rows):
        for icol in range(cols):
            ax[irow, icol] = fig.add_subplot(gs[irow, icol])

            if irow >= 4:
                ax[irow, icol].set_xticks([])
                ax[irow, icol].set_yticks([])

    return gs, ax



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

first_answer_df_ = all_metric_df.loc[all_metric_df[('META_INFO', 'a_ind')] == 0]
assert first_answer_df_.equals(first_answer_df), 'Inconsistend first word dataframes extracted via two different ways'


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
# Plotting for results
# -----------------------------------------------------------------------------

# Note that the multiple index key was allowed for seaborn in 0.9.0 but not
# anymore in the newest 0.11.0. Hence add a function to drop the first level

fig_folder = './figures/'
block_list = ['pre_stim', 'stim_1', 'stim_2', 'post']

freq_list = ['Pooling', '0', '4', '10', '40']
subject_list = [0, 1, 2, 3, 5, 6, 7, 8, 10]

unit_row = 4
unit_col = 8

n_row = 6
n_col = 5

metric = 'Onset'  # 'Fluency' 'Fluency'  #
xlabel = 'Time / s'  # 'Answer duration / Baseline'  #

plt_sig_t = True
fig_type = 'hist'
suffix_sig_t = '_st' if plt_sig_t else ''

for isubj in subject_list:

    fig = plt.figure(
        constrained_layout=True, figsize=[n_col*unit_col, n_row*unit_row])
    hist_gs, hist_ax = subplot_generator(fig, n_row, n_col)

    for i in range(5):
        if i != 4:
            if isubj == 0:
                scale = 1
                subj_str = 'All_subjects'
                subset_info = {('META_INFO', 'block'): i,
                               }
            else:
                scale = 8
                subj_str = 'S' + str(isubj)
                subset_info = {('META_INFO', 'block'): i,
                               ('META_INFO', 'subject'): isubj,
                               }
            select_df = subset_df(first_answer_df, subset_info)
            sns.histplot(x=metric.lower(), data=drop_col(select_df),
                         ax=hist_ax[i, 0], bins=20)
            hist_ax[i, 0].set_title(block_list[i] + ' ' + freq_list[0])
            hist_ax[i, 0].set_xlabel(xlabel)

            if metric == 'Onset':
                hist_ax[i, 0].set_xlim([0, 11])
                hist_ax[i, 0].set_ylim([0, 280/scale])
            elif metric == 'Fluency':
                hist_ax[i, 0].set_xlim([0, 3])
                hist_ax[i, 0].set_ylim([0, 420/scale])
            for j in range(1, 5):
                further_subset_info = {'freq': int(freq_list[j]),
                                       }
                freq_select_df = subset_df(select_df, further_subset_info)
                sns.histplot(x=metric.lower(), data=freq_select_df,
                             ax=hist_ax[i, j], bins=20)
                hist_ax[i, j].set_title(
                    block_list[i] + '-' + freq_list[j] + 'Hz')
                hist_ax[i, j].set_xlabel(xlabel)
                if metric == 'Onset':
                    hist_ax[i, j].set_xlim([0, 11])
                    hist_ax[i, j].set_ylim([0, 70/scale])
                elif metric == 'Fluency':
                    hist_ax[i, j].set_xlim([0, 3])
                    hist_ax[i, j].set_ylim([0, 105/scale])
        else:

            if isubj == 0:
                metric_df_cp = first_answer_df.copy()
            else:
                subj_info = {('META_INFO', 'subject'): isubj}
                metric_df_cp = subset_df(first_answer_df.copy(), subj_info)

            violin_ax = fig.add_subplot(hist_gs[-2:, 1:])
            sns.boxplot(x='freq', y=metric.lower(), hue='block',
                        showfliers=False, data=drop_col(metric_df_cp),
                        ax=violin_ax)
            if plt_sig_t:
                box_pairs = [
                    ((0, 0), (0, 1)),
                    ((0, 0), (0, 2)),
                    ((0, 0), (0, 3)),
                    ((0, 3), (0, 1)),
                    ((0, 3), (0, 2)),
                    ((4, 0), (4, 1)),
                    ((4, 0), (4, 2)),
                    ((4, 0), (4, 3)),
                    ((4, 3), (4, 1)),
                    ((4, 3), (4, 2)),
                    ((10, 0), (10, 1)),
                    ((10, 0), (10, 2)),
                    ((10, 0), (10, 3)),
                    ((10, 3), (10, 1)),
                    ((10, 3), (10, 2)),
                    ((40, 0), (40, 1)),
                    ((40, 0), (40, 2)),
                    ((40, 0), (40, 3)),
                    ((40, 3), (40, 1)),
                    ((40, 3), (40, 2)),
                    ]

                add_stat_annotation(violin_ax, data=metric_df_cp, x='freq',
                                    y=metric.lower(), hue='block',
                                    box_pairs=box_pairs, test='t-test_ind',
                                    loc='inside', verbose=2,
                                    comparisons_correction=None,
                                    stats_params={'nan_policy': 'omit'})
            violin_ax.set_xticklabels([text + ' Hz' for text in freq_list[1:]])

    title = '{0}_{1}_{2}{3}'.format(metric, fig_type, subj_str, suffix_sig_t)
    fig.suptitle(title)
    plt.savefig('{0}{1}.png'.format(fig_folder, title))
    plt.close(fig)


# plt.show()












import pdb;pdb.set_trace()
