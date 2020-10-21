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
unit_col = 4

n_row = 6
n_col = 5


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)


def perm_test_one(xs, ys, nperm):
    n = xs.shape[0]
    true_diff = np.nanmean(xs) - np.nanmean(ys)

    min_t = np.minimum(true_diff, -true_diff)
    max_t = np.maximum(true_diff, -true_diff)
    p_left, p_right, p_both = 0, 0, 0
    zs = np.concatenate([xs, ys])

    nperm_zs = np.repeat(zs[:, np.newaxis], nperm, axis=1)
    shuffle_nperm_zs = shuffle_along_axis(nperm_zs, axis=0)

    mean1 = np.nanmean(shuffle_nperm_zs[:n], axis=0)
    mean2 = np.nanmean(shuffle_nperm_zs[n:], axis=0)
    t_perm = mean1 - mean2

    # add 1 because of emipiral sample to avoid pval=0
    # http://www.statsci.org/smyth/pubs/permp.pdf
    p_left = (t_perm < min_t).sum() + 1
    p_right = (t_perm > max_t).sum() + 1
    p_both = p_left + p_right - 1
    nperm += 1

    return p_left / nperm, p_right / nperm, p_both / nperm

def perm_test_one_old(xs, ys, nperm):
    n = xs.shape[0]
    true_diff = np.nanmean(xs) - np.nanmean(ys)

    min_t = np.minimum(true_diff, -true_diff)
    max_t = np.maximum(true_diff, -true_diff)
    p_left, p_right, p_both = 0, 0, 0
    zs = np.concatenate([xs, ys])
    for j in range(nperm):
        np.random.shuffle(zs)
        t_perm = np.nanmean(zs[:n]) - np.nanmean(zs[n:])
        p_left += t_perm < min_t
        p_right += t_perm > max_t
        p_both += np.abs(true_diff) < np.abs(t_perm)
    return p_left / nperm, p_right / nperm, p_both / nperm


metric = 'Fluency'  # Onset'' 'Fluency'  #
xlabel = 'Time / s'  # 'Answer duration / Baseline'  #

fig_type = 'pval_table'


def pval_table(cp_df, metric):
    print(1)

    pval_df_col = ['large_block', 'small_block', 'pval']
    # pval_df_list = []
    pval_df = pd.DataFrame(columns=pval_df_col)
    for idrow in range(4):
        subset_info = {'block': idrow}
        df_row = subset_df(cp_df, subset_info)
        df_row_val = df_row[metric].values

        for idcol in range(4):
            if idrow == idcol:
                pval_df = pval_df.append({'large_block': idrow,
                                          'small_block': idcol,
                                          'pval': 0.0}, ignore_index=True)
                continue
            subset_info = {'block': idcol}
            df_col = subset_df(cp_df, subset_info)
            df_col_val = df_col[metric].values

            p_l, p_r, p_two = perm_test_one(df_row_val, df_col_val, 10000)

            pval_df = pval_df.append({'large_block': idrow,
                                      'small_block': idcol,
                                      'pval': p_two}, ignore_index=True)
    new_pval_df = pval_df.pivot('large_block', 'small_block', 'pval')

    return new_pval_df
import pdb;pdb.set_trace()


one_level_df = drop_col(first_answer_df.copy())

for isubj in subject_list:
    print(isubj)

    if isubj != 0:
        subj_str = 'S' + str(isubj)
        subset_info = {'subject': isubj}
        select_df = subset_df(one_level_df.copy(), subset_info)
    else:
        subj_str = 'All_subjects'
        select_df = one_level_df.copy()

    fig = plt.figure(
        constrained_layout=True, figsize=[4*unit_col, 2*unit_row])

    gs = GridSpec(2, 4, figure=fig)

    pval_ax = []
    pval_ax.append(fig.add_subplot(gs[:, :2]))
    pval_ax.append(fig.add_subplot(gs[0, 2]))
    pval_ax.append(fig.add_subplot(gs[0, 3]))
    pval_ax.append(fig.add_subplot(gs[1, 2]))
    pval_ax.append(fig.add_subplot(gs[1, 3]))

    overall_stim_df = pval_table(select_df, metric=metric.lower())
    sns.heatmap(overall_stim_df, annot=True, ax=pval_ax[0])

    for id_f, f in enumerate([0, 4, 10, 40]):
        further_subset_info = {'freq': f}
        stim_df = subset_df(select_df.copy(), further_subset_info)

        spec_stim_df = pval_table(stim_df, metric=metric.lower())
        sns.heatmap(spec_stim_df, annot=True, ax=pval_ax[id_f+1])

    stim_list = ['All_stim', '0Hz', '4Hz', '10Hz', '40Hz']
    for id_ax, id_stim in zip(pval_ax, stim_list):

        id_ax.set_xlim([-0.5, 4.5])
        id_ax.set_ylim([-0.5, 4.5])
        id_ax.set_xticklabels(['pre_stim', 'stim_1', 'stim_2', 'post_stim'])
        id_ax.set_yticklabels(['pre_stim', 'stim_1', 'stim_2', 'post_stim'])
        id_ax.set_title(id_stim)
    title = '{0}_{1}_{2}'.format(metric, fig_type, subj_str)
    fig.suptitle(title)
    plt.savefig('{0}{1}_both.png'.format(fig_folder, title))

    plt.close(fig)


# plt.show()












import pdb;pdb.set_trace()
