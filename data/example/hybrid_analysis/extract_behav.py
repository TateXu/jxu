#========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2021-05-09 15:08:43
# Name       : extract_behav.py
# Version    : V1.0
# Description: Joint analysis between rs and audio performance
#========================================
import pandas as pd

def subset_df(input_df, info_dict):

    temp_df = input_df.copy()
    for k, v in info_dict.items():
        temp_df = temp_df.loc[temp_df[k] == v]

    return temp_df

def drop_col(input_df):
    input_df.columns = input_df.columns.droplevel()
    return input_df

def behav(subject=1, session=0, metric='onset', loc='first'):
    file_root = '/home/jxu/File/Data/NIBS/Stage_one/EEG/Exp/Data/'
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

    if loc == 'first':
        to_extract_df = first_answer_df.copy()
    elif loc == 'last':
        to_extract_df = last_answer_df.copy()

    # -----------------------------------------------------------------------------
    # Example code for extracting interested subset of data
    # -----------------------------------------------------------------------------
    # Add the info about which subset you want to select, refer to the top dict
    behav_df_list = []
    behav_val_list = []
    for id_block in range(4):
        subset_info = {('META_INFO', 'block'): id_block,
                       ('META_INFO', 'session'): session,
                       ('META_INFO', 'subject'): subject,
                       }
        behav_df = subset_df(to_extract_df, subset_info)
        behav_val = behav_df.METRICS[metric].values

        behav_df_list.append(behav_df.METRICS)
        behav_val_list.append(behav_val)

    subset_info = {('META_INFO', 'session'): session,
                   ('META_INFO', 'subject'): subject,
                   }
    behav_df_list.append(subset_df(to_extract_df, subset_info))

    return behav_df_list, behav_val_list, to_extract_df


# val = behav(subject=1, session=1, metric='onset')



# fig_folder = './figures/'
# block_list = ['pre_stim', 'stim_1', 'stim_2', 'post', 'pooling']

# freq_list = ['Pooling', '0', '4', '10', '40']
# subject_list = [0, 1, 2, 3, 5, 6, 7, 8, 10]
# language_level = [7, 5, 2, 10, 6, 3, 8, 1]
# import pdb;pdb.set_trace()

# unit_row = 4
# unit_col = 8

# n_row = 6
# n_col = 5

# metric = 'Onset'  # 'Fluency' 'Fluency'  #
# xlabel = 'Time / s'  # 'Answer duration / Baseline'  #

# for i in range(5):

    # fig = plt.figure(figsize=[20, 10], num=i)
    # if i != 4:
        # subset_info = {'block': i}
        # select_df = subset_df(drop_col(first_answer_df.copy()), subset_info)
    # else:
        # select_df = drop_col(first_answer_df.copy())
    # # ax = sns.boxplot(x='subject', y=metric.lower(), hue='freq',
    # #                  order=language_level, data=select_df)
    # ax = sns.boxplot(hue='subject', y=metric.lower(), x='freq',
                     # hue_order=language_level, data=select_df)
    # ax.set_title(block_list[i])

    # # plt.savefig(metric.lower() + '_groupby_freq_' + str(i) + '_' + block_list[i] + '.png')
    # plt.savefig(metric.lower() + '_groupby_subject_' + str(i) + '_' + block_list[i] + '.png')
# import pdb;pdb.set_trace()


