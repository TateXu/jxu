##=============================================================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2020-08-29 12:55:41
# Name       : test_class.py
# Version    : V1.0
# Description: Minimal example test for the NIBSAudio class
##=============================================================================

import seaborn as sns

from jxu.data.preprocess import NIBSAudio
import numpy as np
from matplotlib import pyplot as plt

import pickle

from scipy.stats import f_oneway

def ttest(vec1, vec2):

    import numpy as np
    # https://en.wikipedia.org/wiki/Student%27s_t-test#Equal_or_unequal_sample_sizes,_similar_variances_'%22%60UNIQ--postMath-00000011-QINU%60%22'
    m1 = np.nanmean(vec1)
    m2 = np.nanmean(vec2)

    v1 = np.nanvar(vec1, ddof=1)
    v2 = np.nanvar(vec2, ddof=1)

    n1 = 50 - np.isnan(onset[0]).sum()
    n2 = 50 - np.isnan(onset[1]).sum()

    s_p = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2)/(n1 + n2 - 2))

    return (m1 - m2)/(s_p * np.sqrt(1/n1 + 1/n2))

# Initialize the instance for audio cleaning

def map_stim(ses):
    stim_list = [0, 4, 10, 40]

    return stim_list.index(ses)


def anova(mat):

    val_list = []
    clean_vec = lambda x: x[~ np.isnan(x)]
    for row in range(mat.shape[0]):
       val_list.append(clean_vec(mat[row]))

    try:
        stat, pval = f_oneway(*val_list)
    except:
        import pdb;pdb.set_trace()

    return stat, pval


def first_onset(onset_list, nr_flag):

    ind = 0 if nr_flag == 'first' else -1

    return np.asarray(
        [sg_onset if not isinstance(sg_onset, list) else sg_onset[ind] for sg_onset in onset_list])



def perm_test(xs, ys, nperm):
    n, k = len(xs), 0
    diff = np.abs(np.nanmean(xs) - np.nanmean(ys))
    zs = np.concatenate([xs, ys])
    for j in range(nperm):
        np.random.shuffle(zs)
        k += diff < np.abs(np.nanmean(zs[:n]) - np.nanmean(zs[n:]))
    return k / nperm


name_list = ['pre', 'stim_1', 'stim_2', 'post']
name_list_vec = []
for i in range(4):
    for j in range(i + 1, 4):
        name_list_vec.append( '{0} vs {1}'.format(name_list[i], name_list[j]))



nr_flag = 'last'
binary_pval = np.empty((4, 4, 4))
tri_pval = np.empty((4, 3, 3))
all_pval = np.empty((6, 4))


force_flag = True
square_plot = False

box_plot = True
diag_ind = [1, 2, 3, 6, 7, 11]


subject_list = [1, 2, 3, 5, 6, 7]
box_pval_mat = np.zeros((4, len(subject_list), 6))
run_data = False
if run_data:
    for ind_nr_subject, nr_subject in enumerate(subject_list):
        print(nr_subject)
        if square_plot:
            fig, axe = plt.subplots(2, 2)

        binary_pval = np.empty((4, 4, 4))
        tri_pval = np.empty((4, 3, 3))

        for nr_ses in range(4):
            ses_audio = NIBSAudio(subject=nr_subject, session=nr_ses)

            # Load stadardised audio files: 44100sps, 16bit, mono channel

            ses_audio.audio_load(audio_type='question')
            ses_audio.audio_load(preload=False, std=True)
            ses_audio.audio_to_seg(opt_noise_level=5)
            ses_audio.valid_seg()

            # Significant test for onset
            ori_onset = ses_audio.valid_seg_marker[:, 2]
            onset_vec = first_onset(ori_onset, nr_flag=nr_flag)
            onset = onset_vec.copy().reshape(4, 50)

            onset[onset==None] = np.nan
            onset = onset.astype('float64')

            mean_vec = np.nanmean(onset, axis=1)
            var_vec = np.nanvar(onset, axis=1)


            blank = np.isnan(onset).sum(axis=1)
        #     for i in range(4):
                # plt.subplot(4, 1, i + 1)
                # plt.hist(onset[i], bins=20)
        #         plt.xlim([-0.05, np.nanmax(onset)])
            all_pval[ind_nr_subject, nr_ses] = anova(onset)[1]
            try:
                if not force_flag:
                    with open('Pair_pval_{0}.pkl'.format(nr_flag), 'rb') as f:
                        binary_pval = pickle.load(f)
                else:
                    raise ValueError
            except:
                for i in range(4):
                    for j in range(i + 1, 4):
                        # binary_pval[nr_ses, i, j] = anova(np.vstack((onset[i], onset[j])))[1]
                        binary_pval[nr_ses, i, j] = perm_test(
                            onset[i], onset[j], 10000)

            val_stim = list(ses_audio.subject_list.values())[nr_subject][nr_ses]

            if box_plot:

                stim_ind = map_stim(val_stim)
                box_pval_mat[stim_ind, ind_nr_subject] = binary_pval[nr_ses].reshape((16,))[diag_ind]


            if square_plot:
                ax = axe[divmod(map_stim(val_stim), 2)]
                im = ax.imshow(binary_pval[nr_ses])
                ax.set_xticks(np.arange(len(name_list)))
                ax.set_yticks(np.arange(len(name_list)))
                ax.set_xticklabels(name_list)
                ax.set_yticklabels(name_list)
                for i in range(len(name_list)):
                    for j in range(len(name_list)):
                        text = ax.text(j, i, '{:.2f}'.format(binary_pval[nr_ses, i, j]), ha="center", va="center", color="w")

                ax.set_title('Session {0} ({1} Hz)'.format(str(nr_ses), str(val_stim)))
                ax.set_xlim([-0.5, 3.5])
                ax.set_ylim([-0.5, 3.5])

            # DF = 98 - blank_1 - blank_2
        plt.tight_layout()

        name = list(ses_audio.subject_list.keys())[nr_subject]

        if square_plot:
            plt.savefig('Subject_{0}_{1}.pdf'.format(name, nr_flag))
    import pdb;pdb.set_trace()
    with open('All_pair_pval_{0}.pkl'.format(nr_flag), 'wb') as f_out:
        pickle.dump(box_pval_mat, f_out)

else:
    with open('All_pair_pval_{0}.pkl'.format(nr_flag), 'rb') as f:
        box_pval_mat = pickle.load(f)
hist_plot = True

if hist_plot:
    temp = NIBSAudio(subject=1, session=1)
    stimval = [0, 4, 10, 40]


    all_onset = np.empty((4, 4, 50, 6), dtype='object')


    fig, axe = plt.subplots(4, 4, figsize=(40,20))
    for nr_stim, val_ses in enumerate(stimval):

        for ind_nr_subject, nr_subject in enumerate(subject_list):
            print(nr_subject)

            true_ses = list(temp.subject_list.values())[nr_subject].index(val_ses)

            ses_audio = NIBSAudio(subject=nr_subject, session=true_ses)

            # Load stadardised audio files: 44100sps, 16bit, mono channel

            ses_audio.audio_load(audio_type='question')
            ses_audio.audio_load(preload=False, std=True)
            ses_audio.audio_to_seg(opt_noise_level=5)
            ses_audio.valid_seg()


            # Significant test for onset
            ori_onset = ses_audio.valid_seg_marker[:, 2]
            onset_vec = first_onset(ori_onset, nr_flag=nr_flag)
            onset = onset_vec.copy().reshape(4, 50)

            onset[onset==None] = np.nan
            onset = onset.astype('float64')
            all_onset[nr_stim, :,:,ind_nr_subject] = onset.copy()



        # for ind_title, title in enumerate(name_list):

            # hist_ax = axe[ind_title, nr_stim]

            # data = all_onset[nr_stim, ind_title].flatten().astype('float64')
            # hist_ax.hist(data[~np.isnan(data)], orientation='vertical', bins=20)
            # hist_ax.set_title(title + '{0}Hz'.format(val_ses))
            # hist_ax.set_xlim([0, 12])
            # hist_ax.set_ylim([0, 70])

    # plt.tight_layout()
    # plt.savefig('All_Stim_{0}.pdf'.format(nr_flag))

    all_stat_data = np.empty((16, 300))
    all_stat_name = []

    for nr_stim, val_ses in enumerate(stimval):
        for ind_title, title in enumerate(name_list):
            all_stat_data[nr_stim * 4 + ind_title] = all_onset[nr_stim, ind_title].flatten().astype('float64')
            all_stat_name.append('{0}_{1}Hz'.format(title, str(val_ses)))

#     with open('all_onset_{0}.pkl'.format(nr_flag), 'wb') as f_in:
        # pickle.dump(all_onset, f_in)
    # import pdb;pdb.set_trace()
    all_stat_data = all_stat_data.astype('float64')
    all_sq_plot = True
    all_binary_pval = np.zeros((16, 16))
    if all_sq_plot:
        with open('pooling_pval.pkl', 'rb') as f_load:
            all_binary_pval = pickle.load(f_load)
        # for i in range(16):
            # for j in range(i + 1, 16):
                # print(i)
                # print(j)
                # # binary_pval[nr_ses, i, j] = anova(np.vstack((onset[i], onset[j])))[1]
                # all_binary_pval[i, j] = perm_test(
                    # all_stat_data[i], all_stat_data[j], 10000)


        fig, ax = plt.subplots(1, 1, figsize=(20,20))
        im = ax.imshow(all_binary_pval)
        ax.set_xticks(np.arange(len(all_stat_name)))
        ax.set_yticks(np.arange(len(all_stat_name)))
        ax.set_xticklabels(all_stat_name)
        ax.set_yticklabels(all_stat_name)
        for i in range(len(all_stat_name)):
            for j in range(len(all_stat_name)):
                text = ax.text(j, i, '{:.2f}'.format(all_binary_pval[i, j]), ha="center", va="center", color="w")

        ax.set_title('Pooling pval')
        ax.set_xlim([-0.5, 15.5])
        ax.set_ylim([-0.5, 15.5])
    import pdb;pdb.set_trace()
    plt.savefig('all.pdf')
    plt.show()





if box_plot:
    fig, axe = plt.subplots(2, 4, figsize=(20, 10))
    stimval = [0, 4, 10, 40]

    for nr_stim in range(4):

        row, col = divmod(nr_stim, 2)
        ax = axe[divmod(nr_stim, 2)]
        hist_ax = axe[row, col*2]
        box_ax = axe[row, col*2+1]

        hist_ax.hist(box_pval_mat[nr_stim].flatten(), orientation='horizontal', bins=20)
        box_ax.boxplot(box_pval_mat[nr_stim])

        box_ax.set_xticks(1+np.arange(len(name_list_vec)))
        box_ax.set_xticklabels(name_list_vec,  rotation=45)

        box_ax.set_title('Stim: {0}Hz'.format(str(stimval[nr_stim])))

    plt.tight_layout()

    plt.savefig('Joint_{0}.pdf'.format(nr_flag))
    plt.show()




with open('Pair_pval_{0}.pkl'.format(nr_flag), 'wb') as f:
    pickle.dump(binary_pval, f)


import pdb;pdb.set_trace()
