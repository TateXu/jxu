#========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2020-11-26 13:23:30
# Name       : RS_plot.py
# Version    : V1.0 # Description: .
#========================================

import matplotlib.pyplot as plt

import mne
import pdb

from jxu.data.eeg_process import NIBSEEG
from jxu.data.utils import nibs_event_dict

import numpy as np
import pickle
import pandas as pd

from mne.decoding import CSP
from mne.viz import plot_topomap as topo

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
import seaborn as sns
import gc

def data_to_df(epoch, label):

    chn = epoch.info['ch_names']
    # n_chn = len(epoch.info['ch_names'])
    data = epoch.get_data()

    df = pd.DataFrame(data[0], index=chn, columns=range(data.shape[2]))
    df.index.name = 'Channel'

    tmp = df.reset_index().melt(id_vars='Channel', var_name='Time', value_name='Amplitude')

    return tmp.assign(label=[label]*len(tmp.index))


task_list = ['RS_close', 'RS_open',
            'QA_trial', 'QA_audio', 'QA_ans', 'QA_rec', 'QA_cen_word',
            'Arti_trial', 'Arti_action', 'Arti_rec']
stg_list = ['pre', 'post', 'stim_1', 'stim_2', 'all']

task = 'RS_close'
stg = 'all'
chn_picks = None #  ['Fz', 'Cz', 'Pz']
bands_id = 0

stg_id = stg_list.index(stg)
_, evt_dict, _, _ = nibs_event_dict()
try:
    evt_id, _, tmin, tmax = evt_dict[task]
except ValueError:
    evt_id, _ = evt_dict[task]
    tmin = -0.2
    tmax = 0.5
    print('==================================================================')
    print('tmin and tmax are not available for currect task: ' + task)
    print('Taking default value -0.2 and 0.5')
    print('==================================================================')

bands_list_name = ['BP(3-70)', 'delta(1-4)', 'theta(4-8)', 'Alpha(8-12)',
                   'L_Beta(12-20)', 'H_Beta(20-30)', 'L_Gamma(30-70)']
bands_list = [(3.0, 70.0), (1.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 20.0), (20.0, 30.0), (30.0, 70.0)]

path = '/home/jxu/File/Data/NIBS/Stage_one/EEG/Processed/RS_epoch/'
from matplotlib import gridspec
wratio = [1]
wratio.extend([6, 1] * 7)
hratio = [1, 6, 6, 6, 6]

stim_list = [0, 4, 10, 40]
stim_list_str = ['']
stim_list_str.extend([str(x)+' Hz' for x in stim_list])

fig_list = np.empty((9, ), dtype='object')
axes_list = np.empty((9, ), dtype='object')

for fig_id in range(9):
    fig, axes = plt.subplots(5, 15, figsize=(50, 25),
                             gridspec_kw={'width_ratios': wratio, 'height_ratios': hratio})
    for i in range(5):
        axes[i, 0].text(0.4, 0.5, stim_list_str[i], rotation='vertical', fontsize='xx-large', fontweight='heavy')
        axes[i, 0].set_axis_off()

    for j in range(7):
        axes[0, j*2+1].text(0.4, 0.5, bands_list_name[j], rotation='horizontal', fontsize='xx-large', fontweight='heavy')
        axes[0, j*2+1].set_axis_off()
        axes[0, j*2+2].set_axis_off()

    fig_list[fig_id] = fig
    axes_list[fig_id] = axes
no_gel = ['FC3', 'FFC5h', 'FCC5h', 'FFC3h', 'FCC3h', 'T7', 'FTT7h', 'TTP7h']


from jxu.data.utils import bad_chn_loader

all_RS_diff = np.empty((8, 4, 7, 126))


process = True
individual = True
plot_topo_flag = False

if individual:
    for id_sj, subj in enumerate([1, 2, 3, 5, 6, 7, 8, 10]):
        for ses in range(4):
            for bands_id, bands in enumerate(bands_list[:1]):
      #           if bands_id == 0:
                    # continue
                # if subj == 1 and ses > 0:
      #               import pdb;pdb.set_trace()


                ses_eeg = NIBSEEG(subject=subj, session=ses, bands=[bands])
                stim_freq = ses_eeg.subject_list[[*ses_eeg.subject_list.keys()][ses_eeg.subject]][ses_eeg.session]
                stim_id = stim_list.index(stim_freq)

                if process:
                    ses_eeg.raw_load()
                    ses_eeg.set_montage()
                    ses_eeg.data_concat(cp_flag=True)

                    ses_eeg.set_channels()
                    ses_eeg.rereference('average')
                    ses_eeg.raw_filter(notch=True)
                    events, event_id = mne.events_from_annotations(ses_eeg.raw_data_clean)

                    all_epoch = mne.Epochs(ses_eeg.data[0], events,
                                        event_id=[evt_id], tmin=tmin, tmax=tmax,
                                        picks=chn_picks,
                                        baseline=None, preload=True)
                    # all_epoch.drop_channels(all_epoch.info['bads'])
                    trial_index = np.hsplit(np.arange(len(all_epoch)), 4)
                    if stg_id == 4:
                        stg_epoch = [all_epoch[trial_index[epoch_id]] for epoch_id in range(4)]
                    else:
                        stg_epoch = all_epoch[trial_index[stg_id]]

                    with open('{0}S{1}_Ses{2}_{3}.pkl'.format(path, str(subj), str(ses), bands_list_name[bands_id]), 'wb') as f:
                        pickle.dump(stg_epoch, f)

                    del ses_eeg
                    gc.collect()
                else:
                    with open('{0}S{1}_Ses{2}_{3}.pkl'.format(path, str(subj), str(ses), bands_list_name[bands_id]), 'rb') as f:
                        tmp = pickle.load(f)
                    bad_dict = bad_chn_loader(subj, ses)
                    bad_channels = []
                    for bad in [*bad_dict.values()]:
                        bad_channels.extend(bad)
                    import pdb;pdb.set_trace()

                    tmp[0].info['bads'] = bad_channels
                    tmp[3].info['bads'] = bad_channels

                    tmp_pre = tmp[0].pick(picks='eeg')
                    tmp_post = tmp[3].pick(picks='eeg')

                    if plot_topo_flag:
                        pre_data = tmp_pre.get_data()[0]
                        post_data = tmp_post.get_data()[0]

                        if bands_id == 0:
                            diff = 1e6*np.mean(post_data, axis=1) - 1e6*np.mean(pre_data, axis=1)
                        else:
                            diff = 10*np.log(np.var(post_data, axis=1)) - 10*np.log(np.var(pre_data, axis=1))
                        all_RS_diff[id_sj, ses, bands_id] = diff
                        print(len(tmp_pre.info['bads']))

                        # tmp_pre.drop_channels(bad_channels)
                        # tmp_post.drop_channels(bad_channels)
                        # pre_data = tmp_pre.get_data()[0]
                        # post_data = tmp_post.get_data()[0]

                        print(len(tmp_pre.info['bads']))
                        if bands_id == 0:
                            diff = 1e6*np.mean(post_data, axis=1) - 1e6*np.mean(pre_data, axis=1)
                            axes_list[id_sj][stim_id+1, bands_id*2+2].set_title('uV')
                            cmin, cmax = -1.5, 1.5
                        else:
                            diff = 10*np.log(np.var(post_data, axis=1)) - 10*np.log(np.var(pre_data, axis=1))
                            axes_list[id_sj][stim_id+1, bands_id*2+2].set_title('dB')
                            cmin, cmax = -12.5, 12.5

                        im_tmp = topo(diff, tmp_pre.info, axes=axes_list[id_sj][stim_id+1, bands_id*2+1], show=False, vmin=cmin, vmax=cmax)
                        fig_list[id_sj].colorbar(im_tmp[0], cax=axes_list[id_sj][stim_id+1, bands_id*2+2], orientation='vertical')

                    del tmp_pre, tmp_post, tmp, ses_eeg


                print('S{0}_Ses{1}_Band{2}'.format(str(subj), str(ses), str(bands_id)))

        fig_list[id_sj].savefig('S{0}_rs_topo_offset_nodelete.jpg'.format(str(subj)))

    import pdb;pdb.set_trace()
    with open('{0}All_diff_bp.pkl'.format(path), 'wb') as f:
        pickle.dump(all_RS_diff, f)
    import pdb;pdb.set_trace()

    # import pdb;pdb.set_trace()
else:
    with open('{0}All_diff_bp.pkl'.format(path), 'rb') as f:
        mean_all = pickle.load(f)
    with open('{0}S{1}_Ses{2}_{3}.pkl'.format(path, str(1), str(1), bands_list_name[0]), 'rb') as f:
        tmp = pickle.load(f)
    import pdb;pdb.set_trace()

    bads = ['FC3', 'FFC5h', 'FCC5h', 'FFC3h', 'FCC3h', 'T7', 'FTT7h', 'TTP7h',
            'FT7', 'FC5', 'C5', 'C3', 'F9', 'F5', 'CCP5h', 'AFF5h', 'FC1',
            'P9', 'TP10', 'FFT7h', 'TP7', 'FFT9h', 'F3', 'FT9']
    info = tmp[0].pick(picks='eeg').info
    loc_chn = [info.ch_names.index(chn)  for chn in bads]

    select = mean_all

    # select = np.delete(mean_all, tuple(loc_chn), axis=3)
    # info = tmp[0].pick(picks='eeg').drop_channels(bads).info

    ses_eeg = NIBSEEG(subject=1, session=1)

    for freq_id, freq in enumerate([0, 4, 10, 40]):
        for bands_id, bands in enumerate(bands_list):
            tmp = np.empty((8, len(info.ch_names)))
            for id_sj, subj in enumerate([1, 2, 3, 5, 6, 7, 8, 10]):
                ses = ses_eeg.subject_list[[*ses_eeg.subject_list.keys()][subj]].index(freq)
                tmp[id_sj] = select[id_sj, ses, bands_id]

            if bands_id == 0:
                axes_list[-1][freq_id+1, bands_id*2+2].set_title('uV')
                cmin, cmax = -1.5, 1.5
            else:
                axes_list[-1][freq_id+1, bands_id*2+2].set_title('dB')
                cmin, cmax = -12.5, 12.5

            im_tmp = topo(tmp.mean(axis=0), info, axes=axes_list[-1][freq_id+1, bands_id*2+1], show=False, names=info['ch_names'], show_names=True,
                          vmin=cmin, vmax=cmax)
            fig_list[-1].colorbar(im_tmp[0], cax=axes_list[-1][freq_id+1, bands_id*2+2], orientation='vertical')
    fig_list[-1].savefig('All_rs_topo_offset_nodelete.pdf')

import pdb;pdb.set_trace()
#             # ----------- Get sensor plot --------------------

            # fig, ax = plt.subplots(1,1, figsize=(206,4))
            # df_pre = data_to_df(epoch[0], 'pre')
            # df_post = data_to_df(epoch[3], 'post')
            # all_df = pd.concat([df_pre, df_post], axis=0)
            # tax = sns.barplot(x='Channel', y='Amplitude', hue='label',
                                # data=all_df, ax=ax)
            # tax.set_title('Pre vs Post: S{0}_Ses{1}_{2}Hz'.format(str(subj), str(ses), str(stim_freq)))
            # tax.figure.savefig('./classification/sensor_barplot/S{0}_Ses{1}_{2}Hz.jpg'.format(str(subj), str(ses), str(stim_freq)))


