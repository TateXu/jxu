from jxu.data.loader import *
import numpy as np
from numpy.fft import fft, fftfreq
from scipy import signal
import matplotlib.pyplot as plt
from itertools import product
from mne.viz import plot_epochs_image
import platform

import mne

import warnings
import pickle
from jxu.data.utils import *
from jxu.viz.utils import *
import argparse
import pdb


parser = argparse.ArgumentParser(description='Processing NIBS data.', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-lr', '--load_raw', action='store_true', help='Property: flag;\nDeafult=False;\nFunc: Loading completely raw data;')
parser.add_argument('-sc', '--save_clean', action='store_true', help='Property: flag;\nDeafult=False;\nFunc: Saving filtered& cropped raw data;')
parser.add_argument('-lc', '--load_clean', action='store_true', help='Property: flag;\nDefault=False;\nFunc: Loading filtered& cropped raw data;')
parser.add_argument('-ec', '--epoch_clean', action='store_true', help='Property: flag;\nDeafult=False;\nFunc: Epoching filtered& cropped raw data;')
parser.add_argument('-erp', '--erp_flag', action='store_true', help='Property: flag;\nDeafult=False;\nFunc: Epoching/Processing ERP signal when True;')
parser.add_argument('-p', '--plt_flag', action='store_true', help='Property: flag;\nDeafult=False;\nFunc: Plotting figures when True;')
local_args = parser.parse_args()
locals().update(vars(local_args))


nr_events_predefined, event_dict, label_dict, event_dict_expand = nibs_event_dict()

if platform.system() == 'Linux':
    path = '/home/jxu/File/Data/NIBS/Stage_one/EEG/ZWS/ZWS_SESSION_1/'  
elif platform.system() == 'Darwin':
    path = '/Users/xujiachen/File/Data/NIBS/Stage_one/ZWS/ZWS_SESSION_1/'


load_audio_para = True
sort_eeg_data = True
if load_audio_para:
    folder_path = path + 'Audio_Recording/Exp_data/'

    with open(folder_path + 'Valid_segs/onset_list.pkl', 'rb') as f:
        onset_list = pickle.load(f)
    with open(folder_path + 'Valid_segs/duration_list.pkl', 'rb') as f:
        duration_list = pickle.load(f)
    duration_list = np.asarray(duration_list)
    onset_list = np.asarray(onset_list)
    del folder_path

    onset_sort_ind = []
    for i in range(4):
        temp = onset_list[40 * i: 40 * i + 40]
        no_answer_ind = np.where(temp==None)[0]
        answer_ind = np.where(temp[:, 1]!= None)[0]

        onset_answer = temp[answer_ind]
        sorted_onset_answer = np.vstack((onset_answer[np.argsort(onset_answer[:, 1])], temp[no_answer_ind]))
        onset_sort_ind.append((sorted_onset_answer[:,0] - 40 * i).tolist() )

if load_raw:
    raw = eeg_loader(subject=0, session=1)

    filters = [(0.1, 90)]

    trigger_detector(raw, event_dict=event_dict,
                     event_dict_expand=event_dict_expand,
                     nr_events_predefined=nr_events_predefined)

    fmin, fmax = filters[0]
    # filter data

    """
    1. Filter type (high-pass, low-pass, band-pass, band-stop, FIR, IIR)
    2. Cutoff frequency (including definition)
    3. Filter order (or length)
    4. Roll-off or transition bandwidth
    5. Passband ripple and stopband attenuation
    6. Filter delay (zero-phase, linear-phase, non-linear phase) and causality
    7. Direction of computation (one-pass forward/reverse, or two-pass forward and reverse)


    A non-causal, i.e, two-pass forward and reverse, 5th order Butterworth BP filter between 0.1Hz to 90Hz with transition band 0.1Hz (L) and 90*0.25=18Hz (H). 
    """

    raw.set_channel_types({'Audio': 'stim', 'tACS': 'stim'})
    print('------- Start filtering-------')
    iir_params = dict(order=5, ftype='butter')  # , rp=1.
    raw_f = raw.copy().filter(l_freq=fmin, h_freq=fmax, method='iir', iir_params=iir_params, verbose=False)
    # raw_f = raw.copy().filter(fmin, fmax, method='fir', verbose=False, fir_design='firwin')
    raw_f.notch_filter(freqs=np.arange(50, 99, 50), picks='eeg')

    print('------- Setting Montage -------')

    raw_f.set_channel_types({'EOG151': 'eog', 'EOG152': 'eog'})
    montage = mne.channels.read_montage("standard_1005")
    raw_f.rename_channels({'O9': 'I1', 'O10': 'I2'})
    raw_f.set_montage(montage)
    raw_f.rename_channels({'I1': 'O9', 'I2': 'O10'})
    # raw_f.info['bads'] = []
    # raw_clean.set_channel_types({'Audio': 'stim', 'tACS': 'stim'})

    # Excluding criteria: White noise or not, i.e., a flat spectrum?
    raw_f.set_channel_types({'TP7': 'stim', 'TP8': 'stim',
                             'TTP7h': 'stim', 'TTP8h': 'stim',
                             'TPP7h': 'stim', 'TPP8h': 'stim',
                             'T8': 'stim', 'TPP9h': 'stim',
                             'P7': 'stim', 'AFp1': 'stim'})

    print('------- Removing common average -------')
    raw_ca = raw_f.copy().set_eeg_reference(ref_channels='average')

    # ori_data_eog = raw_f.get_data()[-2:,:]
    # ori_data = raw_f.pick(picks='eeg').get_data()
    # ca_data_eog = raw_ca.get_data()[-2:,:]
    # ca_data = raw_ca.copy().pick(picks='eeg').get_data()
    # pdb.set_trace()
    print('------- Croping and concatenating raw files -------')
    data_run_0_1 = raw_ca[:, : 2446629]
    data_run_2 = raw_ca[:, 2578697:3503294]
    data_run_3 = raw_ca[:, 3552652:]

    ts_run_0_1 = data_run_0_1[1]
    ts_run_2 = data_run_2[1]
    ts_run_3 = data_run_3[1]

    raw_run_0_1 = raw_ca.copy().crop(
        tmin=ts_run_0_1[0], tmax=ts_run_0_1[-1], include_tmax=True)
    raw_run_2 = raw_ca.copy().crop(
        tmin=ts_run_2[0], tmax=ts_run_2[-1], include_tmax=True)
    raw_run_3 = raw_ca.copy().crop(
        tmin=ts_run_3[0], tmax=ts_run_3[-1], include_tmax=True)

    raw_clean = raw_run_0_1.copy()
    raw_clean.append([raw_run_2, raw_run_3])
    trigger_detector(raw_clean)

if save_clean:

    print('------- Saving the raw data into pickle files -------')
    max_file_size = 1.8
    save_filename = path + '/raw_clean_' 
    raw_size = 8 * len(raw_clean) * raw_clean.info['nchan'] / (1024**3)
    nr_save_raw = int(np.ceil(raw_size / max_file_size))

    hard_limit = int(max_file_size *  (1024**3) / raw_clean.info['nchan'] / 8)

    for save_ind in range(nr_save_raw):
        start_ind = hard_limit * save_ind
        end_ind = start_ind + hard_limit if save_ind != nr_save_raw - 1 else len(raw_clean)
        data_save = raw_clean[:, start_ind: end_ind]
        ts_save = data_save[1]
        raw_seg = raw_clean.copy().crop(tmin=ts_save[0], tmax=ts_save[-1], include_tmax=True)

        with open(save_filename + str(save_ind) + '.pkl', 'wb') as f:
            pickle.dump(raw_seg, f)
        print(str(save_ind + 1) + '/' + str(nr_save_raw) + ' saved!')    

if load_clean:
    print('------- Loading the raw data from pickle files -------')
    nr_load_raw = 3
    load_filename = path + '/raw_clean_'
    for load_ind in range(nr_load_raw):
        print('Loading ' + str(load_ind + 1) + '/' + str(nr_load_raw) + '!')    
        with open(load_filename + str(load_ind) + '.pkl', 'rb') as f:
            temp = pickle.load(f)
        if load_ind == 0:
            all_raw_data = temp
        else:
            all_raw_data.append(temp)
    del temp
    print('Loading completed!')
    all_raw_data.annotations.delete(
        np.concatenate(
            (np.where(all_raw_data.annotations.description == 'BAD boundary')[0],
             np.where(all_raw_data.annotations.description == 'EDGE boundary')[0])))

    trigger_detector(all_raw_data, event_dict=event_dict,
                     event_dict_expand=event_dict_expand,
                     nr_events_predefined=nr_events_predefined)
    raw_clean = all_raw_data
    # 'BAD boundary', 'EDGE boundary'
filter_epoch = True
if epoch_clean:

    print('------- Epoching the raw files -------')
    

    if not erp_flag:
        # ----- Trialwise data extraction -----------
        epoch_list = {'Cali_trial_start': [30, 20],
                      'RS_intro_start': [190, 8],
                      'QA_trial_start': [30, 160]}
    else:
        # -------- ERP data extraction --------------
        epoch_list = {'Cali_display_start': [2.2, 20],
                      'Cali_rec_start': [2.2, 20],
                      'QA_audio_start': [2.2, 160],
                      'QA_rec_start': [2.2, 160],
                      'QA_cen_word_start': [2.2, 160],
                      'QA_ans_start': [2.2, 160]}
    X = {}
    events_clean, event_clean_id = mne.events_from_annotations(raw_clean)
    
    pdb.set_trace()
    for ind, (key, val) in enumerate(epoch_list.items()):
        
        if erp_flag:
            epochs = mne.Epochs(raw_clean, events_clean,
                    event_id=[event_dict_expand[key]],
                    tmin=-0.2, tmax=val[0], baseline=None, proj=False,
                    preload=True, verbose=False, on_missing='ignore')
            if filter_epoch:
                filtered_epochs = epochs.copy().filter(l_freq=None, h_freq=15.0)
            else:
                filtered_epochs = epochs.filter(l_freq=None, h_freq=15.0)
            X[key[:-6]] = []
            if 'Cali' in key:
                for nr_run in range(2):
                    tmp = filtered_epochs[10 * nr_run: 10 * (nr_run + 1)]
                    X[key[:-6]].append(tmp)
                del tmp
            elif 'QA' in key:
                for nr_run in range(4):
                    tmp = filtered_epochs[40 * nr_run: 40 * (nr_run + 1)]
                    X[key[:-6]].append(tmp)
                del tmp

        else:
            epochs = mne.Epochs(raw_clean, events_clean,
                                event_id=[event_dict_expand[key]],
                                tmin=0, tmax=val[0], baseline=None, proj=False,
                                preload=True, verbose=False, on_missing='ignore')

            if len(epochs) != val[1]:
                pdb.set_trace()
                raise ValueError('Number of epochs is not consistent with the number of predefined number!')
            event_end = np.where(events_clean[:, 2] == event_dict_expand[key] + 1)[0]
            epoch_annot = []
            for evt_ind, evt_val in enumerate(zip(epochs.selection, event_end)):
                epoch_evt_tmp = events_clean[evt_val[0]: evt_val[1] + 1]
                epoch_annot.append(np.hstack((np.asarray([epoch_evt_tmp[:, 0] - epoch_evt_tmp[0, 0]]).T, epoch_evt_tmp)))

            epochs.all_annot = epoch_annot

            if len(epochs.all_annot) != len(epochs):
                raise ValueError("Unconsistent number of annot and epochs")

            if key == 'Cali_trial_start':
                X['Calibration'] = []
                for nr_run in range(2):
                    tmp = epochs[10 * nr_run: 10 * (nr_run + 1)]
                    tmp.run_annot = epochs.all_annot[10 * nr_run: 10 * (nr_run + 1)]
                    X['Calibration'].append(tmp)
                del tmp
            elif key == 'RS_intro_start':
                X['RS'] = {}
                rs_open_start = events_clean[np.where(
                    events_clean[:, 2] == 32)[0], :]
                rs_close_start = events_clean[np.where(
                    events_clean[:, 2] == 34)[0], :]

                default_rs_order = ['open', 'close']
                rs_order = []
                [rs_order.extend(default_rs_order[::ind]) for ind in np.sign((rs_close_start - rs_open_start)[:, 0])]

                for state in default_rs_order:
                    X['RS'][state] = epochs[np.asarray(rs_order) == state]
            elif key == 'QA_trial_start':
                X['QA'] = []
                for nr_run in range(4):
                    tmp = epochs[40 * nr_run: 40 * (nr_run + 1)]
                    tmp.run_annot = epochs.all_annot[40 * nr_run: 40 * (nr_run + 1)]
                    X['QA'].append(tmp)
                del tmp

    if erp_flag:
        pdb.set_trace()
        with open(path + '/data_qa_erp.pkl', 'wb') as f:
            pickle.dump(X, f)
        print('saved!')
    else:
        pdb.set_trace()
        with open(path + '/data_w_annot.pkl', 'wb') as f:
            pickle.dump(X, f)
        print('saved!')

if erp_flag:
    with open(path + '/data_qa_erp.pkl', 'rb') as f:
        X = pickle.load(f)

    if sort_eeg_data and onset_sort_ind is not None:

        for key in [*X.keys()]:
            if 'QA' not in key:
                continue
            else:
                new_key = key + '_sorted'
                X[new_key] = []
                X[new_key + '_onset'] = []
                for i in range(4):
                    X[new_key].append(X[key][i][onset_sort_ind[i]])
                    X[new_key + '_onset'].append(onset_sort_ind[i])
else:
    with open(path + '/data_w_annot.pkl', 'rb') as f:
        X = pickle.load(f)

    if sort_eeg_data and onset_sort_ind is not None:
        X['QA_sorted'] = []
        for i in range(4):
            X['QA_sorted'].append(X['QA'][i][onset_sort_ind[i]])
pdb.set_trace()
# 'QA_audio_sorted', 'QA_rec_sorted', 'QA_cen_word_sorted', 'QA_ans_sorted'
# X['QA_rec_sorted'][0].plot_image(piegncks=['T7'])
# plot_psd_topomap
# plot_topo_image


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



run = 2




for run in range(4):
    fig = plt.figure(figsize=(16, 16))
    outer = gridspec.GridSpec(3, 4, wspace=0.2, hspace=0.2)
    outer_list = []
    for i in range(12):
        inner = gridspec.GridSpecFromSubplotSpec(2, 2,
                        subplot_spec=outer[i], wspace=0.1, hspace=0.1, width_ratios=[4, 1])
        inner_ax = []
        for j in [0, 2, 1]:
            ax = plt.Subplot(fig, inner[j])
            fig.add_subplot(ax)
            inner_ax.append(ax)
        outer_list.append(inner_ax)
    for id_name, name in enumerate(['audio', 'rec', 'cen_word']):
        chn_list = ['Pz', 'CPz', 'Cz', 'Fz']
        for id_chn, chn in enumerate(chn_list):
            fig_title = name + '_' + chn + '_all_' + str(run)
            X['QA_' + name + '_sorted'][run].plot_image(picks=[chn], title=fig_title, show=False, axes=outer_list[id_name*4 + id_chn])[0]  # No response
            
    fig.savefig('Results/a_all_run_' + str(run) + '.pdf')


import pdb 
pdb.set_trace()



for run in range(4):
    for name in ['audio', 'rec', 'cen_word']:
        fig = plt.figure(figsize=(16, 16))
        outer = gridspec.GridSpec(3, 4, wspace=0.2, hspace=0.2)
        outer_list = []
        for i in range(12):
            inner = gridspec.GridSpecFromSubplotSpec(2, 2,
                            subplot_spec=outer[i], wspace=0.1, hspace=0.1, width_ratios=[4, 1])
            inner_ax = []
            for j in [0, 2, 1]:
                ax = plt.Subplot(fig, inner[j])
                fig.add_subplot(ax)
                inner_ax.append(ax)
            outer_list.append(inner_ax)
        chn_list = ['Pz', 'CPz', 'Cz', 'Fz']
        for id_chn, chn in enumerate(chn_list):
            fig_title = name + '_' + chn + '_fastest_answer_' + str(run)
            X['QA_' + name + '_sorted'][run][:5].plot_image(picks=[chn], title=fig_title, show=False, axes=outer_list[id_chn])[0]  # No response

            fig_title = name + '_' + chn + '_no_answer_' + str(run)
            X['QA_' + name + '_sorted'][run][-5:].plot_image(picks=[chn], title=fig_title, show=False, axes=outer_list[len(chn_list) + id_chn])[0]  # No response

            fig_title = name + '_' + chn + '_all_' + str(run)
            X['QA_' + name + '_sorted'][run].plot_image(picks=[chn], title=fig_title, show=False, axes=outer_list[len(chn_list)*2 + id_chn])[0]  # No response
        fig.savefig('Results/a_' + name + '_run_' + str(run) + '.pdf')

import pdb 
pdb.set_trace()










"""
run = 2
for run in range(4):
    for name in ['audio', 'rec', 'cen_word']:
        for id_chn, chn in enumerate(['Pz', 'CPz', 'Cz', 'Fz']):
            fig_title = name + '_' + chn + '_fastest_answer_' + str(run)
            fig = X['QA_' + name + '_sorted'][run][:5].plot_image(picks=[chn], title=fig_title, show=False, ax=outer_list[id_chn*4])[0]  # No response
            fig.savefig('Results/a_' + fig_title + '.pdf')
            fig_title = name + '_' + chn + '_no_answer_' + str(run)
            fig = X['QA_' + name + '_sorted'][run][-5:].plot_image(picks=[chn], title=fig_title, show=False, ax=outer_list[id_chn*4])[0]  # No response
            fig.savefig('Results/a_' + fig_title + '.pdf')
            fig_title = name + '_' + chn + '_all_' + str(run)
            fig = X['QA_' + name + '_sorted'][run].plot_image(picks=[chn], title=fig_title, show=False, ax=outer_list[id_chn*4])[0]  # No response
            fig.savefig('Results/a_' + fig_title + '.pdf')
import matplotlib.image as mpimg
for name in ['audio', 'rec', 'cen_word']:
    fig, axs = plt.subplots(nrows=3, ncols=2, constrained_layout=True)
    for id_chn, chn in enumerate(['Pz', 'Fz']):
        for id_suffix, suffix in enumerate(['_fastest_answer_', '_no_answer_', '_all_']):
            fig_title = name + '_' + chn + suffix + str(run)
            file = 'Results/a_' + fig_title + '.pdf'
            axs[id_suffix, id_chn].imshow(mpimg.imread(file)) 
            axs[id_suffix, id_chn].set_axis_off() 
    fig.savefig('a_' + name + '.pdf')
pdb.set_trace()
pdb.set_trace()
"""
# nr_run=0
# nr_trial = 0
# plot_joint_t_freq(X['QA_sorted'][nr_run][nr_trial], trigger_annot=X['QA'][nr_run].run_annot[onset_sort_ind[nr_run][nr_trial]][:,[0, -1]],channel=['CP5', 'CP6', 'Fpz'],colorbar=False, fmin=0.0, save=True, fmax=70.0,tmin=0.0, tmax=28.0, n_perseg=50,fig_name='temp_tf_fastest.pdf')


# nr_trial = -1
# plot_joint_t_freq(X['QA_sorted'][nr_run][nr_trial], trigger_annot=X['QA'][nr_run].run_annot[onset_sort_ind[nr_run][nr_trial]][:,[0, -1]],channel=['CP5', 'CP6', 'Fpz'],colorbar=False, fmin=0.0, save=True, fmax=70.0,tmin=0.0, tmax=28.0, n_perseg=50,fig_name='temp_tf_slowest.pdf')

if plt_flag:
    if erp_flag:
        X['QA_audio'][3].plot_image(picks=['Fpz'])
        # plot_epochs_image(X['QA_cen_word'][3], picks=['TP10'])
        # plot_epochs_image(X['QA_audio'][3], picks=['Fpz'])
        # X['QA_audio'][3].average().plot()
        X['Cali_display'][0].plot_image(picks=['Pz'])

    else:
        picks = ['Pz', 'Fz', 'CP5', 'CP6']
        for nr_run in range(1, 2):
            for nr_trial in range(2, 10):
                plot_joint_t_freq([X['QA'][nr_run][nr_trial]],
                                  trigger_annot=X['QA'][nr_run].run_annot[nr_trial][:,[0, -1]],
                                  channel=picks, bg_text='Q&A',
                                  colorbar=False, fmin=0.0, save=True, fmax=70.0,
                                  tmin=0.0, tmax=28.0, n_perseg=50,
                                  fig_name=path + 'Results/new_qa/QA_Run_' + str(nr_run) + '_T' + str(nr_trial) + '.pdf')

                # plot_joint_psd(
                #     X['QA'][nr_run][nr_trial],
                #     freq=[(1.0, 4.0, 'Delta (1-4)'),
                #           (4.0, 8.0, 'Theta (4-8)'),
                #           (8.0, 14.0, 'Alpha (8-14)'),
                #           (14.0, 20.0, 'Low Beta (14-20)'),
                #           (20.0, 30.0, 'High Beta (20-30)'),
                #           (30.0, 70.0, 'Gamma (30-70))'),
                #           (8.9, 9.1, 'Peak Frequency (9Hz)')],
                #     tmin=30.0, tmax=180.0, picks='eeg',
                #     fig_name=path + 'RS_' + state + '_run_' + str(nr_run) + '.pdf',
                #     fig_unit_height=3, save=True,
                #     fig_unit_width=3, fig_height=None, fig_width=None)

        pdb.set_trace()
        for nr_run in range(4):
            plot_joint_t_freq([X['RS']['open'][nr_run], X['RS']['close'][nr_run]], channel=picks,
                              colorbar=False, fmin=0.0, save=True, fmax=70.0,
                              tmin=30.0, tmax=180.0,
                              fig_name=path + 'Results/new/RS_open_vs_close_Run_' + str(nr_run) + '.png')

        for state in ['open', 'close']:
            for nr_run in range(4):
                plot_joint_psd(
                    X['RS'][state][nr_run],
                    freq=[(1.0, 4.0, 'Delta (1-4)'),
                          (4.0, 8.0, 'Theta (4-8)'),
                          (8.0, 14.0, 'Alpha (8-14)'),
                          (14.0, 20.0, 'Low Beta (14-20)'),
                          (20.0, 30.0, 'High Beta (20-30)'),
                          (30.0, 70.0, 'Gamma (30-70))'),
                          (8.9, 9.1, 'Peak Frequency (9Hz)')],
                    tmin=30.0, tmax=180.0, picks='eeg',
                    fig_name=path + 'Results/new/RS_' + state + '_run_' + str(nr_run) + '.pdf',
                    fig_unit_height=3, save=True,
                    fig_unit_width=3, fig_height=None, fig_width=None)
pdb.set_trace()
meta_extract_flag = True
if meta_extract_flag: 

    cali_trial = np.asarray([[114, 132, 93 , 173, 139, 174, 50, 16, 57, 118],
                             [136, 34, 167, 133, 88, 78, 134, 25, 67, 104]])

    trigger = X['QA'][0].all_annot

    meta_infomat = np.zeros([160, 3], dtype=float)  # sentence_length, cen_start, cen_dur

    for nr_trial in range(160):
        trigger[nr_trial][:, -1] == 44
        time = trigger[nr_trial][:, 0] / 500.0
        ind_q_start = np.where(trigger[nr_trial][:, -1] == 44)[0]
        ind_cen_start = np.where(trigger[nr_trial][:, -1] == 50)[0]
        ind_cen_end = np.where(trigger[nr_trial][:, -1] == 51)[0]
        ind_beep_start = np.where(trigger[nr_trial][:, -1] == 46)[0]

        meta_infomat[nr_trial, 0] = time[ind_beep_start] - 0.4 - time[ind_q_start]
        meta_infomat[nr_trial, 1] = time[ind_cen_start] - time[ind_q_start]
        meta_infomat[nr_trial, 2] = time[ind_cen_end] - time[ind_cen_start]

    import pandas as pd
    extract_df = pd.read_pickle('Q_Session_1_exp_180.pkl')
    censor_start = extract_df['SENTENCE_INFO']['beeped_word_timestamp_start'].values
    censor_dur = extract_df['SENTENCE_INFO']['beeped_word_duration'].values
    sen_duration = extract_df['SENTENCE_INFO']['sentence_duration'].values
    sen_text = extract_df['SENTENCE_INFO']['sen_content'].values

    prior_info = np.asarray([sen_duration, censor_start, censor_dur, np.arange(0, 180, 1)]).T
    all_unique_match_list = np.empty((0, 2))
    all_multi_match_list = np.empty((0, 2))
    overlap_list = []
    for list_trial in [range(80), range(80, 120), range(120, 160)]:

        match_list = []
        exceed_list = []
        zero_cnt = 0 
        exceed_cnt = 0
        for ind in list_trial:
            cen_s = meta_infomat[ind, 1]
            cen_d = meta_infomat[ind, 2]
            sen_len = meta_infomat[ind, 0]

            cen_d_ind = np.where(np.abs(cen_d - prior_info[:, 2]) < 0.063)[0]
            after_cen_d = prior_info[cen_d_ind]
            cen_s_ind = np.where(np.abs(cen_s - after_cen_d[:, 1]) < 0.063)[0]
            after_cen_s = after_cen_d[cen_s_ind]
            sen_len_ind = np.where(np.abs(sen_len - after_cen_s[:, 0]) < 0.063)[0]
            after_sen_len = after_cen_s[sen_len_ind]
            # print('------------' + str(ind) + '------------')
            if after_sen_len.shape[0] == 1:
                # print('Trigger info: ' + ', '.join([str(val) for val in meta_infomat[ind]]))
                # print('Matched info: ' + ', '.join([str(val) for val in prior_info[int(after_sen_len[:, -1])] ]))
                match_list.append([ind, int(after_sen_len[:, -1])])
            else:
                # print('Trigger info: ' + ', '.join([str(val) for val in meta_infomat[ind]]))
                if after_sen_len.shape[0] == 0:
                    zero_cnt += 1
                elif after_sen_len.shape[0] > 1:
                    exceed_cnt += 1
                    for ex in after_sen_len[:, -1]:
                        exceed_list.append([ind, int(ex)])
                # print(after_sen_len[:, -1])

        match_list = np.asarray(match_list)
        exceed_list = np.asarray(exceed_list)
        set(cali_trial[0,:]).intersection(set(exceed_list[:,1]))
        overlap_list.append(set(match_list[:,1]).intersection(set(exceed_list[:,1])))
        print(np.unique(match_list[:,1]).shape)
        print(len(match_list))
        print(zero_cnt)
        print(exceed_cnt)
        all_unique_match_list = np.vstack((all_unique_match_list, match_list))
        all_multi_match_list = np.vstack((all_multi_match_list, exceed_list))

    trial_list = np.asarray([[2, 117],
                            [8, 143],
                            [10, 3],
                            [13, 17],
                            [17, 146],
                            [20, 69],
                            [33, 130],  # 138 == 130
                            [41, 101], 
                            [49, 116],  # 48-80 are with not recordings.
                            [50, 20],
                            [54, 5],
                            [66, 130],
                            [67, 85],
                            [69, 49],
                            [72, 69],
                            [84, 49],
                            [85, 55],   # 55 == 135
                            [90, 22],
                            [107, 89],
                            [108, 49],
                            [116, 75],
                            [118, 1],
                            [119, 81],
                            [137, 75],
                            [140, 3],
                            [144, 86],
                            [150, 69],
                            [154, 116]])
    all_list = np.vstack((all_unique_match_list, trial_list))
    pdb.set_trace()
    print(overlap_list)
    sorted_all_list = all_list[all_list[:,0].argsort()]
    np.sort(meta_infomat[:,0])

    aaa = extract_df.iloc[sorted_all_list[:, 1]]
    aaa.rename(columns={'index':'global_index'},inplace=True)
    aaa.reset_index()

    pdb.set_trace()


    pdb.set_trace()


"""
# Plot spectrum using basic FFT
import matplotlib.pyplot as plt
t = np.arange(500)
f = 10.0
fs = 500.0
x_sin = 0.024397 * np.sin(2 * np.pi * f * t / fs + 0.9 * np.pi)
sp_x = np.fft.fft(x_sin)
plt.plot(sp_x, 'r--')



# Apply EMD to the tACS-EEG signal
from PyEMD import EMD
emd = EMD(std_thr=0.0000001, range_thr=0.0000005)
imfs = emd(data.squeeze())

fig = plt.figure()
for nr_row in range(7): ax0 = fig.add_subplot(8, 2, nr_row*2 + 1); ax0.plot(imfs[nr_row][15000:25500]); ax1 = fig.add_subplot(8, 2, nr_row*2 + 2); ax1.plot(np.log(np.abs(np.fft.fft(imfs[nr_row][15000:25500]))));
ax0 = fig.add_subplot(8, 2, 15); ax0.plot(data.squeeze()[15000:25500]); ax1 = fig.add_subplot(8, 2, 16); ax1.plot(np.log(np.abs(np.fft.fft(data.squeeze()[15000:25500]))));


ax00 = fig.add_subplot(7, 2, 1)
ax10 = fig.add_subplot(223)

ax01 = fig.add_subplot(222)
ax11 = fig.add_subplot(224)


for i in range(8):
signal = imfs[0][15000:15500]
spec = np.log(np.abs(np.fft.fft(signal)))



ax00.plot(imfs[0][15000:15500])
ax10.plot(imfs[1][15000:15500])

ax01.plot(np.log(np.abs(np.fft.fft(imfs[0][15000:15500]))) )
ax11.plot(np.log(np.abs(np.fft.fft(imfs[1][15000:15500]))) )

"""

"""
if colorbar:
    xlabel, ylabel, title, scale, rotation, axis_lim = [], [], [], [], [], []
    xlabel += [['', 'Time/s', ''], ['', 'Time/s', '']] * nr_raw
    ylabel += [['Amplitude', 'Frequency/Hz', 'Amplitude'], ['', 'Amplitude/uV', '']] * nr_raw
    title += [['Spectrum', 'Short Time Fourier Transform', 'Cbar'], ['Electrodes Location', 'Amplitude of EEG signal', '']] * nr_raw

    scale += [[[4.3, 2], [1, 1], [0.2, 1]], [[1, 1], [1, 1], [1, 1]]] * nr_raw
    rotation += [[90, 0, 90], [0, 0, 0]] * nr_raw
    axis_lim += [[(0, 70, 0, 70), 0, (0, 70, 0, 70)], [0, 0, 0]] * nr_raw
    row_ratios = []
    row_ratios += [2, 1] * nr_raw

    xticks, xticklabels = [], []
    xticks += [[None, np.linspace(tmin, tmax, num_t_step), None],
               [None, np.linspace(tmin, tmax, num_t_step), None]] * nr_raw

    xticklabels += [[None, [str(i) for i in np.linspace(tmin, tmax, num_t_step)], None],
                    [None, [str(i) for i in np.linspace(tmin, tmax, num_t_step)], None]] * nr_raw

    xticks = np.asarray(xticks)
    xticklabels = np.asarray(xticklabels)
    fig, axes = fig_init(nr_row=2 * nr_raw, nr_col=3,
                         row_ratios=row_ratios, col_ratios=[1, 3, 1],
                         fig_unit_height=3, fig_unit_width=3,
                         xlabel=np.asarray(xlabel),
                         ylabel=np.asarray(ylabel),
                         title=np.asarray(title),
                         scale=np.asarray(scale),
                         rotation=np.asarray(rotation),
                         axis_lim=np.asarray(axis_lim))
"""