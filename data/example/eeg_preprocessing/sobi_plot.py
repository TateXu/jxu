#========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2020-11-26 13:23:30
# Name       : RS_plot.py
# Version    : V1.0 # Description: .
#========================================

import matplotlib.pyplot as plt
import pdb

from jxu.data.eeg_process import NIBSEEG
from jxu.data.utils import nibs_event_dict
import pandas as pd

import numpy as np
import pickle

from mne.decoding import CSP
from mne.viz import plot_topomap as topo
import seaborn as sns

bands_list_name = ['BP(3-70)', 'delta(1-4)', 'theta(4-8)', 'Alpha(8-12)',
                   'L_Beta(12-20)', 'H_Beta(20-30)', 'L_Gamma(30-70)']
bands_list = [(3.0, 70.0), (1.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 20.0),
              (20.0, 30.0), (30.0, 70.0)]

path = '/home/jxu/File/Data/NIBS/Stage_one/EEG/Processed/RS_epoch/'
# freq_pkl to collect all individual into one single freq. pkl
# all_pkl to collect all freq pkl into a single all_pkl
# all_sobi use all_pkl to generate sobi
individual = 'all_sobi'  # 'freq_sobi' 'all_sobi' 'all_pkl' all_sobi'
process = False
col_topo = 2

bands_id = 0
bands = bands_list_name[0]
ses_eeg = NIBSEEG()

n_comp = 120
comp_start = 0


if not process:

    hratio = [1]
    hratio.extend([6] * n_comp)
    wratio = [1]

    if 'freq' in individual:
        col_name = ['Topo for all', 'pre', 'post', 'difference']
        fig_list = np.empty((4, ), dtype='object')
        axes_list = np.empty((4, ), dtype='object')
        if col_topo == 2:
            wratio.extend([6, 1, 10, 10, 10])
        else:
            wratio.extend([6, 1, 10, 6, 1, 10, 10])

        for fig_id in range(4):
            fig, axes = plt.subplots(n_comp + 1, 8-col_topo,
                                     figsize=(sum(wratio), 1 + n_comp * 6),
                                     gridspec_kw={'width_ratios': wratio,
                                                  'height_ratios': hratio})
            for i in range(n_comp):
                axes[i+1, 0].text(0.4, 0.5, 'IC ' + str(i+1+comp_start),
                                  rotation='vertical', fontsize='xx-large',
                                  fontweight='heavy')
                axes[i+1, 0].set_axis_off()

            for id_j, j in enumerate([1, 3, 6-col_topo, 7-col_topo]):
                axes[0, j].text(0.4, 0.5, col_name[id_j], rotation='horizontal',
                                fontsize='xx-large', fontweight='heavy')
            for j in range(8-col_topo):
                axes[0, j].set_axis_off()

            fig_list[fig_id] = fig
            axes_list[fig_id] = axes

    elif 'all' in individual:

        col_name = ['Topo', 'All', 'All diff', '0 Hz', '0 Hz diff', '4 Hz', '4 Hz diff',
                    '10 Hz', '10 Hz diff', '40 Hz', '40 Hz diff']
        wratio.extend([6, 1])
        wratio.extend([10] * 10)

        fig_main, axes_main = plt.subplots(n_comp + 1, 13,
                                           figsize=(sum(wratio), sum(hratio)),
                                           gridspec_kw={'width_ratios': wratio,
                                                        'height_ratios': hratio})
        for i in range(n_comp):
            axes_main[i+1, 0].text(0.4, 0.5, 'IC ' + str(i+1+comp_start),
                                   rotation='vertical', fontsize='xx-large',
                                   fontweight='heavy')
            axes_main[i+1, 0].set_axis_off()

        for id_j, j in enumerate([1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
            axes_main[0, j].text(0.4, 0.5, col_name[id_j], rotation='horizontal',
                                 fontsize='xx-large', fontweight='heavy')
        for j in range(13):
            axes_main[0, j].set_axis_off()

"""
Input data: S1_Ses0_filter-band.pkl, which is epoched RS close data.

kkk

process=True, individual='freq_pkl'
Load the filtered RS epoch data and


"""

if process:
    if individual == 'freq_pkl':
        for freq_id, freq in enumerate([0, 4, 10, 40]):
            all_pre = np.empty((126, 0))
            all_post = np.empty((126, 0))
            for id_sj, subj in enumerate([1, 2, 3, 5, 6, 7, 8, 10]):
                ses = ses_eeg.subject_list[[*ses_eeg.subject_list.keys()][subj]].index(freq)
                with open('{0}S{1}_Ses{2}_{3}.pkl'.format(
                        path, str(subj), str(ses), bands_list_name[bands_id]), 'rb') as f:
                    tmp = pickle.load(f)
                data_pre = tmp[0].pick(picks='eeg').get_data()[0]
                data_post = tmp[3].pick(picks='eeg').get_data()[0]
                all_pre = np.concatenate((all_pre, data_pre), axis=1)
                all_post = np.concatenate((all_post, data_post), axis=1)
                print('Freq{0}_Subject{1}'.format(str(freq), str(subj)))

            with open('{0}All_concat_freq_{1}Hz_{2}.pkl'.format(
                       path, str(freq), bands_list_name[bands_id]), 'wb') as f_out:
                pickle.dump([all_pre, all_post], f_out, protocol=pickle.HIGHEST_PROTOCOL)
        import pdb;pdb.set_trace()

    elif individual == 'freq_sobi':

        print('----------------- SOBI Freq starts ---------------------------')
        from sobi.sobi import sobi

        for freq_id, freq in enumerate([0, 4, 10, 40]):
            with open('{0}All_concat_freq_{1}Hz.pkl'.format(path, str(freq)), 'rb') as f_in:
                all_pre, all_post = pickle.load(f_in)
            with open('{0}S{1}_Ses{2}_{3}.pkl'.format(path, str(1), str(1), bands_list_name[0]), 'rb') as f:
                tmp = pickle.load(f)

            all_data = np.concatenate((all_pre, all_post), axis=1)

            res = sobi(all_data, eps=1.0e-5)

            with open('{0}SOBI_freq_{1}Hz.pkl'.format(path, str(freq)), 'wb') as f_out:
                pickle.dump(res, f_out, protocol=pickle.HIGHEST_PROTOCOL)

            print('------------------------------------------------------')
            print(freq)
            print('------------------------------------------------------')
    elif individual == 'all_pkl':

        all_data = np.empty((126, 0))
        for freq_id, freq in enumerate([0, 4, 10, 40]):
            with open('{0}All_concat_freq_{1}Hz_{2}.pkl'.format(
                    path, str(freq), bands_list_name[bands_id]), 'rb') as f_in:
                all_pre, all_post = pickle.load(f_in)

            all_data = np.concatenate((all_data, all_pre, all_post), axis=1)
        with open('{0}All_concat_{1}.pkl'.format(
                path, bands_list_name[bands_id]), 'wb') as f_out:
            pickle.dump(all_data, f_out, protocol=pickle.HIGHEST_PROTOCOL)
        import pdb;pdb.set_trace()

    elif individual == 'all_sobi':

        from sobi.sobi import sobi
        with open('{0}All_concat_{1}.pkl'.format(
                path, bands_list_name[bands_id]), 'rb') as f_in:
            all_data = pickle.load(f_in)
        import pdb;pdb.set_trace()

        res = sobi(all_data, eps=1.0e-5)

        with open('{0}SOBI_all_{1}.pkl'.format(
                path, bands_list_name[bands_id]), 'wb') as f_out:
            pickle.dump(res, f_out, protocol=pickle.HIGHEST_PROTOCOL)
        import pdb;pdb.set_trace()

    elif individual == 'all_pca_sobi':

        from sobi.sobi import sobi
        from sklearn.decomposition import PCA

        print('-------------------------------------------------------')
        print('Load All concat data and apply PCA and compute SOBI ICA')
        print('-------------------------------------------------------')

        with open('{0}All_concat_{1}.pkl'.format(
                path, bands_list_name[bands_id]), 'rb') as f_in:
            all_data = pickle.load(f_in)
        pca_obj = PCA()

        source_data = pca_obj.fit_transform(all_data.T).T
        filters =  pca_obj.components_.T

        r_eigvals = pca_obj.explained_variance_ratio_
        import pdb;pdb.set_trace()

        T_eigvals = 0.99 # 76 - 0.99999

        # select the first several components which take x% eigvals.
        nr_comp = np.where(r_eigvals.cumsum() > T_eigvals)[0][0]

        res = sobi(source_data[:nr_comp], eps=1.0e-5)

        with open('{0}PCA_{1}_SOBI_all_{2}.pkl'.format(
                path, str(nr_comp), bands_list_name[bands_id]), 'wb') as f_out:
            pickle.dump([res, pca_obj], f_out, protocol=pickle.HIGHEST_PROTOCOL)
        import pdb;pdb.set_trace()

else:
    import mne

    if 'freq' in individual:
        for freq_id, freq in enumerate([0, 4, 10, 40]):

            with open('{0}SOBI_freq_{1}Hz.pkl'.format(path, str(freq)), 'rb') as f_in:
                res = pickle.load(f_in)
            with open('{0}S{1}_Ses{2}_{3}.pkl'.format(path, str(1), str(1), bands_list_name[0]), 'rb') as f:
                tmp = pickle.load(f)
            info = tmp[0].pick(picks='eeg').info
            source, A_, W_ = res
            len_data = int(source.shape[1]/2)
            source_pre = source[:, :len_data]
            source_post = source[:, len_data:]
            A_pre = A_
            A_post = A_

            ep_pre = mne.io.RawArray(source_pre, info)
            ep_post = mne.io.RawArray(source_post, info)

            drop_bad = True
            if drop_bad:
                bads = ['FC3', 'FFC5h', 'FCC5h', 'FFC3h', 'FCC3h', 'T7', 'FTT7h', 'TTP7h',
                        'FT7', 'FC5', 'C5', 'C3', 'F9', 'F5', 'CCP5h', 'AFF5h', 'FC1',
                        'P9', 'TP10', 'FFT7h', 'TP7', 'FFT9h', 'F3', 'FT9']
                info = tmp[0].pick(picks='eeg').info
                loc_chn = [info.ch_names.index(chn) for chn in bads]

                select_A_pre = np.delete(A_pre, tuple(loc_chn), axis=0)
                select_A_post = np.delete(A_post, tuple(loc_chn), axis=0)
                info = tmp[0].pick(picks='eeg').drop_channels(bads).info
            else:
                select_A_pre = A_pre
                select_A_post = A_post

            for id_comp in range(n_comp):
                im_tmp_pre = topo(select_A_pre[:, id_comp + comp_start], info, axes=axes_list[freq_id][id_comp+1, 1], show=False)
                fig_list[freq_id].colorbar(im_tmp_pre[0], cax=axes_list[freq_id][id_comp+1, 2], orientation='vertical')
                tmp_1 = ep_pre.plot_psd(fmax=70.0, picks=[id_comp + comp_start], ax=axes_list[freq_id][id_comp+1, 3], show=False)

#                 im_tmp_post = topo(select_A_post[:, id_comp + comp_start], info, axes=axes_list[freq_id][id_comp+1, 4], show=False)
#                 fig_list[freq_id].colorbar(im_tmp_post[0], cax=axes_list[freq_id][id_comp+1, 5], orientation='vertical')
                tmp_2 = ep_post.plot_psd(fmax=70.0, picks=[id_comp + comp_start], ax=axes_list[freq_id][id_comp+1, 6-col_topo], show=False)

                xdata = axes_list[freq_id][id_comp+1, 3].lines[2].get_xdata()
                ydata_pre = axes_list[freq_id][id_comp+1, 3].lines[2].get_ydata()
                ydata_post = axes_list[freq_id][id_comp+1, 6-col_topo].lines[2].get_ydata()
                y_diff = ydata_post - ydata_pre

                axes_list[freq_id][id_comp+1, 7-col_topo].plot(xdata, y_diff, color='k', linestyle='-', linewidth=0.5)
                axes_list[freq_id][id_comp+1, 7-col_topo].grid(True, linestyle='dotted')

            fig_list[freq_id].tight_layout()

            fig_list[freq_id].savefig('SOBI_{0}Hz_last_half_wo_bad_chn.jpg'.format(str(freq)))
    elif 'all' in individual:

        freq_list = [0, 4, 10, 40]
        stg_list = ['pre', 'post']
        subj_list = [1, 2, 3, 5, 6, 7, 8, 10]
        column_names = ['stim_freq', 'session', 'subject', 'IC', 'freq_val', 'psd_val']
        all_df = pd.DataFrame(columns=column_names)
        generate_df = True
        accm_len = 0

        with open('{0}SOBI_all_{1}.pkl'.format(path, bands_list_name[bands_id]), 'rb') as f_in:
            res = pickle.load(f_in)
        with open('{0}S{1}_Ses{2}_{3}.pkl'.format(path, str(1), str(1), bands_list_name[0]), 'rb') as f:
            tmp = pickle.load(f)
        info = tmp[0].pick(picks='eeg').info
        source_, A_, W_ = res

        if generate_df:
            import pdb;pdb.set_trace()
            n_comp = 120
            comp_start = 0
            feat_type = 'apf'
            if feat_type == 'psd':
                for freq_id, freq in enumerate(freq_list):
                    _subj_list = subj_list.copy()

                    for stg_id, stg in enumerate(stg_list):
                        for subj_id, subj in enumerate(_subj_list):
                            data = source_[:, accm_len: accm_len + 180001]
                            ep = mne.io.RawArray(data, info)
                            import pdb;pdb.set_trace()

                            for id_comp in range(n_comp):
                                fig_tmp, ax_tmp = plt.subplots(1, 1, figsize=(10, 6))
                                tmp_1 = ep.plot_psd(fmax=70.0, picks=[id_comp + comp_start], ax=ax_tmp, show=False, verbose='WARNING',
                                                    n_fft=4096)

                                xdata = ax_tmp.lines[2].get_xdata()
                                psd = ax_tmp.lines[2].get_ydata()
                                data_dict = {'stim_freq': [freq] * len(psd),
                                            'session': [stg] * len(psd),
                                            'subject': [subj] * len(psd),
                                            'IC': [id_comp+comp_start] * len(psd),
                                            'freq_val': xdata,
                                            'psd_val': psd}
                                tmp_df = pd.DataFrame.from_dict(data_dict, orient='columns')
                                all_df = all_df.append(tmp_df)

                                del fig_tmp, ax_tmp

                            print('Finish extracting data from freq_{0}Hz, session_{1}, subject_{2}'.format(str(freq), stg, str(subj)))
                            print('Segment {0}'.format(str(accm_len/180001)))
                            accm_len += 180001

                all_df.to_pickle('{0}SOBI_all_df_default_{1}_highres.pkl'.format(path, bands_list_name[bands_id]))
                import pdb;pdb.set_trace()
            elif feat_type == 'apf':
                from philistine.mne import attenuation_iaf, savgol_iaf

                column_names = ['stim_freq', 'session', 'subject', 'IC', 'IAPF', 'CoG', 'Alpha_Band']
                all_df = pd.DataFrame(columns=column_names)
                for freq_id, freq in enumerate(freq_list):
                    _subj_list = subj_list.copy()

                    for stg_id, stg in enumerate(stg_list):
                        for subj_id, subj in enumerate(_subj_list):
                            data = source_[:, accm_len: accm_len + 180001]
                            ep = mne.io.RawArray(data, info)

                            for id_comp in range(n_comp):
                                try:
                                    tmp_obj = savgol_iaf(ep, picks=[id_comp], resolution=0.02)
                                except:
                                    tmp_obj = savgol_iaf(ep, picks=[id_comp], resolution=0.02, fmin=7.0, fmax=14.0)
                                data_dict = {'stim_freq': [freq],
                                            'session': [stg],
                                            'subject': [subj],
                                            'IC': [id_comp+comp_start],
                                            'IAPF': [tmp_obj[0]],
                                            'CoG': [tmp_obj[1]],
                                            'Alpha_Band': [tmp_obj[2]]
                                             }
                                tmp_df = pd.DataFrame.from_dict(data_dict, orient='columns')
                                all_df = all_df.append(tmp_df)

                            print('Finish extracting data from freq_{0}Hz, session_{1}, subject_{2}'.format(str(freq), stg, str(subj)))
                            print('Segment {0}'.format(str(accm_len/180001)))
                            accm_len += 180001

                all_df.to_pickle('{0}IAPF_savgol_{1}.pkl'.format(path, bands_list_name[bands_id]))
                import pdb;pdb.set_trace()

            elif feat_type == 'HT':
                from jxu.pipelines.features import HT
                column_names = ['stim_freq', 'session', 'subject', 'IC', 'IF_median', 'IF_mean']
                all_df = pd.DataFrame(columns=column_names)
                for freq_id, freq in enumerate(freq_list):
                    _subj_list = subj_list.copy()

                    for stg_id, stg in enumerate(stg_list):
                        for subj_id, subj in enumerate(_subj_list):
                            data = source_[:, accm_len: accm_len + 180001]
                            IF = HT(freq=1000.0).transform(data)
                            IF_median = np.median(IF, axis=-1)
                            IF_mean = np.mean(IF, axis=-1)

                            data_dict = {'stim_freq': [freq] * len(IF_median),
                                        'session': [stg] * len(IF_median),
                                        'subject': [subj] * len(IF_median),
                                        'IC': range(len(IF_median)),
                                        'IF_median': IF_median,
                                        'IF_mean': IF_mean
                                        }
                            tmp_df = pd.DataFrame.from_dict(data_dict, orient='columns')
                            all_df = all_df.append(tmp_df)

                            print('Finish extracting data from freq_{0}Hz, session_{1}, subject_{2}'.format(str(freq), stg, str(subj)))
                            print('Segment {0}'.format(str(accm_len/180001)))
                            accm_len += 180001

                all_df.to_pickle('{0}SOBI_IF_{1}.pkl'.format(path, bands_list_name[bands_id]))


        else:
            drop_bad = True
            if drop_bad:
                bads = ['FC3', 'FFC5h', 'FCC5h', 'FFC3h', 'FCC3h', 'T7', 'FTT7h', 'TTP7h',
                        'FT7', 'FC5', 'C5', 'C3', 'F9', 'F5', 'CCP5h', 'AFF5h', 'FC1',
                        'P9', 'TP10', 'FFT7h', 'TP7', 'FFT9h', 'F3', 'FT9']
                info = tmp[0].pick(picks='eeg').info
                loc_chn = [info.ch_names.index(chn) for chn in bads]

                select_A = np.delete(A_, tuple(loc_chn), axis=0)
                info = tmp[0].pick(picks='eeg').drop_channels(bads).info
            else:
                select_A = A_

            all_df = pd.read_pickle('{0}SOBI_all_df.pkl'.format(path))
            for id_comp in range(n_comp):
                im_tmp = topo(select_A[:, id_comp + comp_start], info, axes=axes_main[id_comp+1, 1], show=False)
                fig_main.colorbar(im_tmp[0], cax=axes_main[id_comp+1, 2], orientation='vertical')
                tmp_all_df = all_df.loc[all_df['IC'] == id_comp + comp_start].copy()
                sns.lineplot(x='freq_val', y='psd_val', hue='session', data=tmp_all_df, ax=axes_main[id_comp+1, 3])

                diff_all_df = all_df.loc[(all_df['session'] == 'post') & (all_df['IC'] == id_comp + comp_start)].copy()
                diff_all_df['diff_val'] = all_df.loc[(all_df['session'] == 'post') & (all_df['IC'] == id_comp + comp_start)].psd_val.values - \
                    all_df.loc[(all_df['session'] == 'pre') & (all_df['IC'] == id_comp + comp_start)].psd_val.values
                sns.lineplot(x='freq_val', y='diff_val', data=diff_all_df, ax=axes_main[id_comp+1, 4])
                for freq_id, freq in enumerate(freq_list):
                    tmp_df = all_df.loc[(all_df['IC'] == id_comp + comp_start) & (all_df['stim_freq'] == freq)]
                    sns.lineplot(x='freq_val', y='psd_val', hue='session', data=tmp_df, ax=axes_main[id_comp+1, freq_id*2+5])
                    axes_main[id_comp+1, freq_id*2+5].set_title('IC{0}: Freq-{1}Hz'.format(str(id_comp+1), str(freq)))

                    diff_df = tmp_df.loc[tmp_df['session'] == 'post'].copy()
                    diff_df['diff_val'] = tmp_df.loc[tmp_df['session'] == 'post'].psd_val.values - \
                        tmp_df.loc[tmp_df['session'] == 'pre'].psd_val.values
                    sns.lineplot(x='freq_val', y='diff_val', data=diff_df, ax=axes_main[id_comp+1, freq_id*2+6])
                print(id_comp)

        fig_main.tight_layout()

        fig_main.savefig('Overall_SOBI_1st_wo_bad_chn.jpg')


        import pdb;pdb.set_trace()


        # slicing the data
        # data to raw array
        # plot psd and get the y value of psd
        # collect the y value from all subjects for each condition
        # plot the lineplot usng seaborn
        # plot the difference plot
        # plot the all plots using lineplot




import pdb;pdb.set_trace()

