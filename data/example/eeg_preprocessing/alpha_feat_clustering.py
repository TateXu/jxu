#========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2021-03-23 12:59:14
# Name       : clusering.py
# Version    : V1.0
# Description: .
#========================================
import pandas as pd
import mne
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import numpy as np
import pickle
import seaborn as sns
from matplotlib import pyplot as plt

from mne.viz import plot_topomap as topo

from matplotlib import gridspec
from jxu.pipelines.features import alpha_peak_feat

def ind_convert(x, subj_list, stim_list, IC_list):

    IC, tmp = divmod(x, len(subj_list)*len(stim_list))
    subj, stim = divmod(tmp, len(stim_list))

    return IC_list[IC], subj_list[subj], stim_list[stim]

path = '/home/jxu/File/Data/NIBS/Stage_one/EEG/Processed/RS_epoch/'

fig_root = '/home/jxu/File/Data/jxu_results/data/example/eeg_preprocessing/'
# sobi_suffix = ''
sobi_suffix = '_BP(3-70)'

all_df = pd.read_pickle('{0}SOBI_all_df_default{1}_highres.pkl'.format(path, sobi_suffix))
# '{0}SOBI_all_{1}.pkl'.format(path, bands_list_name[bands_id])
diff_all_df = all_df.loc[all_df['session'] == 'post'].copy()
diff_all_df['diff_val'] = all_df.loc[all_df['session'] == 'post'].psd_val.values -\
    all_df.loc[all_df['session'] == 'pre'].psd_val.values


bands_list = [(1.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 30.0), (30.0, 70.0)]
bands_list_name = ['delta', 'theta', 'alpha', 'beta', 'gamma']

new_band_name = ['BP(1-70)', 'delta(1-4)', 'theta(4-8)', 'Alpha(8-12)',
                 'L_Beta(12-20)', 'H_Beta(20-30)', 'L_Gamma(30-70)']

all_IC_flag = False
if all_IC_flag:
    n_comp = 120
    IC_list = range(n_comp)
    fig_sobi_suffix = sobi_suffix
else:
    if sobi_suffix == '':
        IC_list = [22, 23, 24, 48, 59, 75, 93]
    elif sobi_suffix == '_BP(3-70)':
        IC_list = [17, 18, 20, 46, 52, 57, 67, 76, 80, 97, 98, 104,
                   107, 109, 115]
        # MI related: 18, 20, 98
    n_comp = len(IC_list)
    all_df = all_df.loc[all_df['IC'].isin(IC_list)]
    fig_sobi_suffix = sobi_suffix + '_selected_IC'
stim_list = [0, 4, 10, 40]
subj_list = [1, 2, 3, 5, 6, 7, 8, 10]

feat_type = 'single_band'  #  fix_nr_sample'BP', 'all_spectra', 'single_band'
id_band = 4

n_clusters = 6

# threshold to assign IC to cluster. 1 means above avg. samples
T_ratio = 1.0

ref_suffix = ''

# Flag: refer to no-stim condition
# If True: only plot the psd of 4Hz, 10Hz and 40Hz activities referred to avg.
# 0Hz activity across all sessions & subjects
ref_no_stim_cluster_flag = False

# Flag: compute the feature matrix from psd at ICA projected space
# after selecting the feature type, extract the desired features from the raw
# psd data. E.g., BP means the power intergral of the interested bands.
generate_featmat = False

load_topo = False
scatter_heatmap_plot = False
psd_plot = False

only_spec_plot = False# only 2*2 spectrum for cluster, for BS abstract
topo_spec_plot = False

tuning_sl_paras_plot = False
source_local_plot = False
psd_source_cluster_plot = False


individual_topo_sl_plot = False

suffix = 'ap_feat'
for feat_type in ['alpha_peak_freq']:  #
    for n_clusters in[4, 6]:

        df = all_df.copy()

        if generate_featmat:

            if feat_type == 'alpha_peak_freq':
                feat_mat = np.empty((n_comp, len(subj_list), len(stim_list), len(bands_list), 6))
                ap_feat_list = []

            with open('{0}IAPF_savgol{1}.pkl'.format(
                    path, sobi_suffix), 'rb') as f:
                Pkg_IF_df = pickle.load(f)
            for id_comp, val_comp in enumerate(IC_list):
                for id_subj, subj in enumerate(subj_list):
                    for id_stim, stim in enumerate(stim_list):
                        for id_stg, stg in enumerate(['pre', 'post']):
                            tmp_Pkg_IF_df = Pkg_IF_df.loc[(Pkg_IF_df['subject'] == subj) &
                                                        (Pkg_IF_df['stim_freq'] == stim) &
                                                        (Pkg_IF_df['session'] == stg) &
                                                        (Pkg_IF_df['IC'] == val_comp)]
                            IAPF_pkg = tmp_Pkg_IF_df.IAPF.values[0]
                            IAPF_AB_pkg = tmp_Pkg_IF_df.Alpha_Band.values[0]

                            each_pdf = df.loc[(df['IC'] == val_comp) &
                                              (df['session'] == stg) &
                                              (df['stim_freq'] == stim) &
                                              (df['subject'] == subj)]
                            psd = each_pdf.psd_val.values
                            freq = each_pdf.freq_val.values
                            apf_dict = alpha_peak_feat(freq, psd,
                                                       prior_IAPF=IAPF_pkg,
                                                       prior_AB=IAPF_AB_pkg)
                            data_dict = {'subject': [subj],
                                         'IC': [val_comp],
                                         'stim_freq': [stim],
                                         'session': [stg],
                                         'IAP_freq': [apf_dict['IAPF'][0]],
                                         'IAP_freq_raw': [apf_dict['IAPF'][1]],
                                         'IAP_freq_smooth': [apf_dict['IAPF'][2]],
                                         'IAP_power_raw': [apf_dict['IAPF'][3]],
                                         'IAP_power_smooth': [apf_dict['IAPF'][4]],
                                         'Left_IAP_freq': [apf_dict['Lower_bound'][0]],
                                         'Left_IAP_power_raw': [apf_dict['Lower_bound'][1]],
                                         'Left_IAP_power_smooth': [apf_dict['Lower_bound'][2]],
                                         'Right_IAP_freq': [apf_dict['Higher_bound'][0]],
                                         'Right_IAP_power_raw': [apf_dict['Higher_bound'][1]],
                                         'Right_IAP_power_smooth': [apf_dict['Higher_bound'][2]],
                                         '3dB_freq_left': [apf_dict['3db'][0]],
                                         '3dB_freq_right': [apf_dict['3db'][1]],
                                         '1dB_freq_left': [apf_dict['1db'][0]],
                                         '1dB_freq_right': [apf_dict['1db'][1]]
                                         }
                            tmp_df = pd.DataFrame.from_dict(data_dict, orient='columns')
                            ap_feat_list.append(tmp_df)
                            del apf_dict
                print(id_comp)
            import pdb;pdb.set_trace()
            ap_feat_df = pd.concat(ap_feat_list)

            with open('{0}cluster_feat_df_{1}{2}{3}{4}_pkgbased.pkl'.format(
                    path, feat_type, suffix, ref_suffix, fig_sobi_suffix), 'wb') as f:
                pickle.dump(ap_feat_df, f)
            import pdb;pdb.set_trace()
        else:
            # with suffix _pkgbased, it used the iapf computed from other pkg
            with open('{0}cluster_feat_df_{1}{2}{3}{4}_pkgbased.pkl'.format(
                    path, feat_type, suffix, ref_suffix, fig_sobi_suffix), 'rb') as f:
                ap_feat_df = pickle.load(f)

        scatter_plot = False
        ap_feat_df['peak_width'] = ap_feat_df['Right_IAP_freq'] - \
                                   ap_feat_df['Left_IAP_freq']
        if scatter_plot:
            width_ratios = [10] * 4
            height_ratios = [5] * (len(IC_list) + 1)
            len_w = len(width_ratios)
            len_h = len(height_ratios)

            fig_ap = plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))
            gs = gridspec.GridSpec(len(height_ratios), len(width_ratios),
                                   height_ratios=height_ratios,
                                   width_ratios=width_ratios)
            ax_ap = np.empty((len_h, len_w), dtype='object')
            for i in range(len_h):
                for j in range(len_w):
                    ax_ap[i, j] = fig_ap.add_subplot(gs[i, j])

            for i in range(len_h):
                if i == 0:
                    tmp_df = ap_feat_df.copy()
                    # subj_str = 'All subjects'
                    subj_str = 'All ICs'
                else:
                    # tmp_df = ap_feat_df.loc[ap_feat_df['subject'] == subj_list[i-1]]
                    # subj_str = f'S{subj_list[i-1]}'

                    tmp_df = ap_feat_df.loc[ap_feat_df['IC'] == IC_list[i-1]]
                    subj_str = f'IC{IC_list[i-1]}'

                for id_stim, stim in enumerate(stim_list):
                    tmp_ses_df = tmp_df.loc[tmp_df['stim_freq'] == stim]

                    sns.scatterplot(x='IAP_freq_raw', y='IAP_power_raw',
                                    hue='session', size='peak_width',
                                    data=tmp_ses_df, ax=ax_ap[i, id_stim])
                    ax_ap[i, id_stim].set_xlim([7.0, 14.0])
                    ax_ap[i, id_stim].set_title(f'{subj_str}_{stim}Hz')

            fig_ap.tight_layout()
            fig_ap.savefig('IC_based.jpg')
            import pdb;pdb.set_trace()

        diff_scatter_plot = False
        if diff_scatter_plot:
            diff_ap_feat_df = ap_feat_df.copy()
            diff_ap_feat_df['IAP_freq_diff'] = diff_ap_feat_df.IAP_freq_raw.diff()
            diff_ap_feat_df['IAP_power_diff'] = diff_ap_feat_df.IAP_power_raw.diff()
            diff_ap_feat_df['peak_width_diff'] = diff_ap_feat_df.peak_width.diff()

            clean_diff_ap_feat_df = diff_ap_feat_df.loc[diff_ap_feat_df['session'] == 'post']
            import pdb;pdb.set_trace()
            width_ratios = [10] * 4
            height_ratios = [5] * (len(subj_list) + 1)
            len_w = len(width_ratios)
            len_h = len(height_ratios)

            fig_ap = plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))
            gs = gridspec.GridSpec(len(height_ratios), len(width_ratios),
                                   height_ratios=height_ratios,
                                   width_ratios=width_ratios)
            ax_ap = np.empty((len_h, len_w), dtype='object')
            for i in range(len_h):
                for j in range(len_w):
                    ax_ap[i, j] = fig_ap.add_subplot(gs[i, j])
            import pdb;pdb.set_trace()

            for i in range(len_h):
                if i == 0:
                    tmp_df = clean_diff_ap_feat_df.copy()
                    subj_str = 'All subjects'
                    # subj_str = 'All ICs'
                else:
                    tmp_df = clean_diff_ap_feat_df.loc[clean_diff_ap_feat_df['subject'] == subj_list[i-1]]
                    subj_str = f'S{subj_list[i-1]}'

                    # tmp_df = clean_diff_ap_feat_df.loc[clean_diff_ap_feat_df['IC'] == IC_list[i-1]]
                    # subj_str = f'IC{IC_list[i-1]}'

                for id_stim, stim in enumerate(stim_list):
                    tmp_ses_df = tmp_df.loc[tmp_df['stim_freq'] == stim]

                    sns.scatterplot(x='IAP_freq_diff', y='IAP_power_diff',
                                    size='peak_width',
                                    data=tmp_ses_df, ax=ax_ap[i, id_stim])
                                    # hue='subject',
                                    # palette=sns.color_palette("tab10", 8),
                    ax_ap[i, id_stim].axvline(x=0.0, color='grey', linestyle=':')
                    ax_ap[i, id_stim].axhline(y=0.0, color='grey', linestyle=':')
                    ax_ap[i, id_stim].set_xlim([-6, 6])
                    ax_ap[i, id_stim].set_ylim([-25, 25])
                    ax_ap[i, id_stim].set_title(f'{subj_str}_{stim}Hz')

            fig_ap.tight_layout()
            fig_ap.savefig('Subject_based_diff.jpg')
            import pdb;pdb.set_trace()

        corr_plot = True
        if corr_plot:
            width_ratios = [10] * 5
            height_ratios = [5] * 2
            len_w = len(width_ratios)
            len_h = len(height_ratios)

            fig_ap = plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))
            gs = gridspec.GridSpec(len(height_ratios), len(width_ratios),
                                   height_ratios=height_ratios,
                                   width_ratios=width_ratios)
            ax_ap = np.empty((len_h, len_w), dtype='object')
            pair = [('IAP_freq_raw', 'peak_width'), ('IAP_power_raw', 'peak_width')]
            for i in range(len_h):
                for j in range(len_w):
                    ax_ap[i, j] = fig_ap.add_subplot(gs[i, j])

            for id_stim, stim in enumerate(['All', 0, 4, 10, 40]):
                if id_stim == 0:
                    tmp_df = ap_feat_df.copy()
                    subj_str = 'All stims'
                else:
                    tmp_df = ap_feat_df.loc[ap_feat_df['stim_freq'] == stim]
                    subj_str = f'{stim}Hz'

                for i in range(len_h):
                    sns.scatterplot(x=pair[i][0], y=pair[i][1],
                                    hue='session',
                                    data=tmp_df, ax=ax_ap[i, id_stim])
                    ax_ap[i, id_stim].set_title(f'{subj_str}')

            fig_ap.tight_layout()
            fig_ap.savefig('Corr_peak_width.jpg')
            import pdb;pdb.set_trace()


        peak_freq_psd_plot = False
        if peak_freq_psd_plot:
            from random import shuffle
            import itertools

            with open('{0}IAPF_savgol{1}.pkl'.format(
                    path, sobi_suffix), 'rb') as f:
                Pkg_IF_df = pickle.load(f)

            with open('{0}SOBI_IF{1}.pkl'.format(
                    path, sobi_suffix), 'rb') as f:
                IF_df = pickle.load(f)
            import pdb;pdb.set_trace()
            stg_list = ['pre', 'post']
            enum_list = list(itertools.product(subj_list, stim_list, stg_list,
                                               IC_list))
            shuffle(enum_list)

            width_ratios = [10] * 8
            height_ratios =[5] * 20
            len_w = len(width_ratios)
            len_h = len(height_ratios)

            fig_apf_psd = plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))
            gs = gridspec.GridSpec(len(height_ratios), len(width_ratios),
                                   height_ratios=height_ratios,
                                   width_ratios=width_ratios)
            ax_apf_psd = np.empty((len_h, len_w), dtype='object')
            for i in range(len_h):
                for j in range(len_w):
                    ax_apf_psd[i, j] = fig_apf_psd.add_subplot(gs[i, j])

            for id_enum, (subj, stim, stg, IC) in enumerate(enum_list[:160]):
                row, col = divmod(id_enum, 8)
                tmp_df = df.loc[(df['subject'] == subj) &
                                (df['stim_freq'] == stim) &
                                (df['session'] == stg) &
                                (df['IC'] == IC)]
                psd = tmp_df.psd_val.values
                freq = tmp_df.freq_val.values

                tmp_IF_df = IF_df.loc[(IF_df['subject'] == subj) &
                                      (IF_df['stim_freq'] == stim) &
                                      (IF_df['session'] == stg) &
                                      (IF_df['IC'] == IC)]

                IF_median = tmp_IF_df.IF_median.values
                IF_mean = tmp_IF_df.IF_mean.values

                tmp_Pkg_IF_df = Pkg_IF_df.loc[(Pkg_IF_df['subject'] == subj) &
                                              (Pkg_IF_df['stim_freq'] == stim) &
                                              (Pkg_IF_df['session'] == stg) &
                                              (Pkg_IF_df['IC'] == IC)]
                IAPF_pkg = tmp_Pkg_IF_df.IAPF.values[0]
                CoG_pkg = tmp_Pkg_IF_df.CoG.values[0]
                AB_pkg = tmp_Pkg_IF_df.Alpha_Band.values[0]

                tmp_ap_df = ap_feat_df.loc[(ap_feat_df['subject'] == subj) &
                                           (ap_feat_df['stim_freq'] == stim) &
                                           (ap_feat_df['session'] == stg) &
                                           (ap_feat_df['IC'] == IC)]

                IAP_freq = tmp_ap_df.IAP_freq.values
                Left_IAP_freq = tmp_ap_df.Left_IAP_freq.values
                Right_IAP_freq = tmp_ap_df.Right_IAP_freq.values

                ax_apf_psd[row, col].plot(freq, psd, 'grey')
                ax_apf_psd[row, col].axvline(x=IF_median, color='blue')
                ax_apf_psd[row, col].axvline(x=IF_mean, color='blue', linestyle='--')
                ax_apf_psd[row, col].axvline(x=IAP_freq, color='k', linestyle='-')
                ax_apf_psd[row, col].axvline(x=IAPF_pkg, color='green', linestyle='-')
                ax_apf_psd[row, col].axvline(x=AB_pkg[0], color='green', linestyle='--')
                ax_apf_psd[row, col].axvline(x=AB_pkg[1], color='green', linestyle='--')

                # ax_apf_psd[row, col].axvline(x=Left_IAP_freq, color='k', linestyle='--')
                # ax_apf_psd[row, col].axvline(x=Right_IAP_freq, color='k', linestyle='--')
                # ax_apf_psd[row, col].axvline(x=7.0, color='r', linestyle='--')
                # ax_apf_psd[row, col].axvline(x=14.0, color='r', linestyle='--')
                # ax_apf_psd[row, col].axvline(x=CoG_pkg, color='green', linestyle=':')


                ax_apf_psd[row, col].set_xlim(0.0, 40.0)
                ax_apf_psd[row, col].set_title(f'S{subj}_{stim}Hz_{stg}_IC{IC}')
                print(id_enum)

            fig_apf_psd.tight_layout()
            fig_apf_psd.savefig('Peak_detection_simplified.jpg')

            import pdb;pdb.set_trace()











        # featmat is the feature matrix of shape #comp *#subj * #stim *len_feat
        assert len(IC_list) == feat_mat.shape[0], 'For updating IC list, feat should be regenerate'
        len_obs = np.product(feat_mat.shape[:3])
        len_feat = np.product(feat_mat.shape[3:])

        ref_label = np.empty((len_obs, 3))
        cnt = 0

        for id_comp, val_comp in enumerate(IC_list):
        # for id_comp in range(n_comp):
            for id_subj, subj in enumerate(subj_list):
                for id_stim, stim in enumerate(stim_list):
                    ref_label[cnt] = [id_comp, subj, stim]
                    cnt += 1
        new_shape = [len_obs]
        new_shape.extend(feat_mat.shape[3:])
        tmp = np.reshape(feat_mat, new_shape)

        # feat_vec is the vectorized featmat, i.e., each row represents one
        # comp&subj& stim's feature to cluster
        feat_vec = np.reshape(tmp, (len_obs, len_feat))

        est_flag = 'KMeans'  # 'DBSCAN'
        if est_flag == 'DBSCAN':
            from sklearn.metrics.pairwise import euclidean_distances
            dist_mat = euclidean_distances(feat_vec, feat_vec)
            threshold = np.quantile(dist_mat, 0.01)
            est = DBSCAN(eps=threshold, min_samples=10).fit(feat_vec)
            all_labels = est.labels_
        elif est_flag == 'KMeans':
            est = KMeans(n_clusters=n_clusters,
                        max_iter=300, n_init=10).fit(feat_vec)
            all_labels = est.labels_
        elif est_flag == 'OPTICS':
            from sklearn.cluster import OPTICS
            est = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)
            est.fit(feat_vec)
            all_labels = est.labels_[est.ordering_]

        label_set = list(set(all_labels))
        ic_cluster = np.empty((n_clusters, n_comp))
        for id_cluster in range(n_clusters):
            ic_cluster[id_cluster] = np.bincount(
                ref_label[all_labels == id_cluster][:, 0].astype('int'),
                minlength=n_comp)
        # import pdb;pdb.set_trace()

        max_IC = False  # mutual exclusive IC topo or above threshold
        if max_IC:
            max_cluster = np.argmax(ic_cluster, axis=0)
        else:
            above_T_cluster = np.where(ic_cluster > np.int(32/n_clusters*T_ratio))

        # for the clustered scatter plots, plot in 3d space or not
        plt_3d = True

        from sklearn.manifold import TSNE
        if plt_3d:
            projection = '3d'
            tsne = TSNE(n_components=3, random_state=0)
        else:
            projection = None
            tsne = TSNE(n_components=2, random_state=0)

        n_topo_row = 6
        if load_topo:
            print('Loading IC topo')
            pca_flag = False
            if pca_flag:
                import scipy
                with open('{0}PCA_76_SOBI_all{1}.pkl'.format(path, sobi_suffix), 'rb') as f_in:
                    res, pca_obj = pickle.load(f_in)

                with open('{0}All_concat{1}.pkl'.format(
                        path, sobi_suffix), 'rb') as f_in_2:
                    all_data = pickle.load(f_in_2)

                W_PCA = pca_obj.components_.T[:, :76]
                source_, A_ICA, W_ICA = res
                import pdb;pdb.set_trace()

                W_ = W_PCA.dot(W_ICA.T)
                A_ = np.linalg.pinv(W_).T
                S_s = np.cov(source_)
                S_x = np.cov(all_data)
                A_ = S_x.dot(W_.dot(scipy.linalg.pinv2(S_s)))
                # source_ == W_.T.dot(all_data)

                import pdb;pdb.set_trace()
            else:
                with open('{0}SOBI_all{1}.pkl'.format(path, sobi_suffix), 'rb') as f_in:
                    res = pickle.load(f_in)
                source_, A_, W_ = res

            with open('{0}SOBI_all{1}.pkl'.format(path, sobi_suffix), 'rb') as f_in:
                res = pickle.load(f_in)
            source_, A_, W_ = res

            with open('{0}S{1}_Ses{2}_{3}.pkl'.format(
                    path, str(1), str(1), new_band_name[0]), 'rb') as f:
                tmp = pickle.load(f)
            info = tmp[0].pick(picks='eeg').info

            drop_bad = True
            if drop_bad:
                bads = ['FC3', 'FFC5h', 'FCC5h', 'FFC3h', 'FCC3h', 'T7', 'FTT7h',
                        'TTP7h', 'FT7', 'FC5', 'C5', 'C3', 'F9', 'F5', 'CCP5h',
                        'AFF5h', 'FC1', 'P9', 'TP10', 'FFT7h', 'TP7', 'FFT9h',
                        'F3', 'FT9']
                info = tmp[0].pick(picks='eeg').info
                loc_chn = [info.ch_names.index(chn) for chn in bads]

                select_A = np.delete(A_, tuple(loc_chn), axis=0)
                info = tmp[0].pick(picks='eeg').drop_channels(bads).info
            else:
                select_A = A_


        if psd_plot:
            height_ratios = [10, 10, 10]
            height_ratios.extend([5] * n_topo_row)

            fig = plt.figure(figsize=(n_clusters*18, sum(height_ratios)))
            gs = gridspec.GridSpec(len(height_ratios), n_clusters*18,
                                    height_ratios=height_ratios)

            # if feat_type != 'BP_3':
                # fig = plt.figure(figsize=(n_clusters*18, sum(height_ratios)))
                # gs = gridspec.GridSpec(len(height_ratios), n_clusters*18,
                                       # height_ratios=height_ratios)
            # else:
                # fig = plt.figure()
                # gs = gridspec.GridSpec(3, n_clusters*18)

            ax_tsne = []
            for i in range(6):
                if plt_3d and i < 4:
                    ax_tmp = fig.add_subplot(gs[0, i*12: i*12+12],
                                            projection=projection)
                else:
                    ax_tmp = fig.add_subplot(gs[0, i*12: i*12+12])
                ax_tsne.append(ax_tmp)

            ax_psd = []
            ax_psd_ref = []
            for i in range(n_clusters):
                ax_tmp = fig.add_subplot(gs[1, i*18: i*18+18])
                ax_ref_tmp = fig.add_subplot(gs[2, i*18: i*18+18])
                ax_psd.append(ax_tmp)
                ax_psd_ref.append(ax_ref_tmp)

            ax_topo = []
            ax_cbar = []
            for id_cluster in range(n_clusters):
                ax_cluster = []
                for id_topo in range(np.int(n_topo_row*3)):
                    irow, icol = divmod(id_topo, 3)
                    grid_col = id_cluster*18 + icol*6
                    ax_tmp_1 = fig.add_subplot(gs[irow+3, grid_col:grid_col+5])
                    ax_tmp_2 = fig.add_subplot(gs[irow+3, grid_col+5])
                    ax_cluster.append((ax_tmp_1, ax_tmp_2))
                ax_topo.append(ax_cluster)

            for id_label, label_val in enumerate(label_set):
                ind = np.where(all_labels == label_val)[0]
                tmp = list(
                    map(lambda x: ind_convert(x, subj_list=subj_list,
                                              stim_list=stim_list,
                                              IC_list=IC_list), ind))
                mask = df[['IC', 'subject', 'stim_freq']].agg(tuple, 1).isin(tmp)
                subset_df = all_df[mask]

                # the length of whole spectra is 144
                assert len(subset_df) == len(tmp) * 144, 'check subset df length'

                # Plot the first row of psd, i.e., not refer to anything when
                # ref flag is false. Otherwise, it plots the referred figures
                sns.lineplot(x='freq_val', y='diff_val', hue='stim_freq',
                             data=subset_df, ax=ax_psd[id_label], palette="tab10")
                ax_psd[id_label].set_title('Cluster {0}'.format(str(label_val)))

                # Plot the refered plots when the refer flag is false, note
                # that it is based on avg. 0Hz across all sessions & subjects
                if not ref_no_stim_cluster_flag:
                    no_stim_df = subset_df.loc[subset_df['stim_freq'] == 0].copy()
                    mean_no_stim = no_stim_df.pivot(index='freq_val',
                                                    columns=['subject', 'IC'],
                                                    values='diff_val').mean(axis=1)
                    if len(mean_no_stim) == 0:
                        continue
                    ref_stim_df = subset_df.loc[subset_df['stim_freq'] != 0].copy()
                    pre_diff_val = ref_stim_df['diff_val'].values
                    ref_stim_df['diff_val'] = pre_diff_val - \
                        np.tile(mean_no_stim, np.int(pre_diff_val.shape[0] / 144))

                    new_cm = sns.color_palette("tab10")[1:]
                    # In some cluster, it is possible that no certain stim
                    # appears, i.e., it is not always equal to 3
                    nr_color = np.unique(ref_stim_df['stim_freq']).shape[0]
                    sns.lineplot(x='freq_val', y='diff_val', hue='stim_freq',
                                 data=ref_stim_df, ax=ax_psd_ref[id_label],
                                 palette=new_cm[:nr_color])

                    ax_psd_ref[id_label].set_title(
                        'Cluster {0} (ref. to no-stim)'.format(str(label_val)))
                print('Cluster {0}/{1}'.format(str(id_label), str(n_clusters)))

#             if feat_type == 'BP_3':
                # X_2d = feat_vec.copy()
                # colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple'
                # for i, c in zip(label_set, colors):
                    # data = list(map(tuple, X_2d[all_labels == i]))
                    # ax_tsne[0].scatter(*zip(*data), c=c, label=i)
                    # ax_tsne[0].set_title('Cluster based')
                    # ax_tsne[0].legend()


            # fig_list.tight_layout()
            # fig_list.savefig(
            #     'psd_avg_{0}_{1}{2}.jpg'.format(feat_type, est_flag, suffix))

            # Load the SOBI ICA topos
            try:
                # all_df = pd.read_pickle('{0}SOBI_all_df.pkl'.format(path))
                # all_df has 139.5 for each IC and all_df has 144
                for id_cluster in range(n_clusters):
                    if max_IC:
                        cluster_IC = np.where(max_cluster == id_cluster)[0]
                    else:
                        cluster_IC = above_T_cluster[1][
                            above_T_cluster[0] == id_cluster]

                    for i, id_IC in enumerate(cluster_IC):
                        if i > np.int(n_topo_row*3 - 1):
                            break
                        im_tmp = topo(select_A[:, IC_list[id_IC]], info,
                                    axes=ax_topo[id_cluster][i][0], show=False)
                        fig.colorbar(im_tmp[0], cax=ax_topo[id_cluster][i][1],
                                    orientation='vertical')
                        ax_topo[id_cluster][i][0].set_title(
                            'IC_{0}'.format(str(IC_list[id_IC])))
                        print(str(id_cluster) + ' ' + str(i))
            except:
                import pdb;pdb.set_trace()

            fig.tight_layout()
            fig.savefig('{0}_cluster_{1}_{2}{3}{4}.jpg'.format(est_flag, str(n_clusters), feat_type, ref_suffix, fig_sobi_suffix))



        if scatter_heatmap_plot:
            ncol = 6
            if not 'ax_tsne ' in vars() and not 'ax_tsne' in globals():
                ax_tsne = [''] * ncol
                fig = plt.figure(figsize=(36, 5))
                for i in range(6):
                    if i < 4:
                        ax_tsne[i] = fig.add_subplot(1, ncol, i + 1,
                                                    projection=projection)
                    else:
                        ax_tsne[i] = fig.add_subplot(1, ncol, i + 1)
            if feat_type == 'BP_3':
                tsne_trans_flag = False
            else:
                tsne_trans_flag = True
            if tsne_trans_flag:
                X_2d = tsne.fit_transform(feat_vec)
            else:
                X_2d = feat_vec.copy()
            colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple'
            for i, c in zip(label_set, colors):
                # select the points which are classified to specific label
                data = list(map(tuple, X_2d[all_labels == i]))
                # Group x, y, z will be individual tuples, e.g.,
                # array([[1, 2],[2, 3],[4, 5]]) -> [(1, 2, 4), (2, 3, 5)]
                ax_tsne[0].scatter(*zip(*data), c=c, label=i)
                ax_tsne[0].set_title('Cluster based')
                ax_tsne[0].legend()

            for i, c in zip(subj_list, colors):
                data = list(map(tuple, X_2d[np.where(ref_label[:, 1] == i)[0]]))
                ax_tsne[1].scatter(*zip(*data), c=c, label=i)
                ax_tsne[1].set_title('Subject based')
                ax_tsne[1].legend()

            for i, c in zip(stim_list, colors):
                data = list(map(tuple, X_2d[np.where(ref_label[:, 2] == i)[0]]))
                ax_tsne[2].scatter(*zip(*data), c=c, label=i)
                ax_tsne[2].set_title('Stim based')
                ax_tsne[2].legend()

            from colour import Color
            red = Color("blue")
            colors = list(red.range_to(Color("black"), n_comp))
            for i, c in zip(IC_list, colors):
                data = list(map(tuple, X_2d[np.where(ref_label[:, 0] == i)[0]]))
                if len(data) != 0:
                    ax_tsne[3].scatter(*zip(*data), c=c.hex, label=i)
                else:
                    continue
            ax_tsne[3].set_title('IC based')

            for i, order in enumerate(['C', 'F']):
                xlabel_name = []
                if order == 'F':
                    suffix += '_group_by_stim'
                    for id_stim in stim_list:
                        for id_subj in subj_list:
                            xlabel_name.append(
                                'S{0}_{1}Hz'.format(str(id_subj), str(id_stim)))
                elif order == 'C':
                    suffix += '_group_by_subject'
                    for id_subj in subj_list:
                        for id_stim in stim_list:
                            xlabel_name.append(
                                'S{0}_{1}Hz'.format(str(id_subj), str(id_stim)))

                label_mat = all_labels.reshape(n_comp, len(subj_list), len(stim_list))
                label_mat = label_mat.reshape(n_comp, len(subj_list)*len(stim_list), order=order)
                tmp = sns.heatmap(data=label_mat, ax=ax_tsne[4+i],
                                xticklabels=xlabel_name)
                ax_tsne[4+i].xaxis.tick_top()  # x axis on top
                ax_tsne[4+i].xaxis.set_label_position('top')
                tmp.set_yticklabels(tmp.get_yticklabels(), fontsize=8, rotation=45)
                tmp.set_xticklabels(tmp.get_xticklabels(), fontsize=8, rotation=45)
                ax_tsne[4+i].set_title(suffix[1:])

                # fig_hm, ax_hm = plt.subplots(1, 1, figsize=(32, 120))
            # fig.savefig('hm_{0}_{1}{2}.jpg'.format(feat_type, est_flag, suffix))
        if only_spec_plot:

            width_ratios = [10, 10]
            height_ratios = [5, 5]

            len_w = len(width_ratios)
            len_h = len(height_ratios)

            fig_onlyspec = plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))
            gs = gridspec.GridSpec(len(height_ratios), len(width_ratios),
                                   height_ratios=height_ratios,
                                   width_ratios=width_ratios)

            ax_onlyspec = np.empty((len_h, len_w), dtype='object')
            for i in range(len_h):
                for j in range(len_w):
                    ax_onlyspec[i, j] = fig_onlyspec.add_subplot(gs[i, j])

            for id_label, label_val in enumerate(label_set):

                print(id_label)
                id_row, id_col = divmod(id_label, 2)
                ind = np.where(all_labels == label_val)[0]
                tmp = list(
                    map(lambda x: ind_convert(x, subj_list=subj_list,
                                            stim_list=stim_list,
                                            IC_list=IC_list), ind))
                mask = diff_all_df[['IC', 'subject', 'stim_freq']].agg(tuple, 1).isin(tmp)
                subset_df = diff_all_df[mask]
                subset_df['stim_freq'].replace({0: 'Sham', 4: '4 Hz',
                                                10: '10 Hz', 40: '40 Hz'},
                                               inplace=True)
                subset_df.rename(columns={'stim_freq': 'tACS frequency',
                                          'freq_val': 'Frequency / Hz',
                                          'diff_val': 'Difference Spectrum / dB'},
                                 inplace=True)

                sns.lineplot(x='Frequency / Hz', y='Difference Spectrum / dB',
                             hue='tACS frequency', data=subset_df,
                             ax=ax_onlyspec[id_row, id_col], palette="tab10")
                ax_onlyspec[id_row, id_col].set_title('Cluster {0}'.format(str(label_val)), fontsize=24)
                ax_onlyspec[id_row, id_col].tick_params(axis='both', which='major', labelsize=20)
                ax_onlyspec[id_row, id_col].xaxis.get_label().set_fontsize(20)
                ax_onlyspec[id_row, id_col].yaxis.get_label().set_fontsize(20)
                ax_onlyspec[id_row, id_col].legend(prop=dict(size=20))
            fig_onlyspec.tight_layout()
            fig_onlyspec.savefig(fig_root+'Onlyspec{0}_cluster_{1}_{2}{3}{4}.jpg'.format(est_flag, str(n_clusters), feat_type, ref_suffix, fig_sobi_suffix))

            import pdb;pdb.set_trace()


        if topo_spec_plot:

            nr_comp = len(IC_list)
            width_ratios = [5, 1, 9]
            width_ratios.extend([9] * n_clusters)
            height_ratios = [5] * nr_comp

            max_pixel = max(sum(width_ratios), sum(height_ratios))
            if max_pixel * 200 > 2**16:
                down_ratio = np.ceil(max_pixel * 200 / 2**16)
            else:
                down_ratio = 1

            width_ratios /= down_ratio
            height_ratios /= down_ratio

            len_w = len(width_ratios)
            len_h = len(height_ratios)

            fig_spectopo = plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))
            import pdb;pdb.set_trace()
            gs = gridspec.GridSpec(len(height_ratios), len(width_ratios),
                                   height_ratios=height_ratios,
                                   width_ratios=width_ratios)

            ax_spectopo = np.empty((len_h, len_w), dtype='object')
            for i in range(len_h):
                for j in range(len_w):
                    ax_spectopo[i, j] = fig_spectopo.add_subplot(gs[i, j])

            for id_IC, IC in enumerate(IC_list):

                # Plot topo with cbar

                im_tmp = topo(select_A[:, IC], info,
                              axes=ax_spectopo[id_IC][0], show=False)
                fig_spectopo.colorbar(im_tmp[0], cax=ax_spectopo[id_IC][1],
                             orientation='vertical')
                ax_spectopo[id_IC][0].set_title(
                    'IC_{0}'.format(str(IC_list[id_IC])))

                IC_df = diff_all_df.loc[diff_all_df['IC'] == IC].copy()
                sns.lineplot(x='freq_val', y='diff_val', hue='stim_freq',
                            data=IC_df, ax=ax_spectopo[id_IC, 2], palette="tab10")
                ax_spectopo[id_IC, 2].set_title('IC {0} All Clusters'.format(str(IC)))
                print(id_IC)

            for id_label, label_val in enumerate(label_set):

                print(id_label)
                ind = np.where(all_labels == label_val)[0]
                tmp = list(
                    map(lambda x: ind_convert(x, subj_list=subj_list,
                                            stim_list=stim_list,
                                            IC_list=IC_list), ind))
                mask = diff_all_df[['IC', 'subject', 'stim_freq']].agg(tuple, 1).isin(tmp)
                subset_df = diff_all_df[mask]

                assert len(subset_df) == len(tmp) * 144, 'check subset df length'

                for id_IC, IC in enumerate(IC_list):
                    IC_df = subset_df.loc[subset_df['IC'] == IC].copy()
                    sns.lineplot(x='freq_val', y='diff_val', hue='stim_freq',
                                 data=IC_df, ax=ax_spectopo[id_IC, id_label + 3], palette="tab10")
                    ax_spectopo[id_IC, id_label + 3].set_title('IC {0} Cluster {1}'.format(str(IC), str(label_val)))
            fig_spectopo.tight_layout()
            fig_spectopo.savefig(fig_root+'Spectopo_{0}_cluster_{1}_{2}{3}{4}.jpg'.format(est_flag, str(n_clusters), feat_type, ref_suffix, fig_sobi_suffix))

            import pdb;pdb.set_trace()


        if source_local_plot:
            from jxu.model.source_localization.IC_source_localize import spatial_pattern_to_source as SP2S
            # IC_list = [18, 20, 98]

            label = ['IC_' + str(i) for i in IC_list]
            rownames = ['Cluster_' + str(i) for i in range(4) ]
            all_IC = select_A[:, IC_list]

            # # For generaling individual IC source local.
            # sl_paras = {'depth': [0.2, 0.4, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8,
                                    # 5.6, 6.4, 8.0, 9.6, 12.8, 16.0, 20, 25,
                                    # 30, 40, 60],
                        # 'loose': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}
            # SP2S(IC=all_IC, IC_label=label, save_plot=True,
                 # separate_plot=True, picks=bads, prefix=fig_root,
                 # tuning_paras=sl_paras)
            weights_mat = ic_cluster / 32.0
            import pdb;pdb.set_trace()
            SP2S(IC=all_IC, IC_label=label, save_plot=True,
                 separate_plot=False, weights=weights_mat, picks=bads,
                 prefix=fig_root, tuning_paras=None)

            import pdb;pdb.set_trace()

        if tuning_sl_paras_plot:
            from itertools import product
            fig_root += 'source_local_tuning_para/'
            # IC_list = [18, 20, 98]
            # each_label = IC_list[2]
            for each_label in IC_list:

                # sl_paras = {'depth': [1.6, 2.4, 3.2, 4.0, 4.8, 5.6],

                sl_paras = {'depth': [0.2, 0.4, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8,
                                      5.6, 6.4, 8.0, 9.6, 12.8, 16.0, 20, 25,
                                      30, 40, 60],
                            'loose': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}
                prefix = fig_root

                width_ratios = [5]
                width_ratios.extend([10] * len(sl_paras['loose']))
                height_ratios = [5]
                height_ratios.extend([5] * 2 * len(sl_paras['depth']))
                len_w = len(width_ratios)
                len_h = len(height_ratios)

                fig_sl_tune = plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))
                gs = gridspec.GridSpec(len(height_ratios), len(width_ratios),
                                    height_ratios=height_ratios,
                                    width_ratios=width_ratios)
                ax_sl_tune = np.empty((len_h, len_w), dtype='object')
                for i in range(len_h):
                    for j in range(len_w):
                        if divmod(i, 2)[1] != 0 and j == 0:  # only for odd rows
                            ax_sl_tune[i, j] = fig_sl_tune.add_subplot(gs[i:i+2, j])
                            ax_sl_tune[i+1, j] = None
                            continue
                        ax_sl_tune[i, j] = fig_sl_tune.add_subplot(gs[i, j])

                im_tmp = topo(select_A[:, each_label], info, axes=ax_sl_tune[0, 0],
                            show=False)
                ax_sl_tune[0, 0].set_title(f'IC_{str(each_label)}')

                assert [*sl_paras.keys()] == ['depth', 'loose'], 'Paras only allowed depth and loose'
                for ind, (depth, loose) in enumerate(product(*sl_paras.values())):
                    id_depth, id_loose = divmod(ind, len(sl_paras['loose']))
                    para_suffix = f'depth_{depth}_loose_{loose}'
                    for id_view, view in enumerate(['lateral', 'medial']):
                        print(f'{ind}_{depth}_{loose}')
                        data = plt.imread(f'{prefix}_IC_{each_label}_{view}_{para_suffix}.jpg')
                        ax_sl_tune[id_depth * 2 + id_view + 1, id_loose + 1].imshow(data)
                        ax_sl_tune[id_depth * 2 + id_view + 1, id_loose + 1].set_title(f'{view}')

                for i, depth in enumerate(sl_paras['depth']):
                    ax_sl_tune[i*2+1, 0].text(x=0.3, y=0.5, s=f'Depth_{depth}', fontsize=30)
                    # ax_sl_tune[i*2+1, 0].set_axis_off()
                    ax_sl_tune[i*2+2, 0].set_axis_off()

                for j, loose in enumerate(sl_paras['loose']):
                    ax_sl_tune[0, j+1].text(x=0.45, y=0.5, s=f'Loose_{loose}', fontsize=30)
                    # ax_sl_tune[0, j+1].set_axis_off()
                fig_sl_tune.tight_layout()
                fig_sl_tune.savefig(f'IC_{each_label}.jpg')
            import pdb;pdb.set_trace()


        if psd_source_cluster_plot:
            assert source_local_plot, 'Must regenerate Cluster source estimate'

            width_ratios = [10] * 4
            height_ratios = [5] * 5
            len_w = len(width_ratios)
            len_h = len(height_ratios)

            fig_psd_source = plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))
            gs = gridspec.GridSpec(len(height_ratios), len(width_ratios),
                                   height_ratios=height_ratios,
                                   width_ratios=width_ratios)
            ax_psd_source = np.empty((len_h - 1, len_w), dtype='object')
            ax_bar = np.empty((2, 1), dtype='object')
            for i in range(len_h):
                if i == 0:
                    ax_bar[0] = fig_psd_source.add_subplot(gs[0, :2])
                    ax_bar[1] = fig_psd_source.add_subplot(gs[0, 2:])
                    continue
                for j in range(len_w):
                    ax_psd_source[i-1, j] = fig_psd_source.add_subplot(gs[i, j])

            for id_label, label_val in enumerate(label_set):
                ind = np.where(all_labels == label_val)[0]
                tmp = list(
                    map(lambda x: ind_convert(x, subj_list=subj_list,
                                              stim_list=stim_list,
                                              IC_list=IC_list), ind))
                # df is the original copy of all_df, i.e., not refer
                # to anything
                mask = df[['IC', 'subject', 'stim_freq']].agg(tuple, 1).isin(tmp)
                subset_df = all_df[mask]

                # the length of whole spectra is 144
                assert len(subset_df) == len(tmp) * 144, 'check subset df length'

                # Plot the first row of psd, i.e., not refer to anything when
                # ref flag is false. Otherwise, it plots the referred figures
                sns.lineplot(x='freq_val', y='diff_val', hue='stim_freq',
                             data=subset_df, ax=ax_psd_source[id_label, 0], palette="tab10")
                ax_psd_source[id_label, 0].set_title('Cluster {0}'.format(str(label_val)))

                # Plot the refered plots when the refer flag is false, note
                # that it is based on avg. 0Hz across all sessions & subjects
                if not ref_no_stim_cluster_flag:
                    no_stim_df = subset_df.loc[subset_df['stim_freq'] == 0].copy()
                    mean_no_stim = no_stim_df.pivot(index='freq_val',
                                                    columns=['subject', 'IC'],
                                                    values='diff_val').mean(axis=1)
                    if len(mean_no_stim) == 0:
                        continue
                    ref_stim_df = subset_df.loc[subset_df['stim_freq'] != 0].copy()
                    pre_diff_val = ref_stim_df['diff_val'].values
                    ref_stim_df['diff_val'] = pre_diff_val - \
                        np.tile(mean_no_stim, np.int(pre_diff_val.shape[0] / 144))

                    new_cm = sns.color_palette("tab10")[1:]
                    # In some cluster, it is possible that no certain stim
                    # appears, i.e., it is not always equal to 3
                    nr_color = np.unique(ref_stim_df['stim_freq']).shape[0]
                    sns.lineplot(x='freq_val', y='diff_val', hue='stim_freq',
                                 data=ref_stim_df, ax=ax_psd_source[id_label, 1],
                                 palette=new_cm[:nr_color])

                    ax_psd_source[id_label, 1].set_title(
                        'Cluster {0} (ref. to no-stim)'.format(str(label_val)))

                for id_view, (view, annot) in enumerate(zip(['lateral', 'medial'], ['Outside -> Inside', 'Inside -> Outside'])):
                    para_suffix = 'Depth_9.6_Loose_0.2'
                    data = plt.imread(f'{fig_root}Cluster_{id_label}_{view}_{para_suffix}.jpg')
                    ax_psd_source[id_label, 2 + id_view].imshow(data)
                    ax_psd_source[id_label, 2 + id_view].set_title(f'Cluster_{id_label}_{view} ({annot})')

                print('Cluster {0}/{1}'.format(str(id_label), str(n_clusters)))


            # --------- Barplot of Cluster and IC related weights -------------
            ic_df = pd.DataFrame(ic_cluster, columns=label, index=rownames)
            ic_df.columns.name = 'IC'
            ic_df.index.name = 'Cluster'
            ic_df_series = ic_df.stack().reset_index(level=1, name='value').reset_index()
            sns.barplot(data=ic_df_series, x='Cluster', y='value', hue='IC', ax=ax_bar[0, 0])
            ax_bar[0, 0].set_title('Group by IC')
            sns.barplot(data=ic_df_series, x='IC', y='value', hue='Cluster', ax=ax_bar[1, 0])
            ax_bar[1, 0].set_title('Group by Cluster')
            fig_psd_source.tight_layout()
            fig_psd_source.savefig('Barplot_IC_Cluster__.jpg')
            import pdb;pdb.set_trace()

        if individual_topo_sl_plot:
            # ------------ Topo and source local for each IC -----------------
            prefix = fig_root
            nr_comp = len(IC_list)
            width_ratios = [5, 1, 10, 10]
            height_ratios = [5] * nr_comp
            len_w = len(width_ratios)
            len_h = len(height_ratios)

            fig_topo_sl = plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))
            gs = gridspec.GridSpec(len(height_ratios), len(width_ratios),
                                   height_ratios=height_ratios,
                                   width_ratios=width_ratios)
            ax_topo_sl = np.empty((len_h, len_w), dtype='object')
            for i in range(len_h):
                for j in range(len_w):
                    ax_topo_sl[i, j] = fig_topo_sl.add_subplot(gs[i, j])

            for id_IC, IC in enumerate(IC_list):

                # Plot topo with cbar

                im_tmp = topo(select_A[:, IC], info,
                              axes=ax_topo_sl[id_IC][0], show=False)
                fig_topo_sl.colorbar(im_tmp[0], cax=ax_topo_sl[id_IC][1],
                                     orientation='vertical')
                ax_topo_sl[id_IC][0].set_title(
                    'IC_{0}'.format(str(IC_list[id_IC])))
                for id_view, (view, annot) in enumerate(zip(['lateral', 'medial'], ['Outside -> Inside', 'Inside -> Outside'])):

                    para_suffix = f'depth_{str(9.6)}_loose_{str(0.6)}'
                    data = plt.imread(f'{prefix}_IC_{IC}_{view}_{para_suffix}.jpg')
                    # data = plt.imread(f'_IC_{IC}_{view}.jpg')
                    ax_topo_sl[id_IC, 2 + id_view].imshow(data)
                    ax_topo_sl[id_IC, 2 + id_view].set_title(f'IC_{IC}_{view} ({annot})')

                print(id_IC)

            fig_topo_sl.tight_layout()
            fig_topo_sl.savefig(f'Individual_IC_source_localization_{para_suffix}.jpg')


        IC_topo_spec_plot = True
        if IC_topo_spec_plot:
            nr_comp = len(IC_list)
            width_ratios = [5, 1, 10, 10, 10, 10, 10, 10]
            height_ratios = [5] * nr_comp

            max_pixel = max(sum(width_ratios), sum(height_ratios))
            if max_pixel * 200 > 2**16:
                down_ratio = np.ceil(max_pixel * 200 / 2**16)
            else:
                down_ratio = 1

            width_ratios = [i / down_ratio for i in width_ratios]
            height_ratios = [i / down_ratio for i in height_ratios]

            len_w = len(width_ratios)
            len_h = len(height_ratios)

            fig_spectopo = plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))
            import pdb;pdb.set_trace()
            gs = gridspec.GridSpec(len(height_ratios), len(width_ratios),
                                   height_ratios=height_ratios,
                                   width_ratios=width_ratios)

            ax_spectopo = np.empty((len_h, len_w), dtype='object')
            for i in range(len_h):
                for j in range(len_w):
                    ax_spectopo[i, j] = fig_spectopo.add_subplot(gs[i, j])

            for id_IC, IC in enumerate(IC_list):

                # Plot topo with cbar

                im_tmp = topo(select_A[:, IC], info,
                              axes=ax_spectopo[id_IC][0], show=False)
                fig_spectopo.colorbar(im_tmp[0], cax=ax_spectopo[id_IC][1],
                             orientation='vertical')
                ax_spectopo[id_IC][0].set_title(
                    'IC_{0}'.format(str(IC_list[id_IC])))

                IC_df = diff_all_df.loc[diff_all_df['IC'] == IC].copy()
                sns.lineplot(x='freq_val', y='diff_val', hue='stim_freq',
                            data=IC_df, ax=ax_spectopo[id_IC, 2], palette="tab10")
                ax_spectopo[id_IC, 2].set_title('IC {0} Difference'.format(str(IC)))
                print(id_IC)

                ref_no_stim_flag = True
                if ref_no_stim_flag:
                    no_stim_df = IC_df.loc[IC_df['stim_freq'] == 0].copy()
                    mean_no_stim = no_stim_df.pivot(index='freq_val',
                                                    columns=['subject', 'IC'],
                                                    values='diff_val').mean(axis=1)
                    ref_stim_df = IC_df.loc[IC_df['stim_freq'] != 0].copy()
                    pre_diff_val = ref_stim_df['diff_val'].values
                    ref_stim_df['diff_val'] = pre_diff_val - \
                        np.tile(mean_no_stim, np.int(pre_diff_val.shape[0] / 144))
                    ref_suffix = '_ref'

                    sns.lineplot(x='freq_val', y='diff_val', hue='stim_freq',
                                 data=ref_stim_df, ax=ax_spectopo[id_IC, 3], palette="tab10")
                    ax_spectopo[id_IC, 3].set_title('IC {0} Difference_ref_to_sham'.format(str(IC)))


                for id_stg, stg in enumerate(['pre', 'post']):
                    tmp_df = all_df.loc[(all_df['IC'] == IC) & (all_df['session'] == stg)].copy()
                    tmp_df = tmp_df.drop(tmp_df[tmp_df['freq_val'] < 3].index)

                    sns.lineplot(x='freq_val', y='psd_val', hue='stim_freq',
                                data=tmp_df, ax=ax_spectopo[id_IC, id_stg + 4], palette="tab10")
                    ax_spectopo[id_IC, id_stg + 4].set_title('IC {0} {1}'.format(str(IC), stg))

                    if ref_no_stim_flag:
                        no_stim_df = tmp_df.loc[tmp_df['stim_freq'] == 0].copy()
                        mean_no_stim = no_stim_df.pivot(index='freq_val',
                                                        columns=['subject', 'IC'],
                                                        values='psd_val').mean(axis=1)
                        ref_stim_df = tmp_df.loc[tmp_df['stim_freq'] != 0].copy()
                        pre_diff_val = ref_stim_df['psd_val'].values
                        ref_stim_df['psd_val'] = pre_diff_val - \
                            np.tile(mean_no_stim, np.int(pre_diff_val.shape[0] / 137))
                        ref_suffix = '_ref'

                        sns.lineplot(x='freq_val', y='psd_val', hue='stim_freq',
                                     data=ref_stim_df, ax=ax_spectopo[id_IC, id_stg + 6], palette="tab10")
                        ax_spectopo[id_IC, id_stg + 6].set_title('IC {0} {1}_ref_to_sham'.format(str(IC), stg))


            fig_spectopo.tight_layout()
            fig_spectopo.savefig(fig_root+'IC_Spectopo_{0}_start_3Hz_ref.jpg'.format(fig_sobi_suffix))

            import pdb;pdb.set_trace()





            import pdb;pdb.set_trace()


        # import pdb;pdb.set_trace()
import pdb;pdb.set_trace()

