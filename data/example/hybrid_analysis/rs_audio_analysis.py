#========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2021-05-09 14:30:24
# Name       : rs_audio_analysis.py
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

from matplotlib import gridspec
from mne.viz import plot_topomap as topo
from jxu.data.example.hybrid_analysis.extract_behav import behav


def ind_convert(x, subj_list, stim_list):

    IC, tmp = divmod(x, len(subj_list)*len(stim_list))
    subj, stim = divmod(tmp, len(stim_list))

    return IC, subj_list[subj], stim_list[stim]

path = '/home/jxu/File/Data/NIBS/Stage_one/EEG/Processed/RS_epoch/'

# all_df = pd.read_pickle('{0}SOBI_all_df_default.pkl'.format(path))
sobi_suffix = '_BP(3-70)'
all_df = pd.read_pickle('{0}SOBI_all_df_default{1}.pkl'.format(path, sobi_suffix))

diff_all_df = all_df.loc[all_df['session'] == 'post'].copy()
diff_all_df['diff_val'] = all_df.loc[all_df['session'] == 'post'].psd_val.values -\
    all_df.loc[all_df['session'] == 'pre'].psd_val.values

bands_list = [(1.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 30.0), (30.0, 70.0)]
bands_list_name = ['delta', 'theta', 'alpha', 'beta', 'gamma']

new_band_name = ['BP(1-70)', 'delta(1-4)', 'theta(4-8)', 'Alpha(8-12)',
                   'L_Beta(12-20)', 'H_Beta(20-30)', 'L_Gamma(30-70)']
n_comp = 120

nr_comp = 120
stim_list = [0, 4, 10, 40]
subj_list = [1, 2, 3, 5, 6, 7, 8, 10]

feat_type = 'single_band'  #  fix_nr_sample'BP', 'all_spectra', 'single_band'
id_band = 4

n_clusters = 6

# threshold to assign IC to cluster. 1 means above avg. samples
T_ratio = 1.0

# Flag: refer to no-stim condition
# If True: only plot the psd of 4Hz, 10Hz and 40Hz activities referred to avg.
# 0Hz activity across all sessions & subjects
ref_no_stim_cluster_flag = False

heatmap_plot = False
psd_plot = False
behav_plot = False

for feat_type in ['BP_3', 'BP']:  # , 'all_spectra'
    for n_clusters in [4]:

        df = diff_all_df.copy()

        if feat_type != 'single_band':
            suffix = ''
        else:
            suffix = '_' + bands_list_name[id_band]

        if ref_no_stim_cluster_flag:
            no_stim_df = df.loc[df['stim_freq'] == 0].copy()
            mean_no_stim = no_stim_df.pivot(index='freq_val',
                                            columns=['subject', 'IC'],
                                            values='diff_val').mean(axis=1)
            ref_stim_df = df.loc[df['stim_freq'] != 0].copy()
            pre_diff_val = ref_stim_df['diff_val'].values
            ref_stim_df['diff_val'] = pre_diff_val - \
                np.tile(mean_no_stim, np.int(pre_diff_val.shape[0] / 144))
            df_to_featmat = ref_stim_df
            ref_suffix = '_ref'
            stim_list = stim_list[1:]

            # aa.pivot(index='freq_val', values='diff_val', columns=['stim_freq', 'session', 'subject', 'IC'])
        else:
            df_to_featmat = diff_all_df
            ref_suffix = ''
        # df_to_featmat is the processed dataframe which will be used to
        # generate features matrices to classify and plot

        with open('{0}cluster_feat_df_{1}{2}{3}.pkl'.format(
                path, feat_type, suffix, ref_suffix), 'rb') as f:
            feat_mat = pickle.load(f)

        # featmat is the feature matrix of shape #comp *#subj * #stim *len_feat
        len_obs = np.product(feat_mat.shape[:3])
        len_feat = np.product(feat_mat.shape[3:])

        ref_label = np.empty((len_obs, 3))
        cnt = 0
        for id_comp in range(n_comp):
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
        ic_cluster = np.empty((n_clusters, 120))
        for id_cluster in range(n_clusters):
            ic_cluster[id_cluster] = np.bincount(
                ref_label[all_labels == id_cluster][:, 0].astype('int'),
                minlength=120)

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
        if psd_plot:
            height_ratios = [10, 10, 10]
            height_ratios.extend([5] * n_topo_row)

            fig = plt.figure(figsize=(n_clusters*18, sum(height_ratios)))
            gs = gridspec.GridSpec(len(height_ratios), n_clusters*18,
                                    height_ratios=height_ratios)

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
                                              stim_list=stim_list), ind))
                if ref_no_stim_cluster_flag:
                    mask = df_to_featmat[['IC', 'subject', 'stim_freq']].agg(tuple, 1).isin(tmp)
                    subset_df = df_to_featmat[mask]
                else:
                    # df is the original copy of diff_all_df, i.e., not refer
                    # to anything
                    mask = df[['IC', 'subject', 'stim_freq']].agg(tuple, 1).isin(tmp)
                    subset_df = diff_all_df[mask]

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
                    ref_stim_df = subset_df.loc[subset_df['stim_freq'] != 0].copy()
                    pre_diff_val = ref_stim_df['diff_val'].values
                    ref_stim_df['diff_val'] = pre_diff_val - \
                        np.tile(mean_no_stim, np.int(pre_diff_val.shape[0] / 144))

                    new_cm = sns.color_palette("tab10")[1:]
                    sns.lineplot(x='freq_val', y='diff_val', hue='stim_freq',
                                data=ref_stim_df, ax=ax_psd_ref[id_label],
                                palette=new_cm[:3])
                    ax_psd_ref[id_label].set_title(
                        'Cluster {0} (ref. to no-stim)'.format(str(label_val)))
                print('Cluster {0}/{1}'.format(str(id_label), str(n_clusters)))

            # Load the SOBI ICA topos
            # with open('{0}SOBI_all.pkl'.format(path), 'rb') as f_in:
        load_SOBI = True
        if load_SOBI:
            pca_flag = False
            if pca_flag:
                import scipy
                n_PCA = 76  # 7, 15, 76
                with open('{0}PCA_{1}_SOBI_all{2}.pkl'.format(path, str(n_PCA), sobi_suffix), 'rb') as f_in:
                    res, pca_obj = pickle.load(f_in)

                with open('{0}All_concat{1}.pkl'.format(
                        path, sobi_suffix), 'rb') as f_in_2:
                    all_data = pickle.load(f_in_2)

                W_PCA = pca_obj.components_.T[:, :n_PCA]
                source_, A_ICA, W_ICA = res

                W_ = W_PCA.dot(W_ICA.T)
                A_ = np.linalg.pinv(W_).T
                S_s = np.cov(source_)
                S_x = np.cov(all_data)
                A_ = S_x.dot(W_.dot(scipy.linalg.pinv2(S_s)))
                # source_ == W_.T.dot(all_data)
            else:
                n_PCA = 120
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
                try:
                    all_df = pd.read_pickle('{0}SOBI_all_df.pkl'.format(path))
                    for id_cluster in range(n_clusters):
                        if max_IC:
                            cluster_IC = np.where(max_cluster == id_cluster)[0]
                        else:
                            cluster_IC = above_T_cluster[1][
                                above_T_cluster[0] == id_cluster]

                        for i, id_IC in enumerate(cluster_IC):
                            if i > np.int(n_topo_row*3 - 1):
                                break
                            im_tmp = topo(select_A[:, id_IC], info,
                                        axes=ax_topo[id_cluster][i][0], show=False)
                            fig.colorbar(im_tmp[0], cax=ax_topo[id_cluster][i][1],
                                        orientation='vertical')
                            ax_topo[id_cluster][i][0].set_title(
                                'IC_{0}'.format(str(id_IC)))
                            print(str(id_cluster) + ' ' + str(i))
                except:
                    import pdb;pdb.set_trace()

        only_topo_plot = True
        if only_topo_plot:
            assert load_SOBI, 'Must turn on load_SOBI to load sobi results'
            topo_col = 8
            n_row, _ = divmod(n_PCA, topo_col)
            n_row += 1

            width_ratios = [5, 1] * topo_col
            height_ratios = [5] * n_row
            len_w = len(width_ratios)
            len_h = len(height_ratios)

            fig_topo = plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))
            gs = gridspec.GridSpec(len(height_ratios), len(width_ratios),
                                   height_ratios=height_ratios,
                                   width_ratios=width_ratios)

            ax_topo = np.empty((len_h, len_w), dtype='object')
            for i in range(len_h):
                for j in range(len_w):
                    ax_topo[i, j] = fig_topo.add_subplot(gs[i, j])

            for id_comp in range(n_PCA):
                id_row, id_col = divmod(id_comp, topo_col)

                im_tmp = topo(select_A[:, id_comp], info,
                            axes=ax_topo[id_row][id_col*2], show=False)
                fig_topo.colorbar(im_tmp[0], cax=ax_topo[id_row][id_col*2 + 1],
                            orientation='vertical')
                ax_topo[id_row][id_col*2].set_title(
                    'IC_{0}'.format(str(id_comp)))

            fig_topo.tight_layout()
            fig_topo.savefig(f'Topo_PCA_{n_PCA}{sobi_suffix}.jpg')


        if heatmap_plot:
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

            tsne_trans_flag = False if feat_type == 'BP_3' else True
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
            colors = list(red.range_to(Color("black"), nr_comp))
            for i, c in zip(range(nr_comp), colors):
                data = list(map(tuple, X_2d[np.where(ref_label[:, 0] == i)[0]]))
                ax_tsne[3].scatter(*zip(*data), c=c.hex, label=i)
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

                label_mat = all_labels.reshape(nr_comp, len(subj_list), len(stim_list))
                label_mat = label_mat.reshape(nr_comp, len(subj_list)*len(stim_list), order=order)
                tmp = sns.heatmap(data=label_mat, ax=ax_tsne[4+i],
                                xticklabels=xlabel_name)
                ax_tsne[4+i].xaxis.tick_top()  # x axis on top
                ax_tsne[4+i].xaxis.set_label_position('top')
                tmp.set_yticklabels(tmp.get_yticklabels(), fontsize=8, rotation=45)
                tmp.set_xticklabels(tmp.get_xticklabels(), fontsize=8, rotation=45)
                ax_tsne[4+i].set_title(suffix[1:])

                # fig_hm, ax_hm = plt.subplots(1, 1, figsize=(32, 120))
            # fig.savefig('hm_{0}_{1}{2}.jpg'.format(feat_type, est_flag, suffix))
        if psd_plot or heatmap_plot:
            fig.tight_layout()
            fig.savefig('{0}_cluster_{1}_{2}{3}{4}.jpg'.format(est_flag, str(n_clusters), feat_type, ref_suffix, sobi_suffix))

        if behav_plot:
            if pca_flag:
                nr_comp = 76
            else:
                nr_comp = 120
            width_ratios = [2, 0.5, 10, 10, 10]
            height_ratios = [2] * nr_comp
            len_w = len(width_ratios)
            len_h = len(height_ratios)

            fig_behav = plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))
            gs = gridspec.GridSpec(len(height_ratios), len(width_ratios),
                                   height_ratios=height_ratios,
                                   width_ratios=width_ratios)

            ax_behav = np.empty((len_h, len_w), dtype='object')
            for i in range(len_h):
                for j in range(len_w):
                    ax_behav[i, j] = fig_behav.add_subplot(gs[i, j])

            behav_df, behav_val, all_behav_df = behav()
            all_behav_df.columns = all_behav_df.columns.droplevel()
            from jxu.data.eeg_process import NIBSEEG
            ses_eeg = NIBSEEG()
            tmp_dict = ses_eeg.subject_list
            [tmp_dict.pop(key) for key in ['test', 'KNL', 'BXB', 'ZWS']]
            stim_mat = np.vstack(list(tmp_dict.values()))
            # map the session based order to stim based order, i.e., to 0,4,10,40Hz
            ind_stim2ses = np.argsort(stim_mat, axis=1)
            all_behav_df.replace(value={0: 'pre', 1: 'stim_1',
                                        2: 'stim_2', 3: 'post'},
                                 to_replace=True)

            for IC in range(nr_comp):
                tmp_df = all_behav_df.copy()
                IC_subset_labels = all_labels[IC*32: IC*32 + 32].reshape((8, 4))
                reorder_IC_subset_labels = np.ones(IC_subset_labels.shape)
                for i in range(8):
                    reorder_IC_subset_labels[i] = IC_subset_labels[i, ind_stim2ses[i, :]]
                # repeat each labels 200 times as there are 200 questions in
                # each session
                ext_IC_subset_labels = np.tile(
                    reorder_IC_subset_labels.ravel(), (200, 1)).ravel(order='F')

                tmp_df['cluster'] = ext_IC_subset_labels.astype('int')

                if heatmap_plot:
                    im_tmp = topo(select_A[:, IC], info, axes=ax_behav[IC, 0],
                                show=False)
                    fig_behav.colorbar(im_tmp[0], cax=ax_behav[IC, 1],
                                    orientation='vertical')

                    ax_behav[IC, 0].set_title('IC_{0}'.format(str(IC)))
                sns.boxplot(x='cluster', y='onset', hue='block',
                            data=tmp_df, ax=ax_behav[IC, 2],
                            showfliers=False)
                sns.boxplot(x='cluster', y='fluency', hue='block',
                            data=tmp_df, ax=ax_behav[IC, 3],
                            showfliers=False)
                sns.boxplot(x='cluster', y='nr_blank', hue='block',
                            data=tmp_df, ax=ax_behav[IC, 4],
                            showfliers=False)
                print(IC)

            handles, labels = ax_behav[0, 2].get_legend_handles_labels()
            fig_behav.legend(handles, labels, loc='upper center', ncol=4)
            for IC in range(nr_comp):
                for j in [2, 3, 4]:
                    ax_behav[IC, j].get_legend().remove()

            fig_behav.tight_layout()
            fig_behav.savefig('PCA_{0}_rs_behav_{1}_{2}{3}{4}.jpg'.format(est_flag, str(n_clusters), feat_type, ref_suffix, sobi_suffix))

import pdb;pdb.set_trace()




























