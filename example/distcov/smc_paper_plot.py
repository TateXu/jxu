#========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2021-03-30 10:06:40
# Name       : smc_paper_plot.py
# Version    : V1.0
# Description: Figures for SMC2021.
#========================================


import os
import numpy as np
import pickle
import pandas as pd
import matplotlib.font_manager
from matplotlib import rcParams
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

from mne.decoding import CSP
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from numpy.linalg import cond
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

from moabb.datasets import MunichMI, BNCI2014001, Cho2017
from jxu.model.Distance_covariance.dist_cov import Distcov
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font',**{'family':'serif','serif':['Times']})
# rc('font', **{'family':'serif','serif':'Computer Modern'})

def load_data(subject, session, ds='Munich'):

    from moabb.paradigms import MotorImagery
    MI = MotorImagery(fmin=8, fmax=32, events=['left_hand', 'right_hand'])
    X, y, metadata = MI.get_data(ds, [subject])
    return X, y


def load_covmat(subject, ds_name='Munich', method='d_cov'):
    with open('/home/jxu/File/Data/Distcov/precompute_data/Distcov_{0}_S{1}_{2}.pkl'.format(
            ds_name, str(subject), method), 'rb') as f:
        distcov_mat = pickle.load(f)

    return distcov_mat

method = 'd_cov_sqr'
dataset = {'Munich': MunichMI(),
           'BNCI2014001': BNCI2014001(),
           'Cho': Cho2017()}

labelfontsize = 20
legendfontsize = 16
for id_ds, (ds_name, ds) in enumerate(dataset.items()):
    if ds_name != 'Munich':
        continue
    subject = 1
    X_raw, y = load_data(subject=subject, session=0, ds=ds)
    scm = Covariances().fit_transform(X_raw, y)
    X = load_covmat(subject=subject, ds_name=ds_name, method=method)

    # for id_trial in [0]:  # range(X.shape[0]):
    fig_hm, axes_hm = plt.subplots(1, 6, figsize=(18, 5), gridspec_kw={'width_ratios': [5, 0.5, 5, 0.5, 5, 0.5]})

    sns.heatmap(data=X[0], ax=axes_hm[0], cbar_ax=axes_hm[1], linewidths=0.0)
    sns.heatmap(data=scm[0], ax=axes_hm[2], cbar_ax=axes_hm[3], linewidths=0.0)

    ratio = np.log10(np.abs(X[0]/scm[0]))
    sns.heatmap(data=ratio, ax=axes_hm[4], cbar_ax=axes_hm[5], linewidths=0.0)


    axes_hm[0].set_title('Distance Covariance Matrix', fontsize=labelfontsize)
    axes_hm[2].set_title('Sample Covariance Matrix', fontsize=labelfontsize)
    axes_hm[4].set_title('Ratio of dCov to SCM in Log10-scale', fontsize=labelfontsize)

    [axes_hm[i].set_xticklabels('') for i in [0, 2, 4]]
    [axes_hm[i].set_yticklabels('') for i in [0, 2, 4]]

#     [axes_hm[i].set_xticks('') for i in [0, 2, 4]]
#     [axes_hm[i].set_yticks('') for i in [0, 2, 4]]
    fig_hm.tight_layout()
    fig_hm.savefig('./smc/covmat.pdf')
import pdb;pdb.set_trace()

all_df = pd.read_pickle('all_acc.pkl')
# all_df = pd.read_pickle('all_acc_distcov.pkl')
chn = [128, 22, 64]
all_df['all_methods'] = all_df.apply(lambda row: row.Metric + f' ({row.Method})', axis=1)

tmp_df = all_df.loc[all_df['Method'] == method].copy()
select_df = tmp_df.replace({'Raw + AIRM': 'SCM + AIRM',
                            'Raw + LogE': 'SCM + LogE',
                            'Raw + MDM': 'SCM + MDM',
                            'Distcov + AIRM': 'dCov + AIRM',
                            'Distcov + LogE': 'dCov + LogE',
                            'Distcov + MDM': 'dCov + MDM',
                            'linear': 'Linear',
                            'poly': 'Polynomial',
                            'rbf': 'Gaussian'})
mdm_df = select_df[tmp_df['Metric'].str.contains("MDM")]
ts_df = select_df[tmp_df['Metric'].str.contains("MDM").replace({True:False, False:True})]

plt.rcParams["font.family"] = "serif"
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times']



fig_proj = plt.figure(figsize=(10, 5))
axes_proj = fig_proj.add_axes([0.1, 0.15, 0.8, 0.7]) # left, bottom, width, height (range 0 to 1)

proj_df = ts_df.loc[ts_df.Classifier == 'Linear'].copy()

# axes_ts.set_title(f'Dataset {ds_name} with #channels={str(chn[id_ds])}')
sns.boxplot(x='Dataset', y='Accuracy', hue='Metric', data=proj_df,
            order=['BNCI2014001', 'Cho', 'Munich'],
            palette=[sns.color_palette()[i] for i in [-1, 1, 2, 4]], ax=axes_proj)
axes_proj.set_ylim([0.3, 1.05])

xlabels_ticks = axes_proj.get_xticklabels()
axes_proj.set_xticklabels(xlabels_ticks, fontsize=labelfontsize)
axes_proj.set_xlabel('')

axes_proj.set_xlabel('Dataset', fontsize=labelfontsize, fontweight='bold')

# ylabels_ticks = axes_ts.get_yticklabels()
axes_proj.set_yticklabels([str(i) for i in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]], fontsize=legendfontsize)

ylabels = axes_proj.get_ylabel()
axes_proj.set_ylabel(ylabels, fontsize=labelfontsize, fontweight='bold')

handles, labels = axes_proj.get_legend_handles_labels()
axes_proj.get_legend().remove()
fig_proj.legend(handles, labels, loc='upper center', ncol=2, fontsize=legendfontsize, frameon=False)

fig_proj.tight_layout()
fig_proj.savefig(f'./smc/proj.pdf')
del fig_proj, axes_proj
import pdb;pdb.set_trace()


















for id_ds, (ds_name, _) in enumerate(dataset.items()):

    fig_ts = plt.figure(figsize=(10, 5))
    axes_ts = fig_ts.add_axes([0.1, 0.15, 0.8, 0.7]) # left, bottom, width, height (range 0 to 1)

    single_df = ts_df.loc[ts_df.Dataset == ds_name].copy()

    # axes_ts.set_title(f'Dataset {ds_name} with #channels={str(chn[id_ds])}')
    sns.boxplot(x='Classifier', y='Accuracy', hue='Metric', data=single_df,
                palette=[sns.color_palette()[i] for i in [-1, 1, 2, 4]], ax=axes_ts)
    axes_ts.set_ylim([0.3, 1.05])

    xlabels_ticks = axes_ts.get_xticklabels()
    axes_ts.set_xticklabels(xlabels_ticks, fontsize=labelfontsize)
    axes_ts.set_xlabel('')

    axes_ts.set_xlabel('Kernel SVM Classifier', fontsize=labelfontsize, fontweight='bold')

    # ylabels_ticks = axes_ts.get_yticklabels()
    axes_ts.set_yticklabels([str(i) for i in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]], fontsize=legendfontsize)

    ylabels = axes_ts.get_ylabel()
    axes_ts.set_ylabel(ylabels, fontsize=labelfontsize, fontweight='bold')

    handles, labels = axes_ts.get_legend_handles_labels()
    axes_ts.get_legend().remove()
    fig_ts.legend(handles, labels, loc='upper center', ncol=2, fontsize=legendfontsize, frameon=False)

    fig_ts.tight_layout()
    fig_ts.savefig(f'./smc/{ds_name}_ts.pdf')
    del fig_ts, axes_ts
    print(id_ds)

fig_mdm = plt.figure(figsize=(10, 5))
axes_mdm = fig_mdm.add_axes([0.1, 0.15, 0.8, 0.75]) # left, bottom, width, height (range 0 to 1)

# axes_mdm.set_title(f'Dataset {ds_name} with #channels={str(chn[id_ds])}')

# cmap = sns.color_palette("Set2")
cmap = [sns.color_palette()[i] for i in [0, 3]]
sns.boxplot(x='Dataset', y='Accuracy', hue='Metric', data=mdm_df, ax=axes_mdm,
            palette=cmap[-2:], order=['BNCI2014001', 'Cho', 'Munich'])

xlabels_ticks = axes_mdm.get_xticklabels()
axes_mdm.set_xticklabels(xlabels_ticks, fontsize=labelfontsize)

xlabels = axes_mdm.get_xlabel()
axes_mdm.set_xlabel(xlabels, fontsize=labelfontsize, fontweight='bold')
ylabels = axes_mdm.get_ylabel()
axes_mdm.set_ylabel(ylabels, fontsize=labelfontsize, fontweight='bold')
handles, labels = axes_mdm.get_legend_handles_labels()

fig_mdm.legend(handles, labels, loc='upper center', ncol=2, frameon=False, fontsize=legendfontsize)

axes_mdm.get_legend().remove()
fig_mdm.tight_layout()
fig_mdm.savefig(f'./smc/mdm.pdf')



import pdb;pdb.set_trace()

