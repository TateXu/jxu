#========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2021-03-30 10:06:40
# Name       : riemann_dist.py
# Version    : V1.0
# Description: .
#========================================


import os
import numpy as np
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

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

from jxu.model.Distance_covariance.dist_cov import Distcov


def score(clf, X, y, scoring):
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    le = LabelEncoder()
    y = le.fit_transform(y)
    acc = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
    return acc.mean()


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


if __name__ == "__main__":
    ###############################################################################
    # Parameter Initialization
    # ------------------------
    process = False
    plt_flag = True
    ds_name = 'BNCI2014001'

    from moabb.datasets import MunichMI, BNCI2014001, Cho2017
    dataset = {'Munich': MunichMI(),
               'BNCI2014001': BNCI2014001(),
               'Cho': Cho2017()}
    ds = dataset[ds_name]

    kernel = 'poly'

    if not plt_flag:
        if process:
            ds = dataset[ds_name]
            for ds_name, ds in dataset.items():
                for subject in ds.subject_list:
                    for method in ['d_cov_sqr', 'd_cor_sqr']:
                        X, y = load_data(subject=subject, session=0, ds=ds)
                        est = Distcov(save=True, subject=subject, ds_name=ds_name,
                                      method=method, multiprocess=True)
                        est.fit_transform(X, y, force=False)
                        print(subject)
        else:
            ds = dataset[ds_name]
            acc_dict = []
            all_cond = np.empty((3, 2), dtype='object')
            for id_ds, (ds_name, ds) in enumerate(dataset.items()):
                for subject in ds.subject_list:
                    all_num_list = []
                    for method in ['d_cov_sqr', 'd_cor_sqr']:
                        parameters = {'C': np.logspace(-2, 2, 10)}
                        opt_SVM = GridSearchCV(
                            SVC(kernel=kernel), parameters)

                        pipelines = {}

                        pipelines['Distcov + AIRM'] = make_pipeline(
                            TangentSpace(metric='riemann'),
                            opt_SVM)

                        pipelines['Raw + AIRM'] = make_pipeline(
                            Covariances('scm'),
                            TangentSpace(metric='riemann'),
                            opt_SVM)

                        pipelines['Distcov + LogE'] = make_pipeline(
                            TangentSpace(metric='logeuclid'),
                            opt_SVM)

                        pipelines['Raw + LogE'] = make_pipeline(
                            Covariances('scm'),
                            TangentSpace(metric='logeuclid'),
                            opt_SVM)

                        pipelines['Distcov + MDM'] = make_pipeline(
                            MDM(metric='riemann'))

                        pipelines['Raw + MDM'] = make_pipeline(
                            Covariances('scm'),
                            MDM(metric='riemann'))
                        X_raw, y = load_data(subject=subject, session=0, ds=ds)
                        X = load_covmat(subject=subject, ds_name=ds_name,
                                        method=method)
                        number_list = []
                        for id_trial in range(X.shape[0]):
                            # cond_dist_list.append(cond(X[id_trial]))
                            eigvals, eigvecs = np.linalg.eigh(X[id_trial])
                            number_list.append(np.sum(eigvals<=0))

                        for name, clf in pipelines.items():
                            data = X if 'Distcov' in name else X_raw
                            acc = score(clf, data, y, scoring='roc_auc')
                            tmp_dict = {'Dataset': ds_name,
                                        'Subject': subject,
                                        'Method': method,
                                        'Metric': name,
                                        'Accuracy': acc,
                                        'Negative eigvals': number_list,
                                        'Classifier': kernel}
                            acc_dict.append(tmp_dict)
                            print(f'{ds_name}-S{subject}-{name}-{method}: {str(acc)}')
            import pdb;pdb.set_trace()

            all_df = pd.DataFrame(acc_dict)
            all_df.to_pickle(kernel + '_' + 'acc_distcov_methods.pkl')
            # all_cond[id_ds] = [cond_list, cond_dist_list]
            import pdb;pdb.set_trace()
    else:
        all_df = pd.read_pickle('all_acc.pkl')
        import pdb;pdb.set_trace()
        # all_df = pd.read_pickle('all_acc_distcov.pkl')
        fig, axes = plt.subplots(3, 2, gridspec_kw={'width_ratios': [10, 7]}, figsize=(34, 15))
        chn = [128, 22, 64]
        all_df['all_methods'] = all_df.apply(lambda row: row.Metric + f' ({row.Method})', axis=1)
        import pdb;pdb.set_trace()

        method = 'd_cov_sqr'
        select_df = all_df.loc[all_df['Method'] == method].copy()

        for id_ds, (ds_name, _) in enumerate(dataset.items()):
            single_df = select_df.loc[select_df.Dataset == ds_name].copy()
            n_subject = np.unique(single_df.Subject.values).shape[0]

            sns.barplot(x='Subject', y='Accuracy', hue='all_methods', data=single_df, ax=axes[id_ds, 0])
            axes[id_ds, 0].set_title(f'Dataset {ds_name} with #channels={str(chn[id_ds])}')
            sns.boxplot(x='Classifier', y='Accuracy', hue='Metric', data=single_df, ax=axes[id_ds, 1])

            xlabels = axes[id_ds, 1].get_xticklabels()
            axes[id_ds, 1].set_xticklabels(xlabels, rotation=45)

            print(id_ds)
        fig.tight_layout()
        fig.savefig(f'{method}_clf.jpg')
        # fig.savefig(f'{kernel}_Results_sqr_mdm.jpg')
            # for m, c in zip(['AIRM', 'LogE'], ['red', 'blue']):
                # t_mean, t_std, t_acc = np.empty((3, 2, n_subject*2))
                # for id_mat, mat in enumerate(['Distcov', 'Raw']):
                    # pipe = mat + ' + ' + m
                    # t_acc[id_mat] = single_df.loc[single_df['Metric'] == pipe].Accuracy.values
                    # t_mean[id_mat] = single_df.loc[single_df['Metric'] == pipe]['Negative eigvals'].apply(np.mean).values
                    # t_std[id_mat] = single_df.loc[single_df['Metric'] == pipe]['Negative eigvals'].apply(np.std).values
                # diff_acc = t_acc[0] - t_acc[1]
                # diff_mean = (t_mean[0] + t_mean[1]) / 2
                # diff_std = (t_std[0] + t_std[1]) / 2

                # sns.scatterplot(x=diff_mean/2, y=diff_acc, size=diff_std/2,
                                # ax=axes[id_ds, 2], color=c)
#             from matplotlib.lines import Line2D
            # axes[id_ds, 2].set_title('#Avg. Neg. Eigvals (x) vs. Acc. Difference (y), Std of #Neg (size)')
            # custom_lines = [Line2D([0], [0], marker='o', color='w', label='AIRM', markerfacecolor='r', markersize=10),
                            # Line2D([0], [0], marker='o', color='w', label='LogEuclidean', markerfacecolor='b', markersize=10)]
            # axes[id_ds, 2].legend(handles=custom_lines, loc='upper right')









