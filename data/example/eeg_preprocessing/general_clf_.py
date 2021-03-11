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

from mne.decoding import CSP
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
# fig_list = []
# ax_list = []
# for i in range(6):
    # fig, ax = plt.subplots(8, 4, figsize=(24, 16), num=i)
    # fig_list.append(fig)
    # ax_list.append(ax)



def load_data(subject, session, bands=[(1.0, 70.0)]):

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

    ses_eeg = NIBSEEG(subject=subject, session=session)
    ses_eeg.raw_load()
    ses_eeg.set_montage()
    ses_eeg.data_concat(cp_flag=True)

    ses_eeg.set_channels()
    ses_eeg.rereference('average')
    ses_eeg.raw_filter(bands=bands, notch=True)
    events, event_id = mne.events_from_annotations(ses_eeg.raw_data_clean)

    all_epoch = mne.Epochs(ses_eeg.data[bands_id], events,
                           event_id=[evt_id], tmin=tmin, tmax=tmax,
                           picks=chn_picks,
                           baseline=None, preload=True)
    all_epoch.drop_channels(all_epoch.info['bads'])
    trial_index = np.hsplit(np.arange(len(all_epoch)), 4)
    if stg_id == 4:
        stg_epoch = [all_epoch[trial_index[epoch_id]] for epoch_id in range(4)]
    else:
        stg_epoch = all_epoch[trial_index[stg_id]]
    if 'RS' in task:
        X_pre = np.asarray(np.hsplit(stg_epoch[0].get_data()[0, :, :-1], 36))
        X_post = np.asarray(np.hsplit(stg_epoch[3].get_data()[0, :, :-1], 36))
        y_pre = np.ones((X_pre.shape[0], 1))
        y_post = np.zeros((X_pre.shape[0], 1))
        X = np.concatenate((X_pre, X_post), axis=0)
        y = np.concatenate((y_pre, y_post), axis=0)

    return ses_eeg, stg_epoch, X*1e6, y

def score(clf, X, y, scoring):
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    le = LabelEncoder()
    y = le.fit_transform(y)
    acc = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
    return acc.mean()

if __name__ == "__main__":
    for id_sj, subj in enumerate([2, 6, 7, 8, 10]):
        for ses in range(4):

            ###########################################################################
            # Parameter Initialization
            # ------------------------

            # L2 regularized SVM classifier using grid search to find reg. coefficients
            parameters = {'C': np.logspace(-2, 2, 10)}
            opt_SVM = GridSearchCV(SVC(kernel='linear'), parameters)

            nr_comp = 40  # Number of applied (visualized) filter components
            nr_decomp = 0  # Use GED or ED to generate filter

            ###########################################################################
            # Initializing algorithms pipeline for analysis
            # ---------------------------------------------
            pipelines = {}
            # Standard CSP based classifier
            pipelines['CSP (' + str(nr_comp) + ') + SVM'] = make_pipeline(
                CSP(n_components=nr_comp, reg='oas'),
                opt_SVM)

            # Standard Riemannian methods
            pipelines['TS + w/o SF + SVM + AIRM'] = make_pipeline(
                Covariances('oas'),
                TangentSpace(metric='riemann'),
                opt_SVM)

            eeg, epoch, X, y = load_data(subj, ses)
            import pdb;pdb.set_trace()

            from mne.preprocessing import ICA
            ica = ICA(method='infomax')
            ica.fit(eeg)
            source = ica.get_sources(eeg)
            import pdb;pdb.set_trace()
            # img_s = source.plot(show=False)
            # img_prop = ica.plot_properties(raw, picks=18)
            img_comp = ica.plot_components(show=False, picks='all')


            eeg.raw_data_clean.drop_channels(eeg.raw_data_clean.info['bads'])
            info = eeg.raw_data_clean.info
            for name, clf in pipelines.items():
                acc = score(clf, X, y, scoring='roc_auc')
                import pdb;pdb.set_trace()
                print('------------------------------------------------------')
                print('S{0}-Ses{1}-{2}: {3}'.format(str(subj), str(ses),
                                                    name, str(acc)))
                print('------------------------------------------------------')
            import pdb;pdb.set_trace()
    import pdb;pdb.set_trace()
    for i in range(6):
        rs, st = divmod(i, 3)

        fig_list[i].tight_layout()
        fig_list[i].savefig('RS_{0}_{1}.jpg'.format(
            rs_list[rs], stg_list[st]))
    plt.show()
    import pdb;pdb.set_trace()

