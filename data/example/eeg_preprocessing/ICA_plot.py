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
from mne.preprocessing import ICA
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
task_list = ['RS_close', 'RS_open',
            'QA_trial', 'QA_audio', 'QA_ans', 'QA_rec', 'QA_cen_word',
            'Arti_trial', 'Arti_action', 'Arti_rec']
stg_list = ['pre', 'post', 'stim_1', 'stim_2', 'all']

task = 'RS_close'
stg = 'all'
chn_picks = None #  ['Fz', 'Cz', 'Pz']
bands_id = 0
data = 'RS'

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

for data in ['RS', 'all']:
    # for id_sj, subj in enumerate([1, 2, 3, 5, 6, 7, 8, 10]):
        # for ses in range(4):

    rerun_list = [(2, 1), (2, 2), (6, 1), (7, 2)]
    for i in range(1):
        for subj, ses in rerun_list:
            # if subj == 1 and ses == 0:
            #    continue
            # try:
            ses_eeg = NIBSEEG(subject=subj, session=ses)
            ses_eeg.raw_load()
            ses_eeg.set_montage()
            ses_eeg.data_concat(cp_flag=True)

            ses_eeg.set_channels()
            ses_eeg.rereference('average')
            ses_eeg.raw_filter(bands=[(0.1, 70.0)], notch=True)
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

            ica = ICA(method='infomax')
            if data == 'all':
                eeg = ses_eeg.raw_data_clean
            elif data == 'RS':
                eeg = stg_epoch[0]
            ica.fit(eeg)
            source = ica.get_sources(eeg)
            # img_s = source.plot(show=False)
            # img_prop = ica.plot_properties(raw, picks=18)
            img_comp = ica.plot_components(show=False, picks='all')
            img_comp.suptitle("S{0}_Ses{1}_{2}".format(str(subj), str(ses), data))

            img_comp.savefig("./ICA_related/image/S{0}_Ses{1}_{2}.jpg".format(
                str(subj), str(ses), data))

            with open('./ICA_related/fitted_obj/S{0}_Ses{1}_{2}.pkl'.format(
                    str(subj), str(ses), data), 'wb') as f:
                pickle.dump(ica, f)
                # with open("./ICA_related/bug_report_S{0}_Ses{1}_{2}.txt".format(
                        # str(subj), str(ses), data), "w") as text_file:
                    # print("Stopped", file=text_file)

    import pdb;pdb.set_trace()


