#========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2020-06-25 13:34:43
# Name       : preprocess.py
# Version    : V1.0
# Description: Class for preprocessing EEG& Audio
#========================================

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
from .utils import BaseEEG


class NIBSEEG():

    def __init__(self, subject=1, session=0, nr_seg=2, bands=[(0.1, 70.0)],
                 root='/home/jxu/File/Data/NIBS/Stage_one/EEG/Exp/',
                 filter_para=dict(method='iir', iir_params=dict(order=5, ftype='butter')),
                 reref='CA', bad_chn_list=None):

        self.subject = subject
        self.session = session
        self.nr_seg = nr_seg
        self.bands = bands
        self.root = root
        self.reref = reref
        self.filter_para = filter_para
        self.bad_chn_list = bad_chn_list
        self.subject_list = {'test': [],
                             'TES': [10, 0, 40, 4],
                             'NUK': [40, 4, 10, 0],
                             'OSA': [0, 40, 10, 4],
                             'KNL': [4, 10, 40, 0],
                             'ZYC': [40, 10, 4, 0],
                             'CCH': [0, 4, 10, 40],
                             'DSW': [10, 0, 40, 4],
                             'VQT': [4, 0, 40, 10],
                             'BXB': [0, 10, 40, 4],
                             'BMC': [0, 40, 4, 10],
                             'ZWS': [10, 0, 40, 4]}

    # -------- File opeartion ---------

    def raw_load(self):
        self.raw_path()
        self.raw_data = [vhdr_load(self.root + eeg_file_path) for eeg_file_path in self.eeg_file_path_list]

        [single_raw.set_channel_types({'Audio': 'stim', 'tACS': 'stim'}) for single_raw in self.raw_data]
        if len(self.raw_data) != self.nr_seg:
            raise ValueError("#Seg != #Raw data")

        return self

    def raw_path(self):

        self.subj_id = list(self.subject_list.keys())[self.subject]
        self.eeg_folder = '{0}/Session_{1}/'.format(self.subj_id, str(int(self.session)))
        self.audio_folder = '{0}/Audio/Session_{1}/Exp_data/'.format(
            self.subj_id, str(int(self.session)).zfill(2))
        self.eeg_file_path_list = [
            '{0}{1}_seg_{2}.vhdr'.format(
                self.eeg_folder, self.subj_id, str(i)) for i in range(self.nr_seg)]

        return self


    def raw_filter(self):
        print('------- Start filtering-------')
        if not hasattr(self, 'raw_data') or self.raw_data is None:
            self.raw_load()

        self.filtered_data = []
        for fmin, fmax in self.bands:
            for raw in self.raw_data:
                raw_f = raw.copy().filter(l_freq=fmin, h_freq=fmax,
                                          verbose=False, **self.filter_para)
                # method='fir', fir_design='firwin'
                raw_f.notch_filter(freqs=np.arange(50, 99, 50), picks='eeg')
                self.filtered_data.append(raw_f)

        return self

    def get_data(self, filter_flag=True):
        if filter_flag:
            if not hasattr(self, 'filtered_data') or self.filtered_data is None:
                self.raw_filter()
            return self.filtered_data
        else:
            if not hasattr(self, 'raw_data') or self.raw_data is None:
                self.raw_load()
            return self.raw_data

    def _get_single_data(self):
        pass


    def concat_crop(self):
        self.concat_data = True

    def save(self):
        max_file_size = 1.8
        save_filename = path + '/raw_seg_'

        data  = self.concat_data.copy()
        raw_size = 8 * len(data) * data.info['nchan'] / (1024**3)
        nr_save_raw = int(np.ceil(raw_size / max_file_size))

        hard_limit = int(max_file_size *  (1024**3) / data.info['nchan'] / 8)

        for save_ind in range(nr_save_raw):
            start_ind = hard_limit * save_ind
            end_ind = start_ind + hard_limit if save_ind != nr_save_raw - 1 else len(data)
            data_save = data[:, start_ind: end_ind]
            ts_save = data_save[1]
            raw_seg = data.copy().crop(tmin=ts_save[0], tmax=ts_save[-1], include_tmax=True)

            with open(save_filename + str(save_ind) + '.pkl', 'wb') as f:
                pickle.dump(raw_seg, f)
            print(str(save_ind + 1) + '/' + str(nr_save_raw) + ' saved!')


    # -------- META info setting ------

    def set_montage(self):
        if self.bad_chn_list is not None:
            bad_chn_dict = dict.fromkeys(self.bad_chn_list, 'stim')
        else:
            bad_chn_dict = None
        montage = mne.channels.read_custom_montage(
            '/home/jxu/anaconda3/lib/python3.7/site-packages/jxu/data/BC-TMS-128.bvef', unit='auto')
        for temp_data in self.raw_data:
            temp_data.set_montage(montage)
            if bad_chn_dict is not None:
                temp_data.set_channel_types(bad_chn_dict)

        return self

    def get_montage(self):
        pass

    def plot_montage(self, axes='mlab3D', name=False, surfaces='head'):

        if axes == 'mlab3D':
            from mayavi import mlab
            import os.path as op

            import mne
            from mne.channels.montage import get_builtin_montages
            from mne.datasets import fetch_fsaverage
            from mne.viz import plot_alignment

            subjects_dir = op.dirname(fetch_fsaverage())
            fig = plot_alignment(self.raw_data[0].info, trans=None, subject='fsaverage',
                                 subjects_dir=subjects_dir, eeg=['projected'],
                                 surfaces=surfaces)
        elif axes == '2D' or axes=='topo':
            self.raw_data[0].info.plot(kind='topomap', show_names=name)
        elif axes == 'xyz3D':
            self.raw_data[0].info.plot(kind='3d')

    def set_trigger_list(self):
        pass


    def get_trigger_list(self):
        pass


    def set_bad_channels(self):
        pass


    def get_bad_channels(self):
        pass


    # -------- Preprocessing ----------

    def filter(self):
        pass


    def notch_filter(self):
        pass


    def rereference(self):
        pass


    # ----------- Sanity Check --------

    def trigger_check(self):
        pass


