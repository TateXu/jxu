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

    def __init__(self, subject=1, session=0, band=[0.1, 70.0], notch=[50.0],
                 root='/home/jxu/File/Data/NIBS/Stage_one/EEG/Exp/',
                 reref='CA', filter_setting=[()]):
        self.subject = subject
        self.session = session
        self.band = band
        self.notch = notch
        self.root = root
        self.reref = reref
        self.filter_setting = filter_setting
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

    def load_raw(self, nr_seg=2):
        self.raw_path(nr_seg=nr_seg)
        self.raw_data = [vhdr_load(self.root + eeg_file_path) for eeg_file_path in self.eeg_file_path_list]

        if len(self.raw_data) != nr_seg:
            raise ValueError("#Seg != #Raw data")

        return self

    def raw_path(self, nr_seg=2):

        self.subj_id = list(self.subject_list.keys())[self.subject]
        self.eeg_folder = '{0}/Session_{1}/'.format(self.subj_id, str(int(self.session)))
        self.audio_folder = '{0}/Audio/Session_{1}/Exp_data/'.format(self.subj_id, str(int(self.session)).zfill(2))
        self.eeg_file_path_list = ['{0}{1}_seg_{2}.vhdr'.format(self.eeg_folder, self.subj_id, str(i)) for i in range(nr_seg)]

        return self

    def get_data(self):
        pass


    def _get_single_data(self):
        pass


    def save(self):
        pass

    # -------- META info setting ------

    def set_montage(self):
        pass


    def get_montage(self):
        pass


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


