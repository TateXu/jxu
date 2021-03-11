# ========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2020-06-25 13:34:43
# Name       : preprocess.py
# Version    : V1.0
# Description: Class for preprocessing EEG& Audio
# ========================================

from jxu.basiccmd.mycmd import create_folder
from jxu.data.loader import vhdr_load
from jxu.data.utils import nibs_event_dict
from jxu.viz.utils import *
import os
import numpy as np
from numpy.fft import fft, fftfreq
from scipy import signal

from itertools import product
from mne.viz import plot_epochs_image
import platform
from copy import deepcopy

import mne

import matplotlib
import matplotlib.pyplot as plt

import warnings
import pickle
import argparse
import pandas as pd
import pdb
from .utils import BaseEEG

class NIBS():
    def __init__(self, subject, session,
                 root='/home/jxu/File/Data/NIBS/Stage_one/EEG/Exp/'):

        self.subject = subject
        self.session = session
        self.root = root
        self.subject_list = {'test': [],
                             'TES': [10, 4, 0, 40],
                             'NUK': [40, 4, 10, 0],
                             'OSA': [0, 40, 10, 4],
                             'KNL': [4, 10, 40, 0],
                             'ZYC': [40, 10, 4, 0],
                             'CCH': [0, 4, 10, 40],
                             'DWS': [10, 0, 40, 4],
                             'VQT': [4, 0, 40, 10],
                             'BXB': [0, 10, 40, 4],
                             'BMC': [0, 40, 4, 10],
                             'ZWS': [10, 0, 40, 4]}

        self.path_init()


    def path_init(self):
        # Initialize the path for EEG and audio recordings
        self.subj_id = list(self.subject_list.keys())[self.subject]
        self.subj_nr = deepcopy(self.subject)
        self.ses_nr = deepcopy(self.session)
        self.eeg_folder = '{0}/Session_{1}/'.format(self.subj_id, str(int(self.ses_nr)))
        self.audio_folder = self.root + '{0}/Audio/Session_{1}/Exp_data/All/'.format(
            self.subj_id, str(int(self.ses_nr)).zfill(2))

        # meta info folder - folder for saving meta info pickle files
        self.meta_folder = self.root + 'Audio/meta/'
        DIR = self.root + self.eeg_folder
        self.nr_seg = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]) // 3
        self.eeg_file_path_list = [
            '{0}{1}_seg_{2}.vhdr'.format(
                self.eeg_folder, self.subj_id, str(i)) for i in range(self.nr_seg)]

        return self



