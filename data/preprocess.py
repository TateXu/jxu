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
from utils import BaseEEG


class NIBSEEG():

    def __init__(self, subject=1, session=1, band=[0.1, 70.0], notch=[50.0],
                 root='/home/jxu/File/Data/NIBS/Stage_one/EEG/ZWS/ZWS_SESSION_1/',
                 reref='CA', filter_setting=[()]):
        pass

    # -------- File opeartion ---------

    def load_raw(self):
        pass


    def raw_path(self):
        pass


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


