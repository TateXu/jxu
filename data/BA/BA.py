
import mne
from mne.io import read_raw_brainvision as rrb
import pickle


def raw_load(filename):

    raw_data = rrb(filename, eog=('EOG151', 'EOG152'), misc='auto',
        scale=1.0, preload=True, verbose=None)
    raw_data.set_channel_types({'Audio': 'stim', 'tACS': 'stim'})

    return raw_data


def set_montage(raw_data, bad_chn_list=None):

    montage = mne.channels.read_custom_montage(
        'BC-TMS-128.bvef', unit='auto')
    raw_data.set_montage(montage)

    if bad_chn_list is not None:
        bad_chn_dict = dict.fromkeys(bad_chn_list, 'stim')
        raw_data.set_channel_types(bad_chn_dict)

    return raw_data

def plot_montage(raw_data, axes='mlab3D', name=False, surfaces='head'):
  
    from mayavi import mlab
    import os.path as op

    from mne.datasets import fetch_fsaverage
    from mne.viz import plot_alignment

    subjects_dir = op.dirname(fetch_fsaverage())
    fig = plot_alignment(raw_data.info, trans=None, subject='fsaverage',
                         subjects_dir=subjects_dir, eeg=['projected'],
                         surfaces=surfaces)



filename = '/home/jxu/File/Data/NIBS/Stage_one/EEG/Exp/NUK/Session_0/NUK_seg_0.vhdr' 

#  data = raw_load(filename)

with open('example.pkl', 'rb') as f_in:
    cropped_data = pickle.load(f_in)

data = set_montage(cropped_data)
plot_montage(cropped_data)

import pdb;pdb.set_trace()