
from mne.io import read_raw_brainvision as rrb
from mne.viz import plot_raw
from mne.viz import plot_raw_psd

channel_list = ['FCC4h', 'C1', 'C2', 'C3', 'C3', 'C4', 'Cz', 'CCP1h',
                'CCP2h', 'CCP3h', 'CCP4h', 'CCP5h', 'CCP6h', 'CP1',
                'CP2', 'CP3', 'CP4', 'CP5', 'CPz', 'CPP1h', 'CPP2h',
                'CPP3h', 'CPP4h', 'CPP5h', 'P1', 'P2', 'P3', 'P4',
                'P5', 'Pz', 'PPO1h', 'PPO2h', 'PO3', 'POz']
subject_list = {'ZWS': [10, 0, 40, 4],
                'NUK': [40, 4, 10, 0],
                'OSA': [0, 40, 10, 4],
                'KNL': [4, 10, 40, 0],
                'ZYC': [40, 10, 4, 0],
                'CCH': [0, 4, 10, 40],
                'DSW': [10, 0, 40, 4],
                'VQT': [4, 0, 40, 10],
                'BXB': [0, 10, 40, 4],
                'BMC': [0, 40, 4, 10]}
subject_name_list = [*subject_list.keys()]
# = vhdr_load('xxx.vhdr').pick_channels(channel_list)
"""
Used Annotations descriptions: ['Stimulus/S 10', 'Stimulus/S 11', 'Stimulus/S 12', 'Stimulus/S 14', 'Stimulus/S 16', 'Stimulus/S 18', 'Stimulus/S 19', 'Stimulus/S 17', 'Stimulus/S 62', 'Stimulus/S 63', 'Stimulus/S 13', 'Stimulus/S 15', 'Stimulus/S 60', 'Stimulus/S 61', 'Stimulus/S  1', 'Stimulus/S  4', 'Stimulus/S 28', 'Stimulus/S  6', 'Stimulus/S 30', 'Stimulus/S 31', 'Stimulus/S 34', 'Stimulus/S 35', 'Stimulus/S 32', 'Stimulus/S 33', 'Stimulus/S  7', 'Stimulus/S 40', 'Stimulus/S 41', 'Stimulus/S 42', 'Stimulus/S 44', 'Stimulus/S 50', 'Stimulus/S 51', 'Stimulus/S 46', 'Stimulus/S 48', 'Stimulus/S 49', 'Stimulus/S 43', 'Stimulus/S 45', 'Stimulus/S 47', 'Stimulus/S 29', 'Stimulus/S  5', 'Stimulus/S 20', 'Stimulus/S 24', 'Stimulus/S 25', 'Stimulus/S254', 'Stimulus/S 26', 'Stimulus/S  2', 'Stimulus/S  3', 'Stimulus/S255']
"""
def str_generator(subject, session):
    if isinstance(subject, str):
        subj_str = subject
    elif isinstance(subject, (int, float, complex)):
        subj_str = subject_name_list[subject]
    else:
        raise ValueError('Invalid subject! Either string or number!')

    if isinstance(session, (int, float, complex)):
        ses_str = str(session)  # .zfill(2)
    else:
        raise ValueError('Invalid session! Must be number!')

    return subj_str, ses_str

def vhdr_load(filename):

    data = rrb(filename, eog=('HEOGL', 'HEOGR', 'VEOGb'), misc='auto',
               scale=1.0, preload=True, verbose=None)

    return data


def single_to_multi(filename, channel_list):

    data = {}

    for chn in channel_list:
        data[chn] = vhdr_load(filename).pick_channels([chn])


def eeg_loader(root_path='/Users/xujiachen/File/Data/NIBS/Stage_one/',
               source='internal', subject=0, session=1, chn=None):

    if source == 'internal':

        subj_str, ses_str = str_generator(subject, session)

        file_name_stem = '{subject}_SESSION_{session}'.format(
            subject=subj_str, session=ses_str)
        folder_path = root_path + '{subject}'.format(
            subject=subj_str, session=ses_str) + '/' + file_name_stem + '/'

        if chn is None:
            data = vhdr_load(folder_path + file_name_stem + '.vhdr')
        elif isinstance(chn, list):
            data = vhdr_load(
                folder_path + file_name_stem + '.vhdr').pick_channels(chn)
        else:
            raise ValueError('Invalid Channel list! Must be list!')

        return data

    elif source == 'moabb':
        pass
    else:
        raise ValueError('Non-default source! Please use vhdr_loader()!')


def audio_loader(root_path='/Users/xujiachen/File/Data/NIBS/Stage_one/',
                 source='internal', subject=0, session=1, trial=0, std=True):

    subj_str, ses_str = str_generator(subject, session)

    file_name_stem = '{subject}_SESSION_{session}'.format(
        subject=subj_str, session=ses_str)
    folder_path = root_path + '{subject}'.format(
        subject=subj_str, session=ses_str) + '/' + file_name_stem + '/Audio_Recording/Exp_data/'

    block, nr_trial = divmod(trial, 20)
    nr_run, nr_block = divmod(block, 2)

    if std:
        audio_name = folder_path + 'rec_QA_run_{nr_run}_block_{nr_block}_trial_{nr_trial}_std.wav'.format(
            nr_run=str(nr_run).zfill(2), nr_block=str(nr_block).zfill(3), nr_trial=str(nr_trial).zfill(3))
    else:
        audio_name = folder_path + 'rec_QA_run_{nr_run}_block_{nr_block}_trial_{nr_trial}.wav'.format(
            nr_run=str(nr_run).zfill(2), nr_block=str(nr_block).zfill(3), nr_trial=str(nr_trial).zfill(3))
    return audio_name


