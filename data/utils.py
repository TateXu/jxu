import mne
import numpy as np
import warnings
from abc import ABCMeta, abstractmethod


class BaseEEG(metaclass=ABCMeta):

    """Base class for EEG preprocessing

    """
    # ---------------------------------
    @abstractmethod
    def __init__(self):
        pass

    # -------- File opeartion ---------
    @abstractmethod
    def load_raw(self):
        raise NotImplementedError()

    @abstractmethod
    def raw_path(self):
        raise NotImplementedError()

    @abstractmethod
    def get_data(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_single_data(self):
        raise NotImplementedError()


    @abstractmethod
    def save(self):
        raise NotImplementedError()

    # -------- META info setting ------
    @abstractmethod
    def set_montage(self):
        raise NotImplementedError()

    @abstractmethod
    def get_montage(self):
        raise NotImplementedError()

    @abstractmethod
    def set_trigger_list(self):
        raise NotImplementedError()

    @abstractmethod
    def get_trigger_list(self):
        raise NotImplementedError()

    @abstractmethod
    def set_bad_channels(self):
        raise NotImplementedError()

    @abstractmethod
    def get_bad_channels(self):
        raise NotImplementedError()


    # ----------- Preprocessing -------
    @abstractmethod
    def filter(self):
        raise NotImplementedError()

    @abstractmethod
    def notch_filter(self):
        raise NotImplementedError()

    @abstractmethod
    def rereference(self):
        raise NotImplementedError()


    # ----------- Sanity Check --------
    @abstractmethod
    def trigger_check(self):
        raise NotImplementedError()




def nibs_event_dict():

    nr_events_predefined = {'ESC': 6,  # pre + 4 runs + post
                            'Test': 0,
                            'Main': 6,  # pre + 4 runs + post
                            'End': 1,
                            'Pre_run': 6,  # pre + 4 runs + post
                            'Post_run': 1,  # post
                            'Run': 10,  # 4 runs = 10 : 1 + 2 + 3 + 4
                            'Block': 8,
                            'Cali_intro': 2,  # pre + post
                            'Cali_trial': 20,  # 10(pre) + 10(post)
                            'Cali_display': 20,  # 10(pre) + 10(post)
                            'Cali_ans': 20,  # 10(pre) + 10(post)
                            'Cali_rec': 20,  # 10(pre) + 10(post)
                            'Stim': 1,  # either stim or sham
                            'Sham': 1,  # either stim or sham
                            'Fade_in': 1,
                            'Fade_out': 1,
                            'Stable_stim': 10,  # 4 runs = 10 : 1 + 2 + 3 + 4
                            'RS_intro': 8,  # 6-8, sometimes no intro
                            'RS_open': 4,
                            'RS_close': 4,
                            'QA_intro': 4,  # 3 or 4, sometimes no intro
                            'QA_trial': 200,  # 50/run * 4 runs
                            'QA_audio': 200,
                            'QA_ans': 200,
                            'QA_rec': 200,
                            'QA_cen_word': 200,
                            'Pause': 10,  # 4 runs = 10 : 1 + 2 + 3 + 4
                            'Break': 250,  # 200 QA + 20 Cali + 30 Arti
                            'Arti_intro': 2,  # pre + during
                            'Arti_trial': 30,  # 2 * 15
                            'Arti_action': 30,
                            'Arti_rec': 30,
                            }
    event_dict = {'ESC': 1,
                'Test': 253,
                'Main': 254,
                'End': 255,
                'Pre_run': [2, 3],
                'Post_run': [8, 9],
                'Run': [4, 5],
                'Block': [6, 7],
                'Cali_intro': [10, 11],
                'Cali_trial': [12, 13],
                'Cali_display': [14, 15],
                'Cali_ans': [16, 17],
                'Cali_rec': [18, 19],
                'Stim': [20, 21],
                'Sham': [22, 23],
                'Fade_in': [24, 25],
                'Fade_out': [26, 27],
                'Stable_stim': [28, 29],
                'RS_intro': [30, 31],
                'RS_open': [32, 33],
                'RS_close': [34, 35],
                'QA_intro': [40, 41],
                'QA_trial': [42, 43],
                'QA_audio': [44, 45],
                'QA_ans': [46, 47],
                'QA_rec': [48, 49],
                'QA_cen_word': [50, 51],
                'Pause': [60, 61],
                'Break': [62, 63],
                'Arti_intro': [70, 71],
                'Arti_trial': [72, 73],
                'Arti_action': [74, 75],
                'Arti_rec': [78, 79]}

    label_dict = [*event_dict.keys()]
    event_dict_expand = {}
    for i, keys in enumerate(label_dict):
        if i < 4:
            event_dict_expand[keys] = event_dict[keys]
            continue
        event_dict_expand[keys + '_start'] = event_dict[keys][0]
        event_dict_expand[keys + '_end'] = event_dict[keys][1]

    return nr_events_predefined, event_dict, label_dict, event_dict_expand


def insert_annot():

    insert_annot_list = np.full([11, 4], None)
    insert_annot_list[10, 1] = {'onset': 4329.352,
                                'duration': 0.001,
                                'description': 'Stimulus/S 72'}

    return insert_annot_list


def offset_loader(nr_subj, nr_ses):

    subject_list = {'test': [None, None, None, None],
                    'TES': [
                        None,
                        None,
                        None,
                        None],
                    'NUK': [
                        [(801.670, 3106.900),
                         (3145.150, 5029.112),
                         (5029.113, 8797.199),
                         (9906.416, None)],  # 5029.113
                        None,
                        None,
                        None],
                    'OSA': [
                        [(787.300, 8101.989),
                         (8217.100, None)],
                        None,
                        None,
                        None],
                    'KNL': [
                        None,
                        None,
                        None,
                        None],
                    'ZYC': [
                        [(838.700, 4950.833),
                         (4950.834, None)],
                        None,
                        None,
                        None],
                    'CCH': [
                        [(808.900, 7608.055),
                         (7643.249, 7907.637),
                         (7919.926, None)],
                        None,
                        None,
                        None,
                        None],
                    'DWS': [
                        [(718.049, 3804.511),
                         (3804.512, 8174.484),
                         (8174.485, None)],
                        None,
                        None,
                        None],
                    'VQT': [
                        [(1023.100, 4115.097),
                         (4115.098, None)],
                        None,
                        None,
                        None],
                    'BXB': [
                        None,
                        None,
                        None,
                        None],
                    'BMC': [
                        [(695.430, 1364.736),
                         (1364.737, None)],
                        None,
                        None,
                        None],
                    'ZWS': [
                        None,
                        None,
                        None,
                        None],
                    }

    subj_name = [*subject_list.keys()]

    return subject_list[subj_name[nr_subj]][nr_ses]



