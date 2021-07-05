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
                'Pre_run': [2, 3, None],
                'Post_run': [8, 9, None],
                'Run': [4, 5, None],
                'Block': [6, 7, None],
                'Cali_intro': [10, 11, None],
                'Cali_trial': [12, 13, None],
                'Cali_display': [14, 15, None],
                'Cali_ans': [16, 17, None],
                'Cali_rec': [18, 19],
                'Stim': [20, 21],
                'Sham': [22, 23],
                'Fade_in': [24, 25],
                'Fade_out': [26, 27],
                'Stable_stim': [28, 29],
                'RS_intro': [30, 31],
                'RS_open': [32, 33, 0, 180],
                'RS_close': [34, 35, 0, 180],
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

    insert_annot_list[6, 2] = {'onset': 8315.128,
                               'duration': 0.001,
                               'description': 'Stimulus/S 42'}
    return insert_annot_list

def bad_chn_loader(nr_subj, nr_ses):
    no_gel = ['FC3', 'FFC5h', 'FCC5h', 'FFC3h', 'FCC3h', 'T7', 'FTT7h', 'TTP7h']
    subject_list = {'test': [None, None, None, None],
                    'TES': [
                        {'extreme': ['FC1', 'CP5', 'FCC1h', 'CP5', 'C5'],
                         'mild': ['C3', 'Fz', 'C1', 'TP7', 'TPP7h', 'AFF1h', 'AFF2h', 'FC5']},
                        {'extreme': ['F3', 'P7', 'FC1', 'FC5', 'TP9', 'TPP7h', 'PPO9h', 'P9', 'PO9', 'TPP9h', 'POO9h'],
                         'mild': ['C5', 'PO7']},
                        {'extreme': ['C3', 'Fz', 'FC1', 'CP3', 'C5', 'FT7', 'TP7', 'CCP5h', 'FFT9h', 'FCC1h'],
                         'mild': ['FC5', 'CPP3h', 'AFp1', 'AFp2']},
                        {'extreme': ['FC1', 'TP7', 'FFC1h', 'FFC2h'],
                         'mild': ['F3', 'C3', 'FC5', 'TPP7h', 'FCC1h', 'TPP9h', 'AFp1', 'AFp2']}
                         ],
                    'NUK': [
                        {'extreme': ['Fz', 'C5', 'AFp1', 'AFp2', 'FFT7h'],
                         'mild': ['C3', 'FC1', 'FC2', 'FC5', 'AF3', 'AF4', 'AFF1h', 'AFF2h', 'FFC1h', 'FFC2h']},
                        {'extreme': ['C3', 'FC1', 'FC5', 'P9'],
                         'mild': ['F3', 'CP5', 'FT9', 'FT7', 'TP7', 'TPP7h', 'AFF1h', 'AFF2h']},
                        {'extreme': ['F3', 'C3', 'FC1', 'FT7', 'F9', 'FFT9h', 'FT9'],
                         'mild': ['F7', 'FC5', 'AFF1h', 'AFF2h', 'FFC1h', 'AFp1', 'AFp2', 'FFT7h']},
                        {'extreme': ['C3', 'FC1', 'FC5', 'FT7', 'AFF1h', 'AFF2h'],
                         'mild': ['Fz', 'C5', 'TP7', 'FCC1h', 'AFp1', 'AFp2', 'CCP3h']}
                        ],
                    'OSA': [
                        {'extreme': ['F3', 'C3', 'FC5', 'TP9', 'F5', 'TP7', 'TPP7h', 'AFF5h', 'TPP9h'],
                         'mild': ['CPP5h']},  # Very noisy, HF noise are too much
                        {'extreme': ['FC5', 'C5', 'TP7'],
                         'mild': ['F3', 'C3', 'Cz', 'FC1', 'CP1', 'FT9', 'AFp2']},  # Even worse, nothing distinguishable
                        {'extreme': ['C3', 'FC5', 'FC1', 'CP5', 'CP3', 'TP7', 'Fpz', 'FT7', 'FCC3h', 'FTT7h', 'CPP3h', 'AFp1', 'AFp2', 'FCC5h', 'FFT7h', 'TTP7h'],
                         'mild': ['F5', 'C5', 'TPP7h']},
                        {'extreme': ['F3', 'C3', 'FC5', 'FT9', 'TP9', 'FT7', 'TP7', 'CCP5h'],
                         'mild': ['AFF5h', 'FFT7h']}
                        ],
                    'KNL': [
                        {'extreme': ['', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                         'mild': ['', '', '', '', '', '', '', '', '', '', '', '', '', '']},
                        {'extreme': ['', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                         'mild': ['', '', '', '', '', '', '', '', '', '', '', '', '', '']},
                        {'extreme': ['', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                         'mild': ['', '', '', '', '', '', '', '', '', '', '', '', '', '']},
                        {'extreme': ['', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                         'mild': ['', '', '', '', '', '', '', '', '', '', '', '', '', '']}
                        ],
                    'ZYC': [
                        {'extreme': ['FC5', 'C5', 'TP7'],
                         'mild': []},  # no use at all
                        {'extreme': ['F3', 'FC1', 'FC5', 'FT9', 'F1', 'AF3', 'F5', 'AF7', 'FT7', 'Fpz', 'FFC1h', 'FFC2h', 'AFF1h', 'F9', 'AFp1', 'AFF5h', 'FFT7h', 'FFT9h'],
                         'mild': []},
                        {'extreme': ['F3', 'C3', 'FC1', 'FC5', 'CP3', 'FT7', 'P9'],
                         'mild': ['Fz', 'C5', 'AFF1h', 'AFF2h', 'CCP5h', 'AFp1', 'AFp2']},
                        {'extreme': ['C3', 'FC1', 'FT9', 'C1', 'C5', 'TP7', 'Fpz', 'AFF1h', 'AFF2h', 'CCP5h', 'AFp1', 'AFp2', 'FFT7h'],
                         'mild': ['FFT9h', 'FFT10h']}
                        ],
                    'CCH': [
                        {'extreme': ['FC3', 'C3', 'CP5', 'TP9', 'F5', 'C5', 'AF7', 'TP7', 'CCP5h', 'TPP7h', 'P9', 'AFF5h', 'TPP9h'],
                         'mild': ['AF3', 'FFC2h']},
                        {'extreme': ['F3', 'FC1', 'F1', 'TP7', 'FT7', 'TPP7h', 'TP10', 'FC5'],
                         'mild': ['CP5', 'FT9', 'TP9', 'C5', 'FFC2h', 'TPP9h']},
                        {'extreme': ['F3', 'FC5', 'F5', 'TP7', 'AFF5h'],
                         'mild': ['FC1', 'C3', 'AFF1h', 'AFF2h', 'FFT7h']},
                        {'extreme': ['F3', 'C3', 'FC1', 'TP7'],
                         'mild': ['FC5', 'AFF1h', 'AFF2h', 'AFp1', 'AFp2']}
                        ],
                    'DWS': [
                        {'extreme': ['F3', 'FT7'],
                         'mild': ['C3', 'Cz', 'FC5', 'FC1', 'TP7', 'C5', 'F5', 'P9', 'CCP5h']},
                        {'extreme': ['C3', 'FC1', 'FT7'],
                         'mild': ['FC5', 'TP7', 'CCP5h', 'AFF1h', 'AFF2h', 'AFp1', 'AFp2', 'CCP3h']},
                        {'extreme': ['C3', 'F3', 'FC1', 'FC5', 'C5', 'FT7', 'TP7', 'AF8'],
                         'mild': ['TPP7h', 'AFF1h', 'AFF2h', 'CCP5h', 'AFp1', 'AFp2']},
                        {'extreme': ['CP5', 'FC1', 'C5', 'FT7', 'TP7', 'TPP7h'],
                         'mild': ['F3', 'C3', 'Fz', 'Cz', 'P5', 'AFF1h', 'AFF2h', 'AFp1', 'AFp2', 'TPP9h']}
                        ],
                    'VQT': [
                        {'extreme': ['F3', 'C3', 'FC1', 'AF3', 'F5', 'C5', 'FT7', 'AF7', 'TP7', 'TPP7h', 'AFF5h', 'TPP9h'],
                         'mild': ['CP5', 'AFF1h', 'AFF2h', 'AFp1', 'AFp2']},
                        {'extreme': ['F3', 'C3', 'F5', 'FT7', 'AFF5h'],
                         'mild': ['FC1', 'C1', 'FCC1h', 'CCP3h', 'AFF5h', 'FFT7h']},  # No alpha existed
                        {'extreme': ['F3', 'C3', 'FC1', 'FC5', 'FT9', 'CP3', 'C5', 'FT7', 'TP7', 'CCP3h', 'CCP5h', 'FFT9h'],
                         'mild': ['Fpz', 'AFF1h', 'AFF2h', 'CCP5h', 'AFp1', 'AFp2', 'FFT7h']},
                        {'extreme': ['C3', 'C5', 'FT7', 'TP7', 'CCP5h'],
                         'mild': ['FC1', 'CP5']}
                        ],
                    'BXB': [
                        {'extreme': ['', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                         'mild': ['', '', '', '', '', '', '', '', '', '', '', '', '', '']},
                        {'extreme': ['', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                         'mild': ['', '', '', '', '', '', '', '', '', '', '', '', '', '']},
                        {'extreme': ['', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                         'mild': ['', '', '', '', '', '', '', '', '', '', '', '', '', '']},
                        {'extreme': ['', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                         'mild': ['', '', '', '', '', '', '', '', '', '', '', '', '', '']}
                        ],
                    'BMC': [
                        {'extreme': ['F3', 'FC5', 'C3', 'TP9', 'C5', 'FT7', 'TP7', 'CCP3h', 'AFp1', 'FFT9h'],
                         'mild': ['AFF1h', 'AFF2h', 'FCC4h', 'AFp2']},  #  very bad quality
                        {'extreme': ['F3', 'FC1', 'FT9', 'C5', 'AF7', 'FT7', 'TP7', 'FCC1h', 'FFT7h', 'AFF5h'],
                         'mild': ['Fpz', 'AFF1h', 'AFF2h', 'AFp1', 'AFp2']},
                        {'extreme': ['F3', 'C3', 'C5', 'F7', 'FC1', 'FT9', 'F1', 'AF3', 'F5', 'AF7', 'FT7', 'TP7', 'F9', 'AFF5h', 'FFT7h', 'FFT9h'],
                         'mild': ['FC5', 'CP5', 'Fpz', 'AFF1h', 'AFF2h', 'CCP5h', 'TPP7h', 'AFp1', 'AFp2', 'CCP3h']},
                        {'extreme': ['F3', 'FC1', 'FC5', 'CP5', 'FT9', 'F5', 'C5', 'FT7', 'CCP5h', 'F9', 'AFp1', 'AFp2', 'AFF5h', 'FFT7h', 'FFT9h'],
                         'mild': ['Fpz', 'C1', 'AFF1h', 'AFF2h']}
                        ],
                    'ZWS': [
                        {'extreme': ['', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                         'mild': ['', '', '', '', '', '', '', '', '', '', '', '', '', '']},
                        {'extreme': ['', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                         'mild': ['', '', '', '', '', '', '', '', '', '', '', '', '', '']},
                        {'extreme': ['', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                         'mild': ['', '', '', '', '', '', '', '', '', '', '', '', '', '']},
                        {'extreme': ['', '', '', '', '', '', '', '', '', '', '', '', '', ''],
                         'mild': ['', '', '', '', '', '', '', '', '', '', '', '', '', '']}
                        ],
                    }

    subj_name = [*subject_list.keys()]
    subject_list[subj_name[nr_subj]][nr_ses]['default'] = no_gel

    return subject_list[subj_name[nr_subj]][nr_ses]




def offset_loader(nr_subj, nr_ses):

    subject_list = {'test': [None, None, None, None],
                    'TES': [
                        [(843.954, 7364.600),
                         (7576.427, 8752.739),
                         (8752.740, None)],
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
                        [(0.0, 690.194),
                         (856.415, None)],
                        [(0.0, 3925.960),
                         (4272.735, None)]
                        ],
                    'KNL': [
                        None,
                        None,
                        None,
                        None],
                    'ZYC': [
                        [(838.700, 4950.833),
                         (4950.834, None)],
                        None,
                        [(0.0, 1763.348),
                         (1848.182, 9215.900),
                         (9306.119, None)],
                        None],
                    'CCH': [
                        [(808.900, 7608.055),
                         (7643.249, 7907.637),
                         (7919.926, None)],
                        [(0.0, 2342.839),
                         (2342.840, 7307.600),
                         (7584.217, None)],
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
                        [(0.0, 4354.860),
                         (4450.739, None)]],
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



