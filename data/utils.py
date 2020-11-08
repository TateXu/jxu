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
    nr_events_predefined = {'Pre_run': 1,
                            'Post_run': 1,
                            'Run': 4,
                            'Block': 8,
                            'Cali_intro': 2,
                            'Cali_trial': 20,
                            'Cali_display': 20,
                            'Cali_ans': 20,
                            'Cali_rec': 20,
                            'Stim': 4,
                            'Sham': 4,
                            'Fade_in': 4,
                            'Fade_out': 4,
                            'Stable_stim': 4,
                            'RS_intro': 8,
                            'RS_open': 4,
                            'RS_close': 4,
                            'QA_intro': 4,
                            'QA_trial': 160,
                            'QA_audio': 160,
                            'QA_ans': 160,
                            'QA_rec': 160,
                            'QA_cen_word': 160,
                            'Pause': 6,
                            'Break': 180}
    """
    Old version of event list
    event_dict = {'Pre_run': [0, 1],
                  'Post_run': [2, 3],
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
                  'Break': [62, 63]}
    """

    event_dict = {'ESC': 1,
                'Test': 253,
                'Main': 254,
                'End':255,
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


def trigger_detector(raw, event_dict, event_dict_expand, nr_events_predefined):

    events, event_id = mne.events_from_annotations(raw)
    label_dict = [*event_dict.keys()]

    full_trigger_name_list = [*event_dict_expand.keys()]
    full_trigger_list = [*event_dict_expand.values()]
    presented_trigger_list = [*event_id.values()]

    unpresented_trigger_list = list(
        set(full_trigger_list).difference(
            set(full_trigger_list).intersection(presented_trigger_list)))
    unpresented_trigger_name_list = [full_trigger_name_list[full_trigger_list.index(tri_val)] for tri_val in unpresented_trigger_list]

    warnings.warn("Following triggers are not presented in the dataset:" +
                  ', '.join(unpresented_trigger_name_list))

    for ind, (key, val) in enumerate(event_dict.items()):
        nr_start = events[np.where(
            events[:, 2] == val[0])[0], :].shape[0]
        nr_end = events[np.where(
            events[:, 2] == val[1])[0], :].shape[0]
        std_nr = nr_events_predefined[key]
        print(key + ', std: ' + str(std_nr) + ', s/e: ' +
              str(nr_start) + '/' + str(nr_end))


