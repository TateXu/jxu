# ========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2020-06-25 13:34:43
# Name       : preprocess.py
# Version    : V1.0
# Description: Class for preprocessing EEG& Audio
# ========================================

import os
import numpy as np
from scipy import signal
import warnings
import pickle
import pdb

from itertools import product
from copy import deepcopy

import mne
from mne import Epochs as mneEpoch
from mne.viz import plot_epochs_image
import matplotlib
# matplotlib.use('TkAgg')

from jxu.basiccmd.mycmd import create_folder
from jxu.data.loader import vhdr_load
from jxu.data.utils import nibs_event_dict
from jxu.viz.utils import *
from .utils import BaseEEG
from .base import NIBS

class NIBSEEG(NIBS):

    def __init__(self, subject=1, session=0,
                 root='/home/jxu/File/Data/NIBS/Stage_one/EEG/Exp/',
                 bands=[(0.1, 70.0)], fs=1000,
                 filter_para=dict(
                     method='iir', iir_params=dict(order=5, ftype='butter')),
                 reref='CA', bad_chn_list=None):
        super().__init__(subject=subject, session=session, root=root)
        self.bands = bands
        self.reref = reref
        self.fs = fs
        self.filter_para = filter_para
        self.bad_chn_list = bad_chn_list
    # -------- File opeartion ---------

    def raw_load(self):

        self.raw_data = [vhdr_load(self.root + eeg_file_path) for eeg_file_path in self.eeg_file_path_list]
        [single_raw.set_channel_types({'Audio': 'stim', 'tACS': 'stim'}) for single_raw in self.raw_data]
        if len(self.raw_data) != self.nr_seg:
            raise ValueError("#Seg != #Raw data")

        return self

    def raw_filter(self, bands=None, notch=True):
        print('------- Start filtering-------')
        if not hasattr(self, 'raw_data_clean') or self.raw_data is None:
            raise ValueError('Should pass concat. data to filter func')

        if self.bands is None and bands is not None:
            self.bands = bands

        if self.data is None:
            print('-------- WARN: The common average is not removed! --------')
            self.data = self.raw_data_clean.copy()
            self.unfiltered_data = self.raw_data_clean.copy()
        else:
            self.unfiltered_data = self.data.copy()

        tmp = []
        for fmin, fmax in self.bands:
            raw_f = self.data.filter(
                l_freq=fmin, h_freq=fmax, verbose=False, n_jobs=16)  #, **self.filter_para

            if notch:
                raw_f.notch_filter(freqs=np.arange(50, self.fs/2-1, 50),
                                picks='eeg', n_jobs=16)

            tmp.append(raw_f)

        self.data = tmp
        print('==============================================================')
        print('--- Data preprocessing is done, below are the attrs ----------')
        print('--------------------------------------------------------------')
        print('--- Dirty (Compeletely) raw data: self.raw_data --------')
        print('--- Clean raw data: self.raw_data_clean --------')
        print('--- Filtered data: self.data ---------------')
        print('--- Unfiltered data: self.unfiltered_data -----')
        print('==============================================================')

        return self

    def get_data(self, filter_flag=True):
        if filter_flag:
            if not hasattr(self, 'unfiltered_data') or self.unfiltered_data is None:
                self.raw_filter()
            return self.data
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

        data = self.concat_data.copy()
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
            '/home/jxu/anaconda3/lib/python3.7/site-packages/jxu/data/BC-TMS-128.bvef')
        for temp_data in self.raw_data:
            temp_data.set_montage(montage)
            if bad_chn_dict is not None:
                temp_data.set_channel_types(bad_chn_dict)

        return self

    def get_montage(self):
        return prinz('use plot_montage() to plot the electrodes location')

    def plot_montage(self, axes='mlab3D', name=False, surfaces='head'):

        if axes == 'mlab3D':
            from mayavi import mlab
            import os.path as op

            from mne.datasets import fetch_fsaverage
            from mne.viz import plot_alignment

            subjects_dir = op.dirname(fetch_fsaverage())
            fig = plot_alignment(self.raw_data[0].info, trans=None, subject='fsaverage',
                                 subjects_dir=subjects_dir, eeg=['projected'],
                                 surfaces=surfaces)
        elif axes == '2D' or axes=='topo':
            import pdb;pdb.set_trace()
            import mne
            img = mne.viz.plot_sensors(self.raw_data_clean.info,
                                       show_names=True, show=False)

            self.layout_2d = img
        elif axes == 'xyz3D':
            self.raw_data[0].info.plot(kind='3d')


        return self

    def set_trigger_list(self):
        pass


    def get_trigger_list(self):
        pass


    def set_channels(self, reset=False, bad_chn_list=[], bad_degree='extreme'):
        from jxu.data.utils import bad_chn_loader

        badchns = bad_chn_loader(self.subject, self.session)

        if bad_degree == 'extreme':
            pass
        elif bad_degree == 'mild':
            badchns.pop('extreme')
        elif bad_degree == 'default':
            for keys in ['extreme', 'mild']:
                badchns.pop(keys)
        else:
            raise ValueError('incorrect bad_degree values')

        for val in badchns.values():
            bad_chn_list.extend(val)

        # try:
            # print('Try to load local chn list')
            # with open(self.root + self.eeg_folder + 'bad_chn_list.pkl',
                      # 'rb') as f_in:
                # bad_chn_list = pickle.load(f_in)

            # if reset:
                # print("Previous bad channels are:")
                # print(bad_chn_list)
                # raise FileNotFoundError
        # except FileNotFoundError:

            # assert self.raw_data[0].info['ch_names'], "Run set_montage() first"
            # valid_chn_name = self.raw_data[0].info['ch_names']

            # if not bad_chn_list:
                # input_flag = 1
                # while input_flag:
                    # bad_chn_name = input("Please input the bad chn name or " +
                                         # " 0 -  finish input\n")

                    # if bad_chn_name == '0':
                        # input_flag = 0
                    # else:
                        # if bad_chn_name in valid_chn_name:
                            # bad_chn_list.append(bad_chn_name)
                        # else:
                            # print("Invalid chn name; All cap except h, Fp, XFp")

            # with open(self.root + self.eeg_folder + 'bad_chn_list.pkl',
                      # 'wb') as f_out:
                # pickle.dump(bad_chn_list, f_out)

        self.bad_chn_list = bad_chn_list
        self.bad_chn_dict = dict(zip(bad_chn_list, ['bads']*len(bad_chn_list)))

        self.raw_data_clean.set_channel_types(
            {'Audio': 'stim', 'tACS': 'stim'})
        self.raw_data_clean.set_channel_types(
            {'EOG151': 'eog', 'EOG152': 'eog'})
        self.raw_data_clean.info['bads'] = [
            'Audio', 'tACS', 'EOG151', 'EOG152'] + self.bad_chn_list
        # self.raw_data_clean.set_channel_types(self.bad_chn_dict)

        return self


    def get_bad_channels(self):
        return print('please check attr: bad_chn_list')


    # -------- Preprocessing ----------

    def filter(self):
        pass


    def notch_filter(self):
        pass


    def rereference(self, reref=None):

        if reref is None:
            reref = self.reref

        if reref == 'average':
            raw_ca = self.raw_data_clean.copy().set_eeg_reference(
                ref_channels='average')

        self.data = raw_ca

        return self


    # ----------- Sanity Check --------

    def data_concat(self, cp_flag=True, filtered_flag=False):

        from copy import deepcopy
        from .utils import offset_loader, insert_annot
        self.nr_evt, self.evt, self.lb_dict, self.evt_ext = nibs_event_dict()

        if filtered_flag:
            if cp_flag:
                raw_concat = deepcopy(self.filtered_data)
            else:
                raw_concat = self.filtered_data
        else:
            if cp_flag:
                raw_concat = deepcopy(self.raw_data)
            else:
                raw_concat = self.raw_data

        raw_concat = mne.concatenate_raws(raw_concat)
        self.trigger_detector(raw_concat)

        events, event_id = mne.events_from_annotations(raw_concat)
        loc, ts_loc = self.find_evt_loc(events, 253)
        ts_list = offset_loader(self.subject, self.session)
        if ts_list is not None:
            crop_list = []
            for (s_ts, e_ts) in ts_list:
                raw_cp = raw_concat.copy()
                raw_seg = raw_cp.crop(tmin=s_ts, tmax=e_ts)
                raw_seg.annotations.onset[[0, -1]]
                crop_list.append(raw_seg)
            self.raw_seg_clean = deepcopy(crop_list)
            self.raw_data_clean = mne.concatenate_raws(crop_list)
        else:
            self.raw_seg_clean = deepcopy(self.raw_data)
            self.raw_data_clean = deepcopy(raw_concat)

        self.reset_annot(self.raw_seg_clean)
        self.trigger_detector(self.raw_data_clean)
        events_, event_id_ = mne.events_from_annotations(self.raw_data_clean)

        return self

    def reset_annot(self, raw_list):
        from mne import Annotations
        from copy import deepcopy
        from .utils import insert_annot

        if not isinstance(raw_list, list):
            raw_list = [raw_list]

        raw_length = np.asarray(
            [temp._raw_lengths[0] for temp in self.raw_data])
        raw_length = np.r_[0, raw_length]
        cum_len = np.cumsum(raw_length)

        onset_list = []
        description_list = []
        duration_list = []

        seg_info = np.zeros([len(raw_list), 5], dtype="int64")
        for i, id_raw in enumerate(raw_list):
            seg_nr = int(id_raw._filenames[0][-5])
            init_shift = cum_len[seg_nr]
            first_sample = id_raw.first_samp
            last_sample = id_raw.last_samp
            assert len(id_raw._raw_lengths) == 1, "Non-unique raw seg len"
            clean_len = id_raw._raw_lengths[0]
            seg_info[i] = [seg_nr, init_shift, first_sample, last_sample,
                           clean_len]
            onset_list.append(id_raw.annotations.onset)
            description_list.append(id_raw.annotations.description)
            duration_list.append(id_raw.annotations.duration)

        cum_len_clean = np.cumsum(seg_info[:, -1])
        abs_loc = seg_info[:, 1:3].sum(axis=1)
        abs_shift = seg_info[:, 1]
        relative_loc = np.r_[0, cum_len_clean][:-1]
        relative_shift = abs_loc - relative_loc

        onset_shift = abs_shift - relative_shift

        self.raw_seg_data_clean = deepcopy(self.raw_seg_clean)

        for i, (id_raw, id_shift) in enumerate(zip(
                self.raw_seg_data_clean, onset_shift)):

            self.raw_seg_data_clean[i]._first_samps = np.asarray([
                id_raw.first_samp + id_shift])
            self.raw_seg_data_clean[i]._last_samps = np.asarray([
                id_raw.last_samp + id_shift])

            annot = id_raw.annotations
            assert annot.orig_time == id_raw.info['meas_date']
            annot.onset += id_shift / id_raw.info["sfreq"]

            self.raw_seg_data_clean[i]._update_times()
            self.raw_seg_data_clean[i].annotations.onset = annot.onset
            if i == 0:
                concat_annot = annot.copy()
            else:
                concat_onset = np.concatenate(
                    [concat_annot.onset, annot.onset])
                concat_dur = np.concatenate(
                    [concat_annot.duration, annot.duration])
                concat_dscp = np.concatenate(
                    [concat_annot.description, annot.description])

                concat_annot = Annotations(concat_onset,
                                           concat_dur,
                                           concat_dscp,
                                           concat_annot.orig_time)

        self.raw_data_clean._first_samps = np.asarray(
            [self.raw_data_clean._first_samps[0] + onset_shift[0]])
        self.raw_data_clean._last_samps = np.asarray(
            [self.raw_data_clean._last_samps[-1] + onset_shift[-1]])

        self.raw_data_clean._update_times()
        self.raw_data_clean.set_annotations(concat_annot)

        self.insert_annot_list = insert_annot()
        if self.insert_annot_list[self.subject, self.session] is not None:
            self.raw_data_clean.annotations.append(
                **self.insert_annot_list[self.subject, self.session])

        return self


    def onset_list(self, raw):
        if not isinstance(raw, list):
            raw = [raw]
        for id_raw in raw:
            print(id_raw.annotations.onset[[0, -1]])

        return self

    def find_evt_loc(self, events=None, evt=None, raw=None):

        if events is None:
            try:
                print('Find event from raw file: Not using passed event')
                events, event_id = mne.events_from_annotations(raw)
            except:
                print('Event matrix and raw file, at least one should be passed')
                import pdb;pdb.set_trace()

        if isinstance(evt, list):
            loc = np.empty((0,), dtype=int)
            for sg_evt in evt:
                sg_loc = np.where(events[:, 2] == sg_evt)[0]
                loc = np.append(loc, sg_loc)
            loc = np.sort(loc)
        else:
            loc = np.where(events[:, 2] == evt)[0]
        ts_loc = events[loc, :]

        return loc, ts_loc

    def data_seg(self, name=None, filtered_seg=0):

        data = self.data[filtered_seg].copy()
        loc, ts_loc = self.find_evt_loc(raw=data, evt=[32, 34])
        if name == "whole":
            return data
        elif name == 'pre':
            tmin = 0.0
            tmax = ts_loc[2, 0] / 1000.0 - 1.0
        elif name == 'stim':
            tmin = ts_loc[2, 0] / 1000.0
            tmax = ts_loc[6, 0] / 1000.0 - 1.0
        elif name == 'post':
            tmin = ts_loc[6, 0] / 1000.0
            tmax = None

        return data.crop(tmin=tmin, tmax=tmax)


    def trigger_detector(self, raw):

        event_dict = self.evt
        event_dict_expand = self.evt_ext
        nr_events_predefined = self.nr_evt
        events, event_id = mne.events_from_annotations(raw)
        # label_dict = [*event_dict.keys()]

        full_trigger_name_list = [*event_dict_expand.keys()]

        full_trigger_list = [*event_dict_expand.values()]
        presented_trigger_list = [*event_id.values()]

        unpresented_trigger_list = list(
            set(full_trigger_list).difference(
                set(full_trigger_list).intersection(presented_trigger_list)))
        unpresented_trigger_name_list = [full_trigger_name_list[full_trigger_list.index(tri_val)] for tri_val in unpresented_trigger_list]

        warnings.warn("Following triggers are not presented in the dataset:" +
                      ', '.join(unpresented_trigger_name_list))

        print("==============================================================")
        for ind, (key, val) in enumerate(event_dict.items()):

            if 'intro' in key:
                continue

            if ind < 4:
                val = [val, val]

            nr_start = events[np.where(
                events[:, 2] == val[0])[0], :].shape[0]
            nr_end = events[np.where(
                events[:, 2] == val[1])[0], :].shape[0]
            std_nr = nr_events_predefined[key]
            print(key + ', std/s/e: ' + str(std_nr) + '/' +
                  str(nr_start) + '/' + str(nr_end))
            # print(key + ', std/s: ' + str(std_nr) + '/' + str(nr_start))


        return self



class NIBSEpoch(mneEpoch):
    def __init__(raw, events, task='RS_close', when='all',
                 *args, **kwargs):
        self.task = task
        self.when = when

        when_list = ['pre', 'stim1', 'stim2', 'post', 'all']
        task_list = ['RS_close', 'RS_open',
                     'QA_trial', 'QA_audio', 'QA_ans', 'QA_rec', 'QA_cen_word',
                     'Arti_trial', 'Arti_action', 'Arti_rec']

        _, evt_dict, _, _ = nibs_event_dict()
        loc_when = when_list.index(self.when)

        evt_id = evt_dict[self.task][0]

        return super().__init__(raw=raw, events=events, event_id=[evt_id],
                                *args, **kwargs)



