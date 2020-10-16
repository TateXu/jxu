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
from jxu.data.utils import *
from jxu.viz.utils import *
import os
import numpy as np
from numpy.fft import fft, fftfreq
from scipy import signal
import matplotlib.pyplot as plt
from itertools import product
from mne.viz import plot_epochs_image
import platform
from copy import deepcopy

from pydub import AudioSegment
from pydub.playback import play as pd_play
from jxu.audio.audiosignal import audio_denoise, wav_std
from auditok import AudioRegion


from scipy.io import wavfile # scipy library to read wav files
import mne

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


# this is just a test for webhook
# this is just a test for webhook


class NIBSEEG(NIBS):

    def __init__(self, subject=1, session=0,
                 root='/home/jxu/File/Data/NIBS/Stage_one/EEG/Exp/',
                 bands=[(0.1, 70.0)],
                 filter_para=dict(
                     method='iir', iir_params=dict(order=5, ftype='butter')),
                 reref='CA', bad_chn_list=None):
        super().__init__(subject=subject, session=session, root=root)
        self.bands = bands
        self.reref = reref
        self.filter_para = filter_para
        self.bad_chn_list = bad_chn_list
    # -------- File opeartion ---------

    def raw_load(self):

        self.raw_data = [vhdr_load(self.root + eeg_file_path) for eeg_file_path in self.eeg_file_path_list]
        [single_raw.set_channel_types({'Audio': 'stim', 'tACS': 'stim'}) for single_raw in self.raw_data]
        if len(self.raw_data) != self.nr_seg:
            raise ValueError("#Seg != #Raw data")

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


class NIBSAudio(NIBS):

    def __init__(self, subject=1, session=0,
                 root='/home/jxu/File/Data/NIBS/Stage_one/EEG/Exp/'):
        # load audio from data folder
        # load default denoise level and other parameters, e.g. skip, etc.
        super().__init__(subject=subject, session=session, root=root)

        self.nr_run = 4
        self.nr_block = 2

        self.nr_cali_trial = 10
        self.nr_block_trial = 25
        self.nr_run_trial = self.nr_block * self.nr_block_trial
        self.nr_qa_trial = self.nr_run * self.nr_run_trial


    def audio_load(self, preload=False, pkg='AudioSegment',
                   audio_type='answer', std=False):

        self.load_std = deepcopy(std)
        if std:
            print('Loading standardised audio file: 44100 sps, 16bit, mono channel')
            std_suffix = '_std'
        else:
            print('Loading original audio file: 44100 sps, 32bit, mono channel')
            std_suffix = ''

        if audio_type == 'answer':
            self.cali_pre_file = [
                self.audio_folder +
                'rec_cali_de_pre_trial_{0}{1}.wav'.format(
                    str(id_trial).zfill(3),
                    std_suffix) for id_trial in range(self.nr_cali_trial)]
            self.cali_post_file = [
                self.audio_folder +
                'rec_cali_de_post_trial_{0}{1}.wav'.format(
                    str(id_trial).zfill(3),
                    std_suffix) for id_trial in range(self.nr_cali_trial)]

            self.qa_file = []
            for id_qa_trial in range(self.nr_qa_trial):

                v1, v2 = divmod(id_qa_trial, self.nr_run_trial)
                id_run, id_block, id_trial = (v1,) + \
                    divmod(v2, self.nr_block_trial)

                self.qa_file.append(
                    self.audio_folder +\
                    'rec_QA_run_{0}_block_{1}_trial_{2}{3}.wav'.format(
                        str(id_run).zfill(2),
                        str(id_block).zfill(3),
                        str(id_trial).zfill(3),
                        std_suffix))

            self.audio_filename_list = [self.cali_pre_file, self.qa_file,
                                        self.cali_post_file]
            if preload:
                self.answer_file_list = self._preload(pkg=pkg)
                print('Loading audio files into self.answer_file_list')
            else:
                import itertools
                self.answer_file_list = list(
                    itertools.chain.from_iterable(self.audio_filename_list))
                print('Loading audio file names into self.answer_filename_list')
        elif audio_type == 'question':
            import pandas as pd
            self.cali_pkl = '{0}cali_info/S{1}_Session{2}_cali.pkl'.format(
                self.meta_folder, str(self.subj_nr).zfill(2),
                str(self.ses_nr).zfill(1))
            self.qa_pkl = '{0}qa_info/S{1}_Session{2}_unshattered_beep_df.pkl'.format(
                self.meta_folder, str(self.subj_nr).zfill(2),
                str(self.ses_nr).zfill(1))

            self.cali_info = pd.read_pickle(self.cali_pkl)
            self.qa_info = pd.read_pickle(self.qa_pkl)

            if 'file_seg_ori' not in self.qa_info.PATH:
                self.qa_info[('PATH', 'file_seg_ori')] = self.qa_info[
                    'PATH']['file_root_ori'].replace(
                        {'/Audio/Database/Unshattered/audio/':
                         '/EEG/Exp/Audio/audio_file/'},
                        regex=True)

        return self

    def _preload(self, pkg):
        # add a logger
        audio_file_list = []
        if pkg == 'scipy':
            for id_list in self.audio_filename_list:
                [audio_file_list.append(
                    wavfile.read(
                        single_audio_file)) for single_audio_file in id_list]
        elif pkg == 'AudioSegment':
            for id_list in self.audio_filename_list:
                [audio_file_list.append(
                    AudioSegment.from_file(
                        single_audio_file)) for single_audio_file in id_list]
        else:
            raise ValueError('Unsupported input for keyword pkg.\n'
                                'Use scipy or AudioSegment')

        return audio_file_list


    def audio_format_check(self, audio_type='answer'):
        # Recorded answer: 44100 sps, 32bit (4 Bytes), mono
        # Google TTS: 24000 sps, 16bit, mono

        if audio_type == 'answer':
            try:
                self.format_check_list = deepcopy(self.answer_file_list)
            except:
                raise ValueError('Load the answer audio first!')
        elif audio_type == 'question':
            try:
                self.format_check_list = deepcopy(self.question_file_list)
            except:
                raise ValueError('Load the question audio first!')

        para_list = np.asarray(
            [[sg_file.frame_rate, sg_file.frame_width,
              sg_file.channels] for sg_file in self.format_check_list]
            )
        if len(np.unique(para_list).shape) == 1:
            return self
        else:
            raise ValueError('Audio data format is not unique')

        return self


    def audio_std(self):

        assert self.answer_file_list is not None
        import pdb;pdb.set_trace()

        print('Standardise the audio files')
        for sg_file in self.answer_file_list:
            _ = wav_std(sg_file, sps=44100, bit=16)

        return self

    def audio_to_seg(self, opt_skip=0.15, opt_ET=51.5, opt_noise_level=4):

        from jxu.audio.audiosignal import audio_denoise, wav_std
        from auditok import AudioRegion
        from pydub.playback import play as pd_play
        import progressbar

        assert self.answer_file_list is not None

        self.denoise_answer_file_list = np.empty((len(self.answer_file_list)),
                                                 dtype=object)
        try:
            for id_sg_file, sg_file in enumerate(self.answer_file_list):
                self.denoise_answer_file_list[id_sg_file] = audio_denoise(
                    filename=sg_file, process=False, denoise_level=opt_noise_level,
                    new_folder=True)
        except FileNotFoundError:
            for id_sg_file, sg_file in enumerate(self.answer_file_list):
                self.denoise_answer_file_list[id_sg_file] = audio_denoise(
                    sg_file, process=True, denoise_level=opt_noise_level,
                    new_folder=True)
        except Exception as ex:
            raise ValueError('Fail to run audio_to_seg.')

        # Mark the audio file with answer
        self.seg_folder = self.audio_folder + 'Segments/'
        create_folder(self.seg_folder)
        create_folder(self.audio_folder + 'Marker/')

        if os.path.exists(self.audio_folder + 'Marker/answer.pkl'):
            with open(
                    self.audio_folder + 'Marker/answer.pkl', 'rb') as f_mark:
                self.ans_marker, self.seg_marker = pickle.load(f_mark)
            init_ind = np.sum(self.ans_marker != None)
        else:
            self.ans_marker = np.empty((self.nr_qa_trial), dtype=object)
            self.seg_marker = np.empty((self.nr_qa_trial, 6), dtype=object)
            init_ind = 0
        bar = progressbar.ProgressBar(max_value=self.nr_qa_trial)

        for id_sg_file in range(10 + init_ind, 210):
            sg_audio = AudioSegment.from_file(
                self.denoise_answer_file_list[id_sg_file])
            continue_flag = 'r'
            # Start to slice into segments and mark them.
            ar_obj = AudioRegion.load(
                self.denoise_answer_file_list[id_sg_file], skip=opt_skip)
            self._raw_seg_marker(ar_obj, ET=opt_ET, skip=opt_skip,
                                 id_trial=id_sg_file-10)
            empty_ans_flag = self.seg_marker[id_sg_file-10][0] == 0
            all_noise_seg_flag = not np.any(np.asarray(self.seg_marker[id_sg_file-10][1]))
            if any([empty_ans_flag, all_noise_seg_flag]):
                while continue_flag == 'r':
                    pd_play(sg_audio)
                    continue_flag = input('Does this audio contain an answer?' +
                                        '(1-yes/ 0-no/ r-repeat)?\n')
                    while continue_flag not in ['r', '1', '0']:
                        continue_flag = input('Invalid input, please only input ' +
                                            'r, 1 or 0!\n')
                    if continue_flag.lower() == 'r':
                        continue
                    else:
                        self.ans_marker[id_sg_file-10] = int(continue_flag)
            else:
                self.ans_marker[id_sg_file-10] = int(1)

            with open(
                    self.audio_folder +
                    'Marker/answer.pkl', 'wb') as f_ans_mark:
                pickle.dump([self.ans_marker, self.seg_marker], f_ans_mark)
            bar.update(id_sg_file - 10)

        return self


    def _raw_seg_marker(self, ar_obj, ET, skip, id_trial):
        # AudioRegion.sr/sw/ch
        # manual detection that the seg is audio(1) or noise(0)
        # return bool

        audio_segs = list(ar_obj.split_and_plot(energy_threshold=ET))
        self.seg_marker[id_trial][0] = len(audio_segs)
        self.seg_marker[id_trial][1] = []  # flag: bool, audio or noise
        self.seg_marker[id_trial][2] = []  # onset
        self.seg_marker[id_trial][3] = []  # end = onset + duration
        self.seg_marker[id_trial][4] = []  # crop_ext_left
        self.seg_marker[id_trial][5] = []  # crop_ext_right
        for id_seg, seg in enumerate(audio_segs):
            seg_name = '{0}T_{1}_seg_{2}.wav'.format(
                self.seg_folder, str(id_trial), str(id_seg))
            seg.save(seg_name)
            as_seg = AudioSegment.from_file(seg_name)
            continue_flag = 'r'
            while continue_flag == 'r':
                pd_play(as_seg)
                continue_flag = input('Does this audio clip contain an' +
                                      ' answer? (1-yes/ 0-no/ r-repeat)\n')
                while continue_flag not in ['r', '1', '0']:
                    continue_flag = input('Invalid input, please only input ' +
                                          'r, 1 or 0!\n')
                if continue_flag.lower() == 'r':
                    continue
                else:
                    self.seg_marker[id_trial, 1].append(
                        int(continue_flag))
                    self.seg_marker[id_trial, 2].append(
                        seg.meta.start + skip)
                    self.seg_marker[id_trial, 3].append(
                        seg.meta.end + skip)

                    if int(continue_flag):
                        crop_l_val = 0.3
                        crop_r_val = 0.3
                        # crop_ext_left_flag = input('Manaul ext for crop' +
                                                   # ' left? (1-yes/0-no)\n')
                        # if int(crop_ext_left_flag):
                            # crop_l_val = float(
                                # input('Input the value for crop left' +
                                      # ' (default: +0.3)\n'))
                        # else:
                            # crop_l_val = 0.3

                        # crop_ext_right_flag = input('Manaul ext for crop' +
                                                    # ' right?(1-yes/0-no)\n')
                        # if int(crop_ext_right_flag):
                            # crop_r_val = float(
                                # input('Input the value for crop right' +
                                      # ' (default: +0.3)\n'))
                        # else
                            # crop_r_val = 0.3

                        self.seg_marker[id_trial, 4].append(
                            crop_l_val)
                        self.seg_marker[id_trial, 5].append(
                            crop_r_val)
        plt.close()

        return self

    def valid_seg(self):
        from jxu.audio.audiosignal import detect_leading_silence, audio_onedim
        from spellchecker import SpellChecker
        import progressbar

        spell = SpellChecker(language='de')

        # return onset& duration of valid segments
        # for answer segments
        assert self.seg_marker is not None, "Run audio_to_seg() first!"

        flag_list = np.asarray(
            [True if 1 in ind else False for ind in self.seg_marker[:, 1]])
        if np.any(flag_list == self.ans_marker):
            print("Inconsistent #seg; Start to check one by one!")

            captured_ind = np.where((flag_list == self.ans_marker) == False)[0]
            print(captured_ind)
            for ind in captured_ind:

                audio_file = AudioSegment.from_wav(
                    self.denoise_answer_file_list[ind + 10])

                pd_play(audio_file)
                contain_clip_flag = input('Is there answer inside? 1-yes/0-no')
                if int(contain_clip_flag):
                    self.seg_marker[ind] = [1, [1], [0.001],
                                            [len(audio_file)/1000.0-0.001],
                                            [0.0], [0.0]]
                    self.ans_marker[ind] = True
                else:
                    self.seg_marker[ind] = [0, [], [], [], [], []]
                    self.ans_marker[ind] = False
                    self.answer_score
            with open(
                    self.audio_folder +
                    'Marker/answer.pkl', 'wb') as f_ans_mark:
                pickle.dump([self.ans_marker, self.seg_marker], f_ans_mark)

        create_folder(self.audio_folder + 'Valid_segs/')

        try:
            with open(self.audio_folder + 'Marker/valid_answer.pkl',
                      'rb') as f_valid_in:
                self.valid_seg_marker = pickle.load(f_valid_in)
            init_ind = np.sum(self.valid_seg_marker[:, 0] != None)
        except FileNotFoundError:
            self.valid_seg_marker = np.empty(
                (self.nr_qa_trial, 5), dtype='object')
            init_ind = 0

        extract_func = lambda x, y: [x[i][y[i]] for i in range(len(x))]

        bar_valid = progressbar.ProgressBar(max_value=self.nr_qa_trial)
        for ind_file, (valid_seg_file, seg_file) in enumerate(
                zip(self.valid_seg_marker, self.seg_marker)):

            bar_valid.update(ind_file)

            if ind_file < init_ind:
                continue

            if not flag_list[ind_file]:
                self.valid_seg_marker[ind_file, 0] = flag_list[ind_file]
                continue
            # import pdb;pdb.set_trace()


            # All seg location in seg_file
            if len(seg_file[1]) == 1:
                single_vl_seg_flag = True
            else:
                single_vl_seg_flag = False

            all_vl_loc = np.where(np.asarray(seg_file[1]) == 1)[0]

            for vl_loc in all_vl_loc:
                vl_loc_list = [vl_loc, vl_loc, 0, 0]

                start, end, ext_l, ext_r = extract_func(seg_file[2:], vl_loc_list)

                crop_start = start - ext_l if start - ext_l >= 0 else 0.0
                crop_end = end + ext_r

                audio_file = AudioSegment.from_wav(
                    self.denoise_answer_file_list[ind_file + 10])
                crop_audio = deepcopy(audio_file)
                audio_seg = crop_audio[crop_start * 1000: crop_end * 1000]
                start_trim = detect_leading_silence(
                    audio_seg, silence_threshold=-45.0, chunk_size=1)
                end_trim = detect_leading_silence(
                    audio_seg.reverse(), silence_threshold=-45.0, chunk_size=1)
                onset = crop_start + start_trim / 1000.0
                duration = len(audio_seg[start_trim:-end_trim-1]) / 1000.0

                continue_flag = 'r'
                vocab_correct_flag = 0
                while continue_flag == 'r':
                    valid_audio = deepcopy(audio_file)
                    valid_clip = valid_audio[
                        onset * 1000: (onset + duration) * 1000]

                    pd_play(valid_clip)
                    valid_seg_loc = '{0}Valid_segs/QA_trial_{1}.wav'.format(
                        self.audio_folder, str(ind_file))
                    valid_clip.export(valid_seg_loc, format='wav')
                    audio_onedim(valid_seg_loc, block=False, metric='default')

                    continue_flag = input('Does this audio clip completely ' +
                                          'contain an answer? (1-yes/ 0-no/' +
                                          'r-repeat/ w-play whole audio' +
                                          '/q-play question audio/d-delete)\n')

                    if not int(vocab_correct_flag) and continue_flag == '1':
                        vocab_text_input = input(
                            'Please input the heard word\n')
                        crct_vocab = spell.correction(vocab_text_input)
                        vocab_correct_flag = crct_vocab == vocab_text_input
                        print('Valid input?  {0} \n Input: {1} \n Correct: {2}\n'.format(
                              vocab_correct_flag, vocab_text_input, crct_vocab))
                        if not int(vocab_correct_flag):
                            acp_flag = input('Accept suggestion? 1-yes' +
                                             'input the new text \n')
                            if acp_flag == '1':
                                vocab_text_input = deepcopy(crct_vocab)
                            else:
                                vocab_text_input = deepcopy(acp_flag)
                            vocab_correct_flag = True

                    while continue_flag not in ['r', '1', '0', 'w', 'q', 'd']:
                        continue_flag = input('Invalid input, please ' +
                                              'only input r, 1 or 0!\n')
                    if continue_flag.lower() == 'r':
                        continue
                    elif continue_flag.lower() == 'd':
                        # Turning flag from True to False
                        self.seg_marker[ind_file][1][vl_loc] = 0

                        # Pop the element of ext_l and ext_r, ps. all with 0.3,
                        # hence the order does not matter
                        self.seg_marker[ind_file][4].pop()
                        self.seg_marker[ind_file][5].pop()

                        print('Delete a pre-marked seg, update the answer.pkl')
                        with open(self.audio_folder + 'Marker/answer.pkl',
                                  'wb') as f_ans_mark:
                            pickle.dump([self.ans_marker, self.seg_marker],
                                        f_ans_mark)
                        print('Rerun the program to refresh flag list')
                        exit()
                    elif continue_flag.lower() == 'q':
                        q_audio = AudioSegment.from_wav(
                            self.qa_info.PATH.file_root_syn[ind_file])
                        pd_play(q_audio)
                        continue_flag = 'r'
                    elif continue_flag.lower() == 'w':
                        pd_play(valid_audio)
                        audio_onedim(
                            self.denoise_answer_file_list[ind_file + 10],
                            valid_seg_loc,
                            block=False, metric='default', num=2)
                        continue_flag = 'r'
                    elif continue_flag.lower() == '0':
                        audio_onedim(
                            self.denoise_answer_file_list[ind_file + 10],
                            valid_seg_loc,
                            block=False, metric='default', num=2)
                        shift_l = float(
                            input('Shift how much towords the left?' +
                                  ' +/-, l/r (unit:s)'))
                        shift_r = float(
                            input('Shift how much towords the right?' +
                                  ' +/-, l/r (unit:s)'))
                        onset -= shift_l
                        if onset < 0:
                            onset = 0.001
                        duration += shift_l + shift_r
                        continue_flag = 'r'
                    elif continue_flag.lower() == '1' and not int(vocab_correct_flag):
                        continue_flag = 'r'
                    elif continue_flag.lower() == '1' and int(vocab_correct_flag):
                        self.save_metrics(
                            ind_file, single_vl_seg_flag,
                            [flag_list[ind_file], valid_seg_loc, onset,
                             duration, vocab_text_input])

                    plt.close('all')

        return self

    def save_metrics(self, ind_file, single_seg_flag, metrics_list):

        assert len(metrics_list) == 5, '#metrics is predefined as 5!'
        for id_sg_metric, single_metric in enumerate(metrics_list):
            if single_seg_flag:
                self.valid_seg_marker[ind_file, id_sg_metric] = single_metric
            else:
                if self.valid_seg_marker[ind_file, id_sg_metric] == None:
                    # initialize as a list to append multiple seg metric
                    self.valid_seg_marker[ind_file, id_sg_metric] = []

                self.valid_seg_marker[ind_file, id_sg_metric].append(
                    single_metric)

        with open(self.audio_folder +
                  'Marker/valid_answer.pkl', 'wb') as f_valid:
            pickle.dump(self.valid_seg_marker, f_valid)

        return self

    def save_vl_marker_pkl(self):

        with open(self.audio_folder +
                  'Marker/valid_answer.pkl', 'wb') as f_valid:
            pickle.dump(self.valid_seg_marker, f_valid)

        return self


    def qa_combine(self, update_pkl=True):

        from jxu.basiccmd.mycmd import create_folder
        from jxu.audio.audiosignal import plain_beep, remove_silence
        from pydub import AudioSegment

        assert self.valid_seg_marker is not None, 'Run valid_seg() first!'

        import pdb;pdb.set_trace()
        try:
            if self.qa_info.PATH.file_qa_combined.shape[0] == self.nr_qa_trial:

                print('QA is already combined. Attribute ' + \
                      'qa_info.PATH.file_qa_combined will be returned.')
                rerun = input('Forced to rerun this QA combine function? 1/0')
                if not rerun:
                    return self
                print('Force to rerun QA combined')
        except AttributeError:
            print('QA is not yet combined. This function will continue')

        self.qa_combine_folder = self.audio_folder + 'Combined/'
        create_folder(self.qa_combine_folder)

        self.qa_info[('PATH', 'valid_ans_seg')] = self.valid_seg_marker[:, 1]
        combined_file_list = []
        for index, entry in self.qa_info.iterrows():
            print(index)
            chunk_folder_path = '/'.join(
                entry.PATH.file_seg_ori.split('/')[:-1]) + '/Chunk/'

            nr_chunk = ~entry.SENTENCE_INFO.last_word_flag + 4

            audio_chunks = []
            for id_chunk in range(nr_chunk):
                audio_chunks.append(
                    AudioSegment.from_wav(
                        chunk_folder_path +
                        'chunk{0}.wav'.format(str(id_chunk))))

            if entry.PATH.valid_ans_seg == None:
                print('No answer for question' + str(index))
                combined_file_list.append(None)
                continue
            else:
                answer = AudioSegment.from_wav(entry.PATH.valid_ans_seg) + 5

            clean_chunks = [
                remove_silence(
                    single_chunk) for single_chunk in audio_chunks]
            beep_duration = 0.015
            beep_chunk = plain_beep(dur=beep_duration, freq=0, vol=0.9,
                                    sps=44100, bit=16, chn=1)
            try:
                combined_sounds = clean_chunks[0] + \
                    beep_chunk + answer + beep_chunk + clean_chunks[2]
            except:
                combined_sounds = clean_chunks[0] + beep_chunk + answer

            combined_name = self.qa_combine_folder + 'QA_T_{0}.wav'.format(
                str(index).zfill(3))

            combined_sounds.export(combined_name, format="wav")
            combined_file_list.append(combined_name)

        self.qa_info[('PATH', 'file_qa_combined')] = combined_file_list
        self.qa_info[
            ('ANSWER_INFO', 'ans_combined_loc')] = self.qa_info[
                ('PATH', 'file_qa_combined')]
        self.qa_info[
            ('ANSWER_INFO', 'ans_valid_clip_loc')] = self.qa_info[
                ('PATH', 'valid_ans_seg')]
        self.qa_info[
            ('ANSWER_INFO', 'ans_full_loc')] = self.qa_file
        if update_pkl:
            self.qa_info.to_pickle(self.qa_pkl)

        return self

    def question_to_chunk(self):
        # !!! Only have to run once !!!

        from jxu.audio.audiosignal import mp3_to_wav, minimal_audio_to_chunk

        all_df_loc = self.meta_folder + \
            'qa_info/all_unshattered_beep_df_randomized.pkl'

        all_question_df = pd.read_pickle(all_df_loc)
        all_question_df[('PATH', 'file_seg_ori')] = all_question_df['PATH']['file_root_ori'].replace(
                {'/home/jxu/File/Data/NIBS/Stage_one/Audio/Database/Unshattered/audio/':
                    '/home/jxu/File/Data/NIBS/Stage_one/EEG/Exp/Audio/audio_file/'},
                regex=True)

        for index, entry in all_question_df.iterrows():
            print(index)
            audio_name = entry.PATH.file_seg_ori[:-4]
            mp3_to_wav(audio_name, sps=44100, channel=1, bit=16,
                       std_suffix='_44100_std')
            chunk_folder_path = '/'.join(audio_name.split('/')[:-1]) + '/Chunk/'

            audio_chunks = minimal_audio_to_chunk(
                audio_name + '_44100_std.wav',
                chunk_folder_path=chunk_folder_path,
                save=True)

            assert len(audio_chunks) in [2, 3], str(index) + \
                ': #Chunk should be either 2 or 3, this is' + str(
                    len(audio_chunks))

            assert ~entry.SENTENCE_INFO.last_word_flag == len(audio_chunks) - 4, '' + \
                'last word flag is in consistant with #chunk'

        return self


    def google_stt(self, audio_loc):
        import io
        # insert the function sample_recog from the example code
        # input should be the path to the combined qa audio, indivisual.
        # output should be text string
        """
        Performs synchronous speech recognition on an audio file

        Args:
        storage_uri URI for audio file in Cloud Storage, e.g. gs://[BUCKET]/[FILE]
        """

        client = speech_v1p1beta1.SpeechClient()

        # storage_uri = 'gs://cloud-samples-data/speech/brooklyn_bridge.mp3'

        # The language of the supplied audio
        language_code = "de-DE"
        # Sample rate in Hertz of the audio data sent
        sample_rate_hertz = 44100

        # Encoding of audio data sent. This sample sets this explicitly.
        # This field is optional for FLAC and WAV audio formats.
        encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16
        config = {
            "language_code": language_code,
            "sample_rate_hertz": sample_rate_hertz,
            "encoding": encoding,
        }

        with io.open(audio_loc, "rb") as f:
            content = f.read()
        audio = {"content": content}
        # audio = {"uri": storage_uri}

        response = client.recognize(config, audio)

        stt_list = []
        stt_prob = []
        for result in response.results:
            # First alternative is the most probable result
            alternative = result.alternatives[0]
            stt_list.append(alternative.transcript)
            stt_prob.append(alternative.confidence)

        return stt_list, stt_prob

    def answer_to_text(self):

        # use google speech to text to covert combined audios

        # add attribute ANSWER_INFO.ans_sen_text
        assert hasattr(self.qa_info.ANSWER_INFO, 'ans_combined_loc'), 'Run ' +\
            'qa_combine() first!'

        sen_text = []
        for index, entry in self.qa_info.iterrows():
            combined_audio_loc = entry.ANSWER_INFO.ans_combined_loc
            if combined_audio_loc is None:
                sen_text.append('')
            else:
                sen_text.append(self.google_stt(combined_audio_loc))

        self.qa_info[('ANSWER_INFO', 'ans_sen_text')] = sen_text
        self._save_pkl(self.qa_info, self.qa_pkl)

        return self



    def text_removal(self):

        # remove excessive text
        # add attribute ANSWER_INFO.vocab_text
        assert hasattr(self.qa_info.ANSWER_INFO, 'ans_sen_text'), 'Run ' + \
            'answer_to_text() first!'

        vocab_text = []
        for index, entry in self.qa_info.iterrows():
            stt_text = entry.ANSWER_INFO.ans_sen_text
            if stt_text == '':
                vocab_text.append('')
            else:
                correct_sen_text = entry.SENTENCE_INFO.sen_content
                correct_vocab = entry.SENTENCE_INFO.beeped_word
                ans_vocab = self.removal(
                    correct_sen_text, correct_vocab, stt_text)

                if len(ans_vocab) != 1:
                    print('list of vocab answer: ' + ' '.join(ans_vocab))
                    input_flag = input('Manually input? 1/0')
                    if input_flag:
                        vocab_text = input('Input the answer')
                vocab_text.append(vocab_text)
        self.qa_info[('ANSWER_INFO', 'ans_vocab_text')] = vocab_text
        self._save_pkl(self.qa_info, self.qa_pkl)


        return self


    def removal(self, answer_sen, answer_vocab, answer_stt):

        import re

        answer_sen_list = re.sub("[^\w]", " ", answer_sen).split()
        answer_stt_list = re.sub("[^\w]", " ", answer_stt).split()
        start_ind = answer_sen_list.index(answer_vocab)
        end_ind = - answer_sen_list[::-1].index(answer_vocab) - 1

        return answer_sen[start_ind:end_ind]


    def text_to_audio():

        # Use google text to speech
        # Answer text -> Audio, i.e., standard audio. Rate should be 0.9, TTS.

        # add attribute ANSWER_INFO.ans_tts_audio_loc

        pass

    def audio_duration(self):

        # add attribute self.qa_info.ANSWER_INFO.ans_duration&
        # .ans_tts_duration
        from jxu.audio.audiosignal import detect_leading_silence as dls
        from pydub import AudioSegment

        assert hasattr(self.qa_info.ANSWER_INFO, 'ans_google_loc'), 'Run ' + \
            'text_to_audio() first!'

        google_dur_list = []
        for index, entry in self.qa_info.iterrows():
            if entry.ANSWER_INFO.ans_google_loc is None:
                google_dur_list.append(None)
                continue
            google_seg = AudioSegment.from_file(
                entry.ANSWER_INFO.ans_google_loc)
            start_trim = dls(google_seg, silence_threshold=-80.0, chunk_size=1)
            end_trim = dls(google_seg.reverse(), silence_threshold=-80.0,
                           chunk_size=1)
            google_dur_list.append(
                len(google_seg[start_trim:-end_trim-1])/1000)

        self.qa_info[
            ('ANSWER_INFO', 'ans_duration')] = self.valid_seg_marker[:, -1]
        self.qa_info[('ANSWER_INFO', 'ans_tts_duration')] = google_dur_list
        self._save_pkl(self.qa_info, self.qa_pkl)

        return self

    def answer_score(self):

        # measure the similarity between given answer and subjects' answer
        # or leverage BERT to score the answer.(i.e., probability)
        pass


    def _save_pkl(self, obj, path):

        if isinstance(obj, pd.DataFrame):
            obj.to_pickle(path)
        else:
            with open(path, 'wb') as f_save:
                pickle.dump(obj, f_save)

        return self

    def _load_pkl(self, path):

        with open(path, 'rb') as f_load:
            obj = pickle.load(f_load)

        return obj

