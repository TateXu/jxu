from jxu.data.loader import *
import numpy as np
from numpy.fft import fft, fftfreq
from scipy import signal
import matplotlib.pyplot as plt
from itertools import product

import mne
from mne.time_frequency.tfr import morlet
from mne.viz import plot_filter, plot_ideal_filter
import warnings
import pickle
from jxu.viz.utils import *
import pdb

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
label_dict = [*event_dict.keys()]
event_dict_expand = {}
for i, keys in enumerate(label_dict):
    event_dict_expand[keys + '_start'] = event_dict[keys][0]
    event_dict_expand[keys + '_end'] = event_dict[keys][1]


def trigger_detector(raw, event_dict=event_dict,
                     event_dict_expand=event_dict_expand,
                     nr_events_predefined=nr_events_predefined):

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


def plot_joint_psd(raw_file, freq=[(0.0, 70.0)], nfft=1000, save=False,
                   tmin=None, tmax=None,
                   fig_name='test.pdf', fig_unit_height=3, picks=None,
                   fig_unit_width=3, fig_height=None, fig_width=None):

    nr_band = len(freq)
    if fig_height is None:
        fig_height = 2 * fig_unit_height

    if fig_width is None:
        fig_width = nr_band * fig_unit_width

    fig = plt.figure(figsize=(fig_width, fig_height))
    axes = []
    gs0 = gs.GridSpec(nrows=2, ncols=nr_band)
    for ax_col in range(nr_band):
        axes.append(fig.add_subplot(gs0[0, ax_col]))
    axes.append(gs0[1, :])
    axes = np.asarray(axes)

    ax3 = plt.subplot(212)

    raw_file.plot_psd_topomap(axes=axes[:-1], tmin=tmin, tmax=tmax,
                              bands=freq, show=False)

    raw_file.plot_psd(average=False, ax=ax3, spatial_colors=True,
                      picks=picks, estimate='amplitude', tmin=tmin, tmax=tmax,
                      fmin=freq[0][0], fmax=freq[-2][1], show=False)


    if save:
        fig.savefig(fig_name)
        print('Figure saved!')



def plot_joint_t_freq(raw_file, channel='C3', n_perseg=1000, nfft=1000,
                      n_overlap=None, colorbar=True, save=False,
                      tmin=None, tmax=None, t_step=None, fmin=0.0, fmax=70.0,
                      num_t_step=None, fig_name='test.pdf'):

    if not isinstance(raw_file, list):
        raw_file = [raw_file]
    if not isinstance(channel, list):
        channel = [channel]

    if n_overlap is None:
        n_overlap = n_perseg // 2.
    if tmin is None:
        tmin = raw_file[0].tmin
    if tmax is None:
        tmax = raw_file[0].tmax
    if num_t_step is None:
        num_t_step = 11
    if t_step is None:
        t_step = (tmax - tmin) / num_t_step


    fs = raw_file[0].info['sfreq']
    down_fs = int(fs / 5)

    t_resolution = n_overlap / fs
    f_resolution = fs / nfft

    min_t_samp = int(tmin / t_resolution)
    max_t_samp = int(tmax / t_resolution)
    min_f_samp = int(fmin / f_resolution)
    max_f_samp = int(fmax / f_resolution)

    nr_raw = len(raw_file) * len(channel)
    xlabel, ylabel, title, scale, rotation, axis_lim = [], [], [], [], [], []
    xlabel += [['', 'Time/s'], ['', 'Time/s']] * nr_raw
    ylabel += [['Amplitude', 'Frequency/Hz'], ['', 'Amplitude/uV']] * nr_raw
    title += [['Spectrum', 'Short Time Fourier Transform'], ['Electrodes Location', 'Amplitude of EEG signal']] * nr_raw

    scale += [[[4.3, 2], [1, 1]], [[1, 1], [1, 1]]] * nr_raw
    rotation += [[90, 0], [0, 0]] * nr_raw
    axis_lim += [[(0, 70, 0, 70), 0], [0, 0]] * nr_raw
    row_ratios = []
    row_ratios += [2, 1] * nr_raw

    xticks, xticklabels = [], []
    xticks += [[None, np.linspace(tmin, tmax, num_t_step)],
               [None, np.linspace(tmin, tmax, num_t_step)]] * nr_raw

    xticklabels += [[None, [str(i) for i in np.linspace(tmin, tmax, num_t_step)]],
                    [None, [str(i) for i in np.linspace(tmin, tmax, num_t_step)]]] * nr_raw

    xticks = np.asarray(xticks)
    xticklabels = np.asarray(xticklabels)
    fig, axes, gs = fig_init(nr_row=2 * nr_raw, nr_col=2,
                             row_ratios=row_ratios, col_ratios=[1, 3],
                             fig_unit_height=3, fig_unit_width=3,
                             xlabel=np.asarray(xlabel),
                             ylabel=np.asarray(ylabel),
                             title=np.asarray(title),
                             scale=np.asarray(scale),
                             rotation=np.asarray(rotation),
                             axis_lim=np.asarray(axis_lim))

    for ind_raw, (sig_raw, sig_chn) in enumerate(product(raw_file, channel)):
        print(ind_raw)
        print(sig_raw)
        print(sig_chn)
        picks = [sig_chn]

        pick_eeg = sig_raw.get_data(picks=picks) * 1e6

        stft_f, stft_t, Zxx = signal.stft(pick_eeg, fs,
                                          nperseg=n_perseg, noverlap=n_overlap)
        stft_fig = axes[ind_raw * 2, 1].pcolormesh(
            stft_t[min_t_samp: max_t_samp + 1],
            stft_f[min_f_samp: max_f_samp + 1],
            np.abs(Zxx[0][0][min_f_samp: max_f_samp + 1,
                             min_t_samp: max_t_samp + 1]),
            vmin=0, vmax=6)
        if colorbar:
            plt.colorbar(mappable=stft_fig, ax=axes[ind_raw * 2, 1],
                         orientation='vertical',
                         use_gridspec=True)
        sig_raw.plot_psd(average=False, ax=axes[ind_raw * 2, 0],
                         spatial_colors=True, picks=picks,
                         estimate='amplitude',
                         tmin=tmin, tmax=tmax,
                         fmin=fmin, fmax=fmax, show=False)
        sig_raw.info['bads'] = picks
        sig_raw.plot_sensors(show=False, show_names=False,kind='select',
                             axes=axes[ind_raw * 2 + 1, 0],
                             title=sig_chn)
        raw_ts = sig_raw.copy().resample(sfreq=down_fs).get_data(picks=picks) * 1e6

        axes[ind_raw * 2, 0].set_xlim([fmin, fmax])
        axes[ind_raw * 2 + 1, 1].plot(np.arange(tmin * down_fs, tmax * down_fs + 1, 1),
            raw_ts[0][0][int(tmin * down_fs): int(tmax * down_fs) + 1])

        axes[ind_raw * 2 + 1, 0].set_title('Channel: ' + sig_chn)

        axes[ind_raw * 2 + 1, 1].set_xlim([tmin * down_fs, tmax * down_fs + 1])
        axes[ind_raw * 2 + 1, 1].set_xticks(down_fs * xticks[ind_raw * 2 + 1, 1])
        axes[ind_raw * 2 + 1, 1].set_xticklabels(xticklabels[ind_raw * 2 + 1, 1])

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    if save:
        fig.savefig(fig_name)
        print('Figure saved!')
"""
for bandpass in filters:
    fmin, fmax = bandpass
    # filter data
    raw_f = raw.copy().filter(fmin, fmax, method='fir',
                              picks=picks, verbose=False, fir_design='firwin')

"""
load = True  # True

path = '/Users/xujiachen/File/Data/NIBS/Stage_one/ZWS/ZWS_SESSION_1/'
if load:
    raw = eeg_loader(subject=0, session=1)

    filters = [(0.1, 90)]

    trigger_detector(raw)

    fmin, fmax = filters[0]
    # filter data

    raw.set_channel_types({'Audio': 'stim', 'tACS': 'stim'})
    pdb.set_trace()
    print('Start filtering!')
    raw_f = raw.copy().filter(fmin, fmax, method='fir', verbose=False, fir_design='firwin')
    raw_f.notch_filter(freqs=np.arange(50, 249, 50), picks='eeg')

    raw_f.set_channel_types({'EOG151': 'eog', 'EOG152': 'eog'})
    print('Removing common average!')

    raw_ca = raw_f.copy().set_eeg_reference(ref_channels='average')

    # ori_data_eog = raw_f.get_data()[-2:,:]
    # ori_data = raw_f.pick(picks='eeg').get_data()
    # ca_data_eog = raw_ca.get_data()[-2:,:]
    # ca_data = raw_ca.copy().pick(picks='eeg').get_data()

    data_run_0_1 = raw_ca[:, : 2446629]
    data_run_2 = raw_ca[:, 2578697:3503294]
    data_run_3 = raw_ca[:, 3552652:]

    ts_run_0_1 = data_run_0_1[1]
    ts_run_2 = data_run_2[1]
    ts_run_3 = data_run_3[1]

    raw_run_0_1 = raw_ca.copy().crop(
        tmin=ts_run_0_1[0], tmax=ts_run_0_1[-1], include_tmax=True)
    raw_run_2 = raw_ca.copy().crop(
        tmin=ts_run_2[0], tmax=ts_run_2[-1], include_tmax=True)
    raw_run_3 = raw_ca.copy().crop(
        tmin=ts_run_3[0], tmax=ts_run_3[-1], include_tmax=True)

    raw_clean = raw_run_0_1.copy()
    raw_clean.append([raw_run_2, raw_run_3])
    pdb.set_trace()

    trigger_detector(raw_clean)

    epoch_list = {'Cali_trial_start': [30, 20],
                  'RS_intro_start': [190, 8],
                  'QA_trial_start': [30, 160]}

    X = {}
    events_clean, event_clean_id = mne.events_from_annotations(raw_clean)

    montage = mne.channels.read_montage("standard_1005")
    raw_clean.rename_channels({'O9': 'I1', 'O10': 'I2'})
    raw_clean.set_montage(montage)
    raw_clean.rename_channels({'I1': 'O9', 'I2': 'O10'})
    raw_clean.info['bads'] = ['TPP9h', 'P7', 'AFp1']
    # raw_clean.set_channel_types({'Audio': 'stim', 'tACS': 'stim'})
    raw_clean.set_channel_types({'TP7': 'stim', 'TP8': 'stim',
                                 'TTP7h': 'stim', 'TTP8h': 'stim',
                                 'TPP7h': 'stim', 'TPP8h': 'stim',
                                 'T8': 'stim'})
    # raw_clean.set_channel_types({'EOG151': 'eog', 'EOG152': 'eog'})

    for ind, (key, val) in enumerate(epoch_list.items()):
        epochs = mne.Epochs(raw_clean, events_clean,
                            event_id=[event_dict_expand[key]],
                            tmin=0, tmax=val[0], baseline=None, proj=False,
                            preload=True, verbose=False, on_missing='ignore')
        if ind == 0:
            X['Calibration'] = [epochs[:10], epochs[10:]]
        elif ind == 1:
            X['RS'] = {}
            rs_open_start = events_clean[np.where(
                events_clean[:, 2] == 32)[0], :]
            rs_close_start = events_clean[np.where(
                events_clean[:, 2] == 34)[0], :]

            default_rs_order = ['open', 'close']
            rs_order = []
            [rs_order.extend(default_rs_order[::ind]) for ind in np.sign((rs_close_start - rs_open_start)[:, 0])]

            for state in default_rs_order:
                X['RS'][state] = epochs[np.asarray(rs_order) == state]
        elif ind == 2:
            X['QA'] = [epochs[:40], epochs[40: 80],
                       epochs[80: 120], epochs[120: 160]]

    with open(path + 'data.pkl', 'wb') as f:
        pickle.dump(X, f)
    print('saved!')
pdb.set_trace()
with open(path + 'data.pkl', 'rb') as f:
    X = pickle.load(f)

"""
full_chn = X['RS']['open'][0].info['ch_names']
exclude = ['Audio', 'tACS', 'EOG151', 'EOG152', 'TP7', 'TP8', 'TTP7h', 'TTP8h',
           'TPP7h', 'TPP8h', 'TPP9h', 'P7', 'AFP1']

picks = list(set(full_chn) - set(exclude))
"""

picks = ['Pz', 'Fz']

pdb.set_trace()
"""
for nr_run in range(4):
    plot_joint_t_freq([X['RS']['open'][nr_run], X['RS']['close'][nr_run]], channel=picks,
                      colorbar=False, fmin=0.0, save=True, fmax=70.0,
                      tmin=30.0, tmax=180.0,
                      fig_name=path + 'RS_open_vs_close_Run_' + str(nr_run) + '.pdf')

for state in ['open', 'close']:
    for nr_run in range(4):
        plot_joint_psd(
            X['RS'][state][nr_run],
            freq=[(1.0, 4.0, 'Delta (1-4)'),
                  (4.0, 8.0, 'Theta (4-8)'),
                  (8.0, 14.0, 'Alpha (8-14)'),
                  (14.0, 20.0, 'Low Beta (14-20)'),
                  (20.0, 30.0, 'High Beta (20-30)'),
                  (30.0, 70.0, 'Gamma (30-70))'),
                  (8.9, 9.1, 'Peak Frequency (9Hz)')],
            tmin=30.0, tmax=180.0, picks='eeg',
            fig_name=path + 'RS_' + state + '_run_' + str(nr_run) + '.pdf',
            fig_unit_height=3, save=True,
            fig_unit_width=3, fig_height=None, fig_width=None)

"""

pdb.set_trace()
"""

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
fs_ = 10e3
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs_ / 2
time = np.arange(N) / float(fs_)
mod = 500*np.cos(2*np.pi*0.25*time)
carrier = amp * np.sin(2*np.pi*3e3*time + mod)
noise = np.random.normal(scale=np.sqrt(noise_power),
                         size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noise
f_s, t_s, Zxx_s = signal.stft(x, fs_, nperseg=1000)
axes[0, 0].pcolormesh(t_s, f_s, np.abs(Zxx), vmin=0, vmax=amp)

pdb.set_trace()


def plot_joint_t_freq(raw, channel='C3', tmin=None, tmax=None, fmin=None, fmax=None):

    from jxu.viz.utils import fig_init
    from matplotlib import pyplot as plt
    fig, axes = fig_init(nr_row=2, nr_col=2,
                         row_ratios=[2, 1], col_ratios=[2, 1])

    if tmin is None:
        tmin = 0
    if fmin is None:
        fmin = 0

    if tmax is None:
        tmax = -1
    if fmax is None:
        fmax = 0

    n_perseg = 1000
    n_overlap = n_perseg // 2.
    fs = raw.info['sfreq']
    t_resolution = n_overlap / fs


    f_resolution = fs / nfft


    min_t_samp = int(tmin / t_resolution)
    max_t_samp = int(tmax / t_resolution)
    min_f_samp = int(fmin / f_resolution)
    max_f_samp = int(fmax / f_resolution)


    signal = raw.get_data(picks=[channel])


    f, t, Zxx = signal.stft(signal, fs, nperseg=n_perseg, noverlap=n_overlap)
    stft_fig = axes[0, 0].pcolormesh(t[min_t_samp: max_t_samp + 1], f[min_f_samp: max_f_samp + 1], np.abs(Zxx), vmin=0, vmax=amp)

    stft_fig = axes[0, 0].pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)

    raw.plot_psd(average=False, spatial_colors=True, picks=picks, fmax=70, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
    plt.colorbar(mappable=stft_fig, ax=axes[0, 0], use_gridspec=True,
                 orientation='vertical')
    ax, aux_ax = setup_axes(fig, gs0, 45)



    # MNE is in V, rescale to have uV
    # if self.resample is not None:
    #        epochs = epochs.resample(self.resample)
    # datamat = 1e6 * epochs.get_data()


events[np.where(events[:,2]==12)[0], :]
cali_trial_start = events[np.where(events[:,2]==12)[0], :]
cali_trial_end = events[np.where(events[:,2]==13)[0], :]

valid_cali_trial_ind = (cali_trial_end - cali_trial_start)[:,0] > 1000

qa_trial_start = events[np.where(events[:,2]==42)[0], :]
qa_trial_end = events[np.where(events[:,2]==43)[0], :]


block_start = events[np.where(events[:,2]==6)[0], :]
block_end = events[np.where(events[:,2]==7)[0], :]

rs_inrto_start = events[np.where(events[:,2]==30)[0], :]
rs_inrto_end = events[np.where(events[:,2]==31)[0], :]
rs_open_start = events[np.where(events[:,2]==32)[0], :]
rs_open_end = events[np.where(events[:,2]==33)[0], :]
rs_close_start = events[np.where(events[:,2]==34)[0], :]
rs_close_end = events[np.where(events[:,2]==35)[0], :]
"""





# fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'],first_samp=raw.first_samp, event_id=new_event_dict)


"""
if colorbar:
    xlabel, ylabel, title, scale, rotation, axis_lim = [], [], [], [], [], []
    xlabel += [['', 'Time/s', ''], ['', 'Time/s', '']] * nr_raw
    ylabel += [['Amplitude', 'Frequency/Hz', 'Amplitude'], ['', 'Amplitude/uV', '']] * nr_raw
    title += [['Spectrum', 'Short Time Fourier Transform', 'Cbar'], ['Electrodes Location', 'Amplitude of EEG signal', '']] * nr_raw

    scale += [[[4.3, 2], [1, 1], [0.2, 1]], [[1, 1], [1, 1], [1, 1]]] * nr_raw
    rotation += [[90, 0, 90], [0, 0, 0]] * nr_raw
    axis_lim += [[(0, 70, 0, 70), 0, (0, 70, 0, 70)], [0, 0, 0]] * nr_raw
    row_ratios = []
    row_ratios += [2, 1] * nr_raw

    xticks, xticklabels = [], []
    xticks += [[None, np.linspace(tmin, tmax, num_t_step), None],
               [None, np.linspace(tmin, tmax, num_t_step), None]] * nr_raw

    xticklabels += [[None, [str(i) for i in np.linspace(tmin, tmax, num_t_step)], None],
                    [None, [str(i) for i in np.linspace(tmin, tmax, num_t_step)], None]] * nr_raw

    xticks = np.asarray(xticks)
    xticklabels = np.asarray(xticklabels)
    fig, axes = fig_init(nr_row=2 * nr_raw, nr_col=3,
                         row_ratios=row_ratios, col_ratios=[1, 3, 1],
                         fig_unit_height=3, fig_unit_width=3,
                         xlabel=np.asarray(xlabel),
                         ylabel=np.asarray(ylabel),
                         title=np.asarray(title),
                         scale=np.asarray(scale),
                         rotation=np.asarray(rotation),
                         axis_lim=np.asarray(axis_lim))
"""