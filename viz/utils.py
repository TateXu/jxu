from matplotlib import pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes

from jxu.data.utils import *
from jxu.data.loader import *
import numpy as np
from numpy.fft import fft, fftfreq
from scipy import signal
import matplotlib.pyplot as plt
from itertools import product
import pdb

import mne

import warnings
import pickle

# mpl.rc('font', family='serif', serif='Times New Roman')


def varname(var):
    print(globals().items())
    return [tpl[0] for tpl in filter(lambda x: var is x[1], globals().items())]


def setup_axes(fig, rect, rotation, scale=[1, 1], axis_lim=(0, 20, 0, 20)):
    tr_rot = Affine2D().scale(scale[0], scale[1]).rotate_deg(rotation)
    grid_helper = floating_axes.GridHelperCurveLinear(tr_rot,
        extremes=axis_lim)
    ax = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax)
    aux_ax = ax.get_aux_axes(tr_rot)
    return ax, aux_ax


def fig_init(nr_row=1, nr_col=1, row_ratios=None, col_ratios=None,
             fig_unit_height=2, fig_unit_width=2, xticks=None,
             yticks=None, xticklabels=None, yticklabels=None,
             xlabel=None, ylabel=None, title=None, aux=False,
             tick_fontsize=10, ticklabel_fontsize=10,
             label_fontsize=14, title_fontsize=16, rotation=None,
             axis_lim=None, scale=None, trigger_annot=None):

    """
    Example usage:
fig, axes = fig_init(nr_row=2, nr_col=2, row_ratios=[1, 1], col_ratios=[2, 1],
                     xticks=np.asarray([[range(0, 10, 1), range(0, 10, 2)],
                                        [range(0, 20, 1), range(0, 20, 2)]]),
                     yticks=np.asarray([[range(0, 10, 1), range(0, 10, 2)],
                                        [range(0, 20, 1), range(0, 20, 2)]]),
                     xticklabels=np.asarray([[[str(i) for i in range(0, 10, 1)],
                                              [str(i) for i in range(0, 10, 2)]],
                                             [[str(i) for i in range(0, 20, 1)],
                                              [str(i) for i in range(0, 20, 2)]]]),
                     yticklabels=np.asarray([[[str(i) for i in range(0, 10, 1)],
                                              [str(i) for i in range(0, 10, 2)]],
                                             [[str(i) for i in range(0, 20, 1)],
                                              [str(i) for i in range(0, 20, 2)]]]),
                     xlabel=np.asarray([['ax', 'bx'], ['cx', 'dx']]),
                     ylabel=np.asarray([['ay', 'by'], ['cy', 'dy']]),
                     title=np.asarray([['a', 'b'], ['c', 'd']]),
                     rotation=np.asarray([[180, 0],[0, 0]]),
                     axis_lim=np.asarray([[(0, 5000, 0, 1000), 0],[0, 0]]))
    """

    gs0 = gs.GridSpec(nrows=nr_row, ncols=nr_col, height_ratios=row_ratios,
                      width_ratios=col_ratios)
    if row_ratios is not None and len(row_ratios) != nr_row:
        raise ValueError('Invalid row_ratio! Must be same length with nr')

    if col_ratios is not None and len(col_ratios) != nr_col:
        raise ValueError('Invalid col_ratio! Must be same length with nr')

    if row_ratios is not None and isinstance(row_ratios, list):
        fig_height = sum(row_ratios) * fig_unit_height
    elif row_ratios is None:
        fig_height = nr_row * fig_unit_height

    if col_ratios is not None and isinstance(col_ratios, list):
        fig_width = sum(col_ratios) * fig_unit_width
    elif col_ratios is None:
        fig_width = nr_col * fig_unit_width

    aux_list = [xticks, yticks, xticklabels, yticklabels,
                xlabel, ylabel, title]
    for ind, item in enumerate(aux_list):
        if item is not None:
            aux = True
            if not isinstance(item, np.ndarray):
                raise ValueError('Aux input must be numpy array!')

    fig = plt.figure(figsize=(fig_width, fig_height))

    axes = []

    for ax_row in range(nr_row):
        axes_sub = []
        for ax_col in range(nr_col):

            if rotation is not None or scale is not None:

                if rotation[ax_row, ax_col] == 0 and np.any(scale[ax_row, ax_col] == [1,1]):
                    wrapped_axe = fig.add_subplot(gs0[ax_row, ax_col])
                else:
                    wrapped_axe = setup_axes(fig, gs0[ax_row, ax_col],
                                             scale=scale[ax_row, ax_col],
                                             rotation=rotation[ax_row, ax_col],
                                             axis_lim=axis_lim[ax_row, ax_col])

            else:
                wrapped_axe = fig.add_subplot(gs0[ax_row, ax_col])

            if not isinstance(wrapped_axe, tuple):
                label_axe, plt_axe = wrapped_axe, wrapped_axe
            else:
                label_axe, plt_axe = wrapped_axe

            if aux:
                if xticks is not None:
                    if xticks[ax_row, ax_col] is not None:
                        label_axe.set_xticks(xticks[ax_row, ax_col])
                if yticks is not None:
                    if yticks[ax_row, ax_col] is not None:
                        label_axe.set_yticks(yticks[ax_row, ax_col])
                if xticklabels is not None:
                    if xticklabels[ax_row, ax_col] is not None:
                        label_axe.set_xticklabels(xticklabels[ax_row, ax_col],
                                                  fontsize=ticklabel_fontsize)
                if yticklabels is not None:
                    if yticklabels[ax_row, ax_col] is not None:
                        label_axe.set_yticklabels(yticklabels[ax_row, ax_col],
                                                  fontsize=ticklabel_fontsize)
                if xlabel is not None:
                    label_axe.set_xlabel(xlabel[ax_row, ax_col],
                                         fontsize=label_fontsize)
                if ylabel is not None:
                    label_axe.set_ylabel(ylabel[ax_row, ax_col],
                                         fontsize=label_fontsize)
                if title is not None:
                    label_axe.set_title(title[ax_row, ax_col],
                                        fontsize=title_fontsize)
            axes_sub.append(plt_axe)
        axes.append(axes_sub)

    fig.tight_layout()
    return fig, np.asarray(axes), gs0



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
                      n_overlap=None, colorbar=True, save=False, bg_text='',
                      tmin=None, tmax=None, t_step=None, fmin=0.0, fmax=70.0,
                      num_t_step=None, trigger_annot=None, fig_name='test.pdf'):
    nr_events_predefined, event_dict, label_dict, event_dict_expand = nibs_event_dict()

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
    ylabel += [['Amplitude/dB', 'Frequency/Hz'], ['', '']] * nr_raw
    title += [['Spectrum', 'Short Time Fourier Transform'], ['Electrodes Location', 'Amplitude of EEG signal']] * nr_raw

    scale += [[[4.3, 2], [1, 1]], [[1, 1], [1, 1]]] * nr_raw
    rotation += [[90, 0], [0, 0]] * nr_raw
    axis_lim += [[(0, 70, 0, 70), 0], [0, 0]] * nr_raw
    row_ratios = []
    row_ratios += [2, 1] * nr_raw

    xticks, xticklabels = [], []
    xticks += [[None, np.linspace(tmin, tmax, num_t_step)],
               [None, np.linspace(tmin, tmax, num_t_step)]] * nr_raw

    xticklabels += [[None, ["{:.2f}".format(i) for i in np.linspace(tmin, tmax, num_t_step)]],
                    [None, ["{:.2f}".format(i) for i in np.linspace(tmin, tmax, num_t_step)]]] * nr_raw

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

        stft_f, stft_t, Zxx = signal.stft(pick_eeg, fs, nfft=nfft,
                                          nperseg=n_perseg, noverlap=n_overlap)

        amp_part = np.abs(Zxx[0][0][min_f_samp: max_f_samp + 1,min_t_samp: max_t_samp + 1])
        stft_fig = axes[ind_raw * 2, 1].pcolormesh(
            stft_t[min_t_samp: max_t_samp + 1],
            stft_f[min_f_samp: max_f_samp + 1],
            amp_part, vmin=0, vmax=30)  # amp_part.max()
        axes[ind_raw * 2, 1].text(x=tmax-(tmax-tmin)/10, y=fmax-(fmax-fmin)/10, s=bg_text, color='r')
        if colorbar:
            plt.colorbar(mappable=stft_fig, ax=axes[ind_raw * 2, 1],
                         orientation='vertical',
                         use_gridspec=True)
        sig_raw.plot_psd(average=False, ax=axes[ind_raw * 2, 0],
                         spatial_colors=True, picks=picks,
                         estimate='amplitude',
                         tmin=tmin, tmax=tmax,
                         fmin=fmin, fmax=fmax, show=False)
        axes[ind_raw * 2, 0].set_xlim([fmin, fmax])

        sig_raw.info['bads'] = picks
        sig_raw.plot_sensors(show=False, show_names=False,kind='select',
                             axes=axes[ind_raw * 2 + 1, 0], title=None)
        raw_ts = sig_raw.copy().resample(sfreq=down_fs).get_data(picks=picks) * 1e6
        if trigger_annot is None:
            axes[ind_raw * 2 + 1, 1].plot(np.arange(tmin * down_fs, tmax * down_fs + 1, 1),
            raw_ts[0][0][int(tmin * down_fs): int(tmax * down_fs) + 1])

            axes[ind_raw * 2 + 1, 0].set_title('Channel: ' + sig_chn)

            axes[ind_raw * 2 + 1, 1].set_xlim([tmin * down_fs, tmax * down_fs + 1])
            axes[ind_raw * 2 + 1, 1].set_xticks(down_fs * xticks[ind_raw * 2 + 1, 1])
            axes[ind_raw * 2 + 1, 1].set_xticklabels(xticklabels[ind_raw * 2 + 1, 1])
        else:
            axes[ind_raw * 2 + 1, 1].set_xticks([])
            axes[ind_raw * 2 + 1, 1].set_yticks([])
            axes[ind_raw * 2 + 1, 1].set_xlabel('')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    if trigger_annot is not None:

        for ind_raw, (sig_raw, sig_chn) in enumerate(product(raw_file, channel)):
                raw_ts = sig_raw.copy().resample(sfreq=down_fs).get_data(picks=picks) * 1e6

                import matplotlib.gridspec as gs
                inner = gs.GridSpecFromSubplotSpec(2, 1, subplot_spec=axes[ind_raw * 2 + 1, 1], wspace=0, hspace=0.2, height_ratios=[1, 6], width_ratios=None)

                trigger_ax = fig.add_subplot(inner[0])
                ts_ax = fig.add_subplot(inner[1])

                # trigger_ax = plt.Subplot(fig, inner[0])
                # ts_ax = plt.Subplot(fig, inner[1])
                # fig.add_subplot(trigger_ax)
                # fig.add_subplot(ts_ax)

                trigger_color = {'44': 'r',
                                 '45': 'r',
                                 '50': 'b',
                                 '51': 'b',
                                 '48': 'k',
                                 '49': 'k'}

                legend_dict = {'44': 'Q', '50': 'Cen.', '48': 'Rec'}
                line_list = []
                legend_list = []

                for trigger_row in trigger_annot:
                    if not str(trigger_row[1]) in [*trigger_color.keys()]:
                        continue   
                    if str(trigger_row[1]) in [*legend_dict.keys()]:
                        line = trigger_ax.arrow(trigger_row[0] / fs, 0, 0, 1, color=trigger_color[str(trigger_row[1])], width=0.03, head_length=0.03)
                        line_list.append(line)
                        legend_list.append(legend_dict[str(trigger_row[1])])
                    else:
                        trigger_ax.arrow(trigger_row[0] / fs, 1, 0, -1, color=trigger_color[str(trigger_row[1])], width=0.03, head_length=0.03)

                trigger_ax.legend(tuple(line_list), tuple(legend_list), loc='upper center', ncol=3)
                trigger_ax.set_xlim([tmin, tmax])
                trigger_ax.set_ylim([-0.2, 1.2])
                trigger_ax.set_xticks([])
                trigger_ax.set_yticks([])
                ts_ax.plot(np.arange(tmin * down_fs, tmax * down_fs + 1, 1),
                    raw_ts[0][0][int(tmin * down_fs): int(tmax * down_fs) + 1])

                axes[ind_raw * 2 + 1, 0].set_title('Channel: ' + sig_chn)

                ts_ax.set_xlim([tmin * down_fs, tmax * down_fs + 1])
                ts_ax.set_xticks(down_fs * xticks[ind_raw * 2 + 1, 1])
                ts_ax.set_xticklabels(xticklabels[ind_raw * 2 + 1, 1])
                ts_ax.set_ylabel('Amplitude/uV')
    if save:
        print('Ready to save!')
        import time
        t1 = time.time()
        fig.savefig(fig_name)
        print('Figure saved! Time: ' + str(time.time() - t1))
    pdb.set_trace()

