from matplotlib import pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes


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
             axis_lim=None, scale=None):

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
