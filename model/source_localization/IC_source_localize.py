# -*- coding: utf-8 -*-
import os.path as op
import numpy as np

import scipy
import mne
from mne.datasets import eegbci
from mne.datasets import fetch_fsaverage

from jxu.data.eeg_process import NIBSEEG
import warnings

def spatial_pattern_to_source(IC=None, IC_label=None, hemi='split',
                              weights=None, separate_plot=True,
                              save_plot=False, prefix='Cluster',
                              picks=None, tuning_paras=None):

    # separate_plot= True -> each IC one fig, False: ,merged plot for all IC
    # based on the weights
    if hemi == 'split':
        brain_fig_size = (1600, 800)
    else:
        brain_fig_size = (800, 800)

    if not separate_plot and weights is None:
        warnings.warn('Plot the merged source but no weights specified, assuming equal weight')
        weights = np.ones((IC.shape[1], ))

    # Download fsaverage files
    fs_dir = fetch_fsaverage(verbose=True)
    subjects_dir = op.dirname(fs_dir)

    # The files live in:
    subject = 'fsaverage'
    trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
    src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
    bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

    ##############################################################################
    # Load the data
    # ^^^^^^^^^^^^^

    nibs_obj = NIBSEEG()
    nibs_obj.raw_load()
    raw = nibs_obj.raw_data[1].crop(tmin=100.0, tmax=110.0)

    montage = nibs_obj.get_montage()
    raw.set_montage(montage)
    raw.set_eeg_reference(projection=True)  # needed for inverse modeling
    # Check that the locations of EEG electrodes is correct with respect to MRI
    mne.viz.plot_alignment(
        raw.info, src=src, eeg=['original', 'projected'], trans=trans,
        show_axes=True, mri_fiducials=True, dig='fiducials')

    ##############################################################################
    # Setup source space and compute forward
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    fwd = mne.make_forward_solution(raw.info, trans=trans, src=src,
                                    bem=bem, eeg=True, mindist=5.0, n_jobs=1)

    print(fwd)
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id=[44], baseline=None, preload=True)

    if picks is not None:
        epochs.pick(picks='eeg').drop_channels(picks)
    evoked = epochs['44'].average()  # trigger 1 in auditory/left

    if IC is None:
        warnings.warn('No input IC data. Using default data')
        IC = raw.get_data(picks='eeg')[:, 0]
        IC = np.asarray([IC])

    # cov = IC.T.dot(IC)
    cov = mne.compute_covariance(epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=True)
    cov.update(data=np.eye(cov.data.shape[0]))

    if tuning_paras is None:
        para_suffix = 'default'
        inv = mne.minimum_norm.make_inverse_operator(
            raw.info, fwd, cov, verbose=True)
        source_plot(IC, IC_label, inv, evoked, hemi, brain_fig_size,
                    prefix, subjects_dir, weights, save_plot, separate_plot,
                    para_suffix)
    else:
        from itertools import product
        assert [*tuning_paras.keys()] == ['depth', 'loose'], 'Paras only allowed depth and loose'
        for depth, loose in product(*tuning_paras.values()):
            para_suffix = f'depth_{depth}_loose_{loose}'
            inv = mne.minimum_norm.make_inverse_operator(
                raw.info, fwd, cov, verbose=True, depth=depth, loose=loose)
            source_plot(IC=IC, IC_label=IC_label, inv=inv, evoked=evoked,
                        hemi=hemi, brain_fig_size=brain_fig_size,
                        prefix=prefix, subjects_dir=subjects_dir,
                        weights=weights, save_plot=save_plot,
                        separate_plot=separate_plot, para_suffix=para_suffix)

    # replace evoked.data into the IC data matrix, e.g.,
    # evoked.data = np.tile()

def source_plot(IC, IC_label, inv, evoked, hemi, brain_fig_size, prefix,
                subjects_dir, weights, save_plot, separate_plot, para_suffix):
    stc_list = []
    brain_image_list = []
    for id_IC, (each_IC, each_label) in enumerate(zip(IC.T, IC_label)):
        evoked.data = np.tile(each_IC, (701, 1)).T
        stc = mne.minimum_norm.apply_inverse(evoked, inv)
        stc_list.append(stc)

        if save_plot and separate_plot:
            for view in ['lateral', 'medial']:
                brain = stc.plot(subjects_dir=subjects_dir, initial_time=0.1,
                                 hemi=hemi, size=brain_fig_size,
                                 title=each_label, views=view)
                brain_image_list.append(brain)
                brain.save_image(f'{prefix}_{each_label}_{view}_{para_suffix}.jpg')

    if not separate_plot:
        len_stc = len(stc_list)
        stc_data = np.empty((stc_list[0].data.shape[0], len_stc))
        for id_stc, each_stc in enumerate(stc_list):
            stc_data[:, id_stc] = each_stc.data[:, 0]
        merged_stc = stc_list[-1]
        merged_stc.data = np.average(
            stc_data, weights=weights, axis=1).reshape((stc_data.shape[0], -1))

        for view in ['lateral', 'medial']:
            brain = merged_stc.plot(subjects_dir=subjects_dir, initial_time=0.1,
                                    hemi=hemi, size=brain_fig_size, title='merged_label',
                                    views=view)
            brain_image_list.append(brain)
            if save_plot:
                brain.save_image(f'{prefix}_{view}_{para_suffix}.jpg')



























