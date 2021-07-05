import os
import pdb
import copy
import yaml
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as signal
from scipy import linalg

from mne.decoding import CSP
from mne import EvokedArray


from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm.classes import LinearSVC as SVM
from sklearn.cross_decomposition import CCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSCanonical
from sklearn.naive_bayes import MultinomialNB as NB

class LogVariance(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform"""
        assert X.ndim == 3
        return np.log(np.var(X, -1))


class FM(BaseEstimator, TransformerMixin):

    def __init__(self, freq=128):
        '''instantaneous frequencies require a sampling frequency to be properly
        scaled,
        which is helpful for some algorithms. This assumes 128 if not told
        otherwise.

        '''
        self.freq = freq

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform. """
        xphase = np.unwrap(np.angle(signal.hilbert(X, axis=-1)))
        return np.median(self.freq * np.diff(xphase, axis=-1) / (2 * np.pi),
                         axis=-1)


class Feat_convertor(BaseEstimator, TransformerMixin):

    def __init__(self, feat=['IA', 'IF'], fs=None):
        '''instantaneous frequencies require a sampling frequency to be properly
        scaled,
        which is helpful for some algorithms. This assumes 128 if not told
        otherwise.

        '''
        self.feat = feat
        self.fs = fs

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform. """

        exp_nr, elec_nr, t_len = X.shape
        X_n = np.array([], dtype=np.float64).reshape(exp_nr, 0, t_len - 1)
        for i in range(len(self.feat)):
            X_n = np.concatenate((X_n, self._base_tf(X, self.feat[i])), axis=1)

        return X_n

    def _base_tf(self, X, method, fs=None):

        if self.fs is None:
            # Load metadata file
            with open('temp_metadata.yml') as infile:
                metadata = yaml.load(infile)
            self.fs = metadata['info']['sfreq']

        # Load finish & Delete buffer metadata file
        anl_sig = signal.hilbert(X, axis=-1)
        IP = np.unwrap(np.angle(anl_sig))
        all_feat = {'IA': np.abs(anl_sig),
                    'IF': self.fs * np.diff(IP, axis=-1) / (2.0 * np.pi),
                    'IP': IP,
                    'RT': X,
                    'Analytic': anl_sig,
                    'RE': np.real(anl_sig),
                    'IM': np.imag(anl_sig)}

        if method is 'IF':
            return all_feat[method]
        else:
            return all_feat[method][:, :, 1:]

    #  def get_freq():


class CPLS(BaseEstimator, TransformerMixin):

    def __init__(self, feat=['IA', 'IF']):
        '''instantaneous frequencies require a sampling frequency to be properly
        scaled,
        which is helpful for some algorithms. This assumes 128 if not told
        otherwise.

        '''
        self.feat = feat
        self.fs = 250

    def fit(self, X, y):
        """fit."""
        return self

    def fit_transform(self, X, y=None):
        """transform. """
        pdb.set_trace()
        return PLSCanonical().fit(X, y).x_rotations_

    def transform(self, X, y=None):
        """transform. """
        n_components = 2
        ind_0 = np.where(y == 0)
        ind_1 = np.where(y == 1)

        X_clf_0 = X[ind_0, :]
        X_clf_1 = X[ind_1, :]
        y_clf_0 = y[ind_0]
        y_clf_1 = y[ind_1]
        rotation_0 = PLSCanonical().fit(X_clf_0[0], y_clf_0).x_rotations_[:, 0]
        rotation_1 = PLSCanonical().fit(X_clf_1[0], y_clf_1).x_rotations_[:, 0]

        cov_mat_0 = TangentSpace().inverse_transform(
            np.reshape(rotation_0, [1, -1]), y_clf_0)

        cov_mat_1 = TangentSpace().inverse_transform(
            np.reshape(rotation_1, [1, -1]), y_clf_1)

        eigen_values, eigen_vectors = linalg.eigh(
            np.dot(linalg.inv(cov_mat_0[0]), cov_mat_1[0]))
        """
        rotation = PLSCanonical().fit(X_clf, y_clf).x_rotations_[:, 0]
        cov_mat = TangentSpace().inverse_transform(
            np.reshape(rotation, [1, -1]), y_clf)


        eigen_values, eigen_vectors = linalg.eigh(cov_mat[0])
        """
        # sort eigenvectors

        ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]
        eigen_vectors = eigen_vectors[:, ix]

        filters_ = eigen_vectors.T
        patterns_ = linalg.pinv2(eigen_vectors)
        pick_filters = filters_[:n_components]
        # eigen_vectors, D = self._ajd_pham(cov_mat)

        datasets = [BNCI2014001()]
        session = datasets[0]._get_single_subject_data(subject=1)
        inf = session['session_T']['run_0']
        info = inf.info
        info['chs'] = info['chs'][:-4]
        info['ch_names'] = info['ch_names'][:-4]
        info['nchan'] = 22

        info = copy.deepcopy(info)
        info['sfreq'] = 1.
        # create an evoked
        patterns = EvokedArray(patterns_.T, info, tmin=0)
        # the call plot_topomap
        patterns.plot_topomap(
            times=[0, 1, 20, 21], ch_type=None, layout=None,
            vmin=None, vmax=None, cmap='RdBu_r', colorbar=True, res=64,
            cbar_fmt='%3.1f', sensors=True,
            scalings=1, units='a.u.', time_unit='s',
            time_format='CPLS%01d', size=1, show_names=False,
            title=None, mask_params=None, mask=None, outlines='head',
            contours=6, image_interp='bilinear', show=True,
            average=None, head_pos=None)
        patterns.plot_topomap(
            times=21, ch_type=None, layout=None,
            vmin=None, vmax=None, cmap='RdBu_r', colorbar=True, res=64,
            cbar_fmt='%3.1f', sensors=True,
            scalings=None, units='a.u.', time_unit='s',
            time_format='CSP%01d', size=1, show_names=False,
            title=None, mask_params=None, mask=None, outlines='head',
            contours=6, image_interp='bilinear', show=True,
            average=None, head_pos=None)
        return patterns
        pdb.set_trace()

    def predict_proba(self, X):
        """transform. """
        pdb.set_trace()
        return X


class TS_Spatial_Filter_full(BaseEstimator, TransformerMixin):

    def __init__(self, clf_str='LR', plot='pattern', n_components=None,
                 log=None, transform_into='average_power', decomp='JEVD'):
        self.clf_str = clf_str
        self.plot = plot
        self.n_components = n_components
        self.transform_into = transform_into
        if transform_into == 'average_power':
            if log is not None and not isinstance(log, bool):
                raise ValueError('log must be a boolean if transform_into == '
                                 '"average_power".')
        else:
            if log is not None:
                raise ValueError('log must be a None if transform_into == '
                                 '"csp_space".')
        self.log = log
        self.decomp = decomp

    def fit(self, X, y):
        X_ = Covariances().fit(X, y).transform(X)
        X_ = TangentSpace().fit(X_, y).transform(X_)

        clf = self.clf_str

        if clf is 'LR':
            self.clf_ = LR()
        elif clf is 'LDA':
            self.clf_ = LDA()
        elif clf is 'SVM':
            self.clf_ = SVM()
        elif clf is 'CCA':
            self.clf_ = CCA()
        elif clf is 'NB':
            self.clf_ = NB()
        else:
            raise ValueError("Wrong classifier! Currently the supported "
                             "classifiers are: LR(), LDA(), SVM(), CCA(), "
                             "NB()")

        ts_coef = self.clf_.fit(X_, y).coef_
        #  ts_mean = np.mean(X_, axis=0)  # Error comparing to Cref:≈10^-11

        cov_mat_coef = TangentSpace().inverse_transform(
            np.reshape(ts_coef, [1, -1]), y)
        cov_mat_mean_ = np.mean(Covariances().fit(X, y).transform(X), axis=0)
        """cov_mat_mean = TangentSpace().inverse_transform(
            np.reshape(ts_mean, [1, -1]), y)
        """
        if self.decomp is 'JEVD' or self.decomp == 'JEVD':
            eigen_values, eigen_vectors = linalg.eigh(
                cov_mat_mean_, cov_mat_coef[0])
        elif self.decomp is 'EVD' or self.decomp == 'EVD':
            eigen_values, eigen_vectors = linalg.eigh(
                cov_mat_coef[0])

        ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]
        eigen_vectors = eigen_vectors[:, ix]

        filters_ = eigen_vectors.T
        patterns_ = linalg.pinv2(eigen_vectors)
        self.filters_ = filters_
        self.patterns_ = patterns_

        if self.plot is 'clf':
            if self.n_components is not None:
                self.coef_ = self.filters_[:self.n_components]
            else:
                self.coef_ = self.filters_
        return self

    def transform(self, X):
        if self.plot is 'clf':
            X = np.asarray([np.dot(self.coef_, epoch) for epoch in X])
            X_cov = Covariances().transform(X)
            X_ts = TangentSpace().transform(X_cov)
            # compute features (mean band power)
            """if self.transform_into == 'average_power':
                X = (X ** 2).mean(axis=2)
                log = True if self.log is None else self.log
                if log:
                    X = np.log(X)
                else:
                    X -= self.mean_
                    X /= self.std_
            """
            return X_ts
        elif self.plot is 'pattern':
            data_re = {'matrix': self.patterns_, 'data': X}
            return data_re
        elif self.plot is 'filter':
            data_re = {'matrix': self.filters_, 'data': X}
            return data_re
        else:
            raise ValueError("Plot string is either 'filter' or 'pattern'"
                             ", for classification please use 'clf'."
                             " (Other strings are not allowed))")


class plot_spatial_filter(BaseEstimator, TransformerMixin):

    def __init__(self, clf_str, n_components, save_path=None, save_type='png',
                 channels=None):
        if clf_str is None:
            raise ValueError("clf_str is compulsory input variable, "
                             "keep same with previous clf_str")
        if n_components is None:
            raise ValueError("n_components is compulsory input variable, "
                             "keep same with previous n_components")
        self.n_components = n_components
        self.clf_str = clf_str

        if save_path is None:
            self.save_path = 'figure/SF/'
            print("Figures will be save into current scirpt's folder.")
        else:
            self.save_path = save_path
            if os.path.exists(save_path) is False:
                os.mkdir(save_path)
                print('New folder are created!')
        self.save_type = save_type

        self.channels = channels

    def fit(self, X, y):
        self.y = y
        with open('temp_metadata.yml') as infile:
            metadata = yaml.load(infile)
        info = metadata['info']
        # os.remove('temp_metadata.yml')
        # Load finish & Delete buffer metadata file

        if self.channels is None:
            self.channels = info['ch_names']

        chn_marker = []
        for i in range(len(info['chs'])):
            if info['chs'][i]['kind'] is 2 and info['ch_names'][i] in self.channels:
                chn_marker.append(i)

        info_plot = copy.deepcopy(info)
        info_plot['chs'] = [info['chs'][i] for i in chn_marker]
        info_plot['ch_names'] = [info['ch_names'][i] for i in chn_marker]
        info_plot['nchan'] = len(chn_marker)
        info_plot['sfreq'] = 1.
        import pdb
        pdb.set_trace()
        # create an evoked
        patterns = EvokedArray(X['matrix'].T, info_plot, tmin=0)
        # the call plot_topomap
        fig = patterns.plot_topomap(
            times=self.n_components, ch_type=None, layout=None,
            vmin=None, vmax=None, cmap='RdBu_r', colorbar=True, res=64,
            cbar_fmt='%3.1f', sensors=True,
            scalings=1, units='Value', time_unit='s',
            time_format=self.clf_str + '%01d', size=1, show_names=False,
            title=None, mask_params=None, mask=None, outlines='head',
            contours=6, image_interp='bilinear', show=False,
            average=None, head_pos=None)
        fig_name = self.save_path + self.clf_str + '_comp' + str(
            len(self.n_components) // 2) + '_'
        cnt = 0
        while os.path.exists(fig_name + str(cnt) + '.' + self.save_type) is True:
                cnt += 1
        if self.save_type is not 'yml':
            fig.savefig(fig_name + str(cnt) + '.' + self.save_type)
        else:
            with open(fig_name + str(cnt) + '.yml', 'w') as infile:
                yaml.dump(patterns, infile, default_flow_style=False)

        return self

    def transform(self, X):
        return X

    def decision_function(self, X):
        return np.ones(X['data'].shape[0])


class TS_Spatial_Filter(BaseEstimator, TransformerMixin):

    def __init__(self, clf_str='LR', plot='pattern', n_components=None,
                 transform_into='csp_space', trans_flag=True):
        self.clf_str = clf_str
        self.plot = plot
        self.n_components = n_components
        self.transform_into = transform_into
        self.file_path = "temp_TS_SF.yml"
        self.trans_flag = trans_flag
        self.delflag = True

    def fit(self, X, y):

        if os.path.exists(self.file_path) is False:
            clf = self.clf_str

            if clf is 'LR':
                self.clf_ = LR()
            elif clf is 'LDA':
                self.clf_ = LDA()
            elif clf is 'SVM':
                self.clf_ = SVM()
            elif clf is 'CCA':
                self.clf_ = CCA()
            else:
                raise ValueError("Wrong classifier! Currently the supported "
                                 "classifiers are: LR(), LDA(), SVM(), CCA()")

            ts_coef = self.clf_.fit(X, y).coef_
            ts_mean = np.mean(X, axis=0)

            cov_mat_coef = TangentSpace().inverse_transform(
                np.reshape(ts_coef, [1, -1]), y)
            cov_mat_mean = TangentSpace().inverse_transform(
                np.reshape(ts_mean, [1, -1]), y)
            eigen_values, eigen_vectors = linalg.eig(
                cov_mat_mean[0], cov_mat_coef[0])
            ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]
            eigen_vectors = eigen_vectors[:, ix]

            filters_ = eigen_vectors.T
            patterns_ = linalg.pinv2(eigen_vectors)
            self.filters_ = filters_
            self.patterns_ = patterns_

            if self.plot is 'pattern':
                return self.patterns_
            elif self.plot is 'filter':
                return self.filters_
            elif self.plot is 'clf':
                if self.n_components is not None:
                    self.coef_ = self.filters_[:self.n_components]
                else:
                    self.coef_ = self.filters_
                if self.trans_flag is True:
                    with open(self.file_path, 'w') as outfile:
                        yaml.dump(self, outfile, default_flow_style=False)
                return self
            else:
                raise ValueError("Plot string is either 'filter' or 'pattern'"
                                 ", for classification please use 'clf'."
                                 " (Other strings are not allowed))")

    def transform(self, X):
        pdb.set_trace()
        self.delflag = not self.delflag
        if (os.path.exists(self.file_path) is True) and (self.delflag is True):
            os.remove(self.file_path)

        return X


class TS_Spatial_Filter_trans(BaseEstimator, TransformerMixin):

    def __init__(self, transform_into='average_power', log=None):

        self.transform_into = transform_into
        self.file_path = "temp_TS_SF.yml"
        if transform_into == 'average_power':
            if log is not None and not isinstance(log, bool):
                raise ValueError('log must be a boolean if transform_into == '
                                 '"average_power".')
        else:
            if log is not None:
                raise ValueError('log must be a None if transform_into == '
                                 '"csp_space".')
        self.log = log

        if os.path.exists(self.file_path) is True:
            os.remove(self.file_path)

    def fit(self, X, y):
        return self

    def transform(self, X):
        if os.path.exists(self.file_path) is True:
            with open(self.file_path) as infile:
                tmp_obj = yaml.load(infile)
            self.coef_ = tmp_obj.coef_
            X = np.asarray([np.dot(self.coef_, epoch) for epoch in X])
            # compute features (mean band power)
            if self.transform_into == 'average_power':
                X = (X ** 2).mean(axis=2)
                log = True if self.log is None else self.log
                if log:
                    X = np.log(X)
                else:
                    X -= self.mean_
                    X /= self.std_
        return X








