import numpy as np

from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin


def alpha_peak_feat(freq, psd, alpha_range=[7.0, 14.0], prior_IAPF=None, prior_AB=None):

    from scipy.signal import savgol_filter
    from scipy.signal import argrelextrema

    a_bp = psd[(freq > alpha_range[0]) & (freq < alpha_range[1])]
    a_ind = freq[(freq > alpha_range[0]) & (freq < alpha_range[1])]
    a_bp_hat = savgol_filter(a_bp, 7, 3)

    if prior_IAPF == None:
        IAPF_raw = np.sum(a_bp*a_ind) / np.sum(a_bp)
        IAPF_smooth = np.sum(a_bp_hat*a_ind) / np.sum(a_bp_hat)
    else:
        IAPF_raw = prior_IAPF.copy()
        IAPF_smooth = prior_IAPF.copy()

    if np.abs(IAPF_raw-IAPF_smooth) > 0.1:
        print(f'IAPF_raw: {str(IAPF_raw)} Hz\n IAPF_smooth: {str(IAPF_smooth)} Hz')
        import pdb;pdb.set_trace()

    IAPF = a_ind[np.abs(a_ind - IAPF_raw).argmin()]
    IAPF_bp_raw = a_bp[a_ind == IAPF][0]
    IAPF_bp_smooth = a_bp_hat[a_ind == IAPF][0]

    maxima = a_ind[argrelextrema(a_bp_hat, np.greater)[0]]
    minima = a_ind[argrelextrema(a_bp_hat, np.less)[0]]

    if minima[(minima-IAPF < 0)].size != 0:
        left_minima_freq = minima[(minima-IAPF < 0)][-1]
    else:
        left_minima_freq = a_ind[0]

    if minima[(minima-IAPF > 0)].size != 0:
        right_minima_freq = minima[(minima-IAPF > 0)][0]
    else:
        right_minima_freq = a_ind[-1]

    if prior_AB == None:
        left_minima_bp_raw = a_bp[(a_ind == left_minima_freq)][0]
        right_minima_bp_raw = a_bp[(a_ind == right_minima_freq)][0]
        left_minima_bp_smooth = a_bp_hat[(a_ind == left_minima_freq)][0]
        right_minima_bp_smooth = a_bp_hat[(a_ind == right_minima_freq)][0]
    else:
        left_minima_bp_raw = prior_AB[0]
        right_minima_bp_raw = prior_AB[1]
        left_minima_bp_smooth = prior_AB[0]
        right_minima_bp_smooth = prior_AB[1]


    ind_IAPF_bp = np.where(a_bp == IAPF_bp_raw)[0][0]
    left_a_bp = a_bp[:ind_IAPF_bp]
    left_a_ind = a_ind[:ind_IAPF_bp]
    right_a_bp = a_bp[ind_IAPF_bp + 1:]
    right_a_ind = a_ind[ind_IAPF_bp + 1:]

    if np.where(IAPF_bp_raw - left_a_bp > 3)[0].size != 0:
        left_3db = left_a_ind[np.where(IAPF_bp_raw - left_a_bp > 3)[0][-1]]
    else:
        left_3db = np.nan

    if np.where(IAPF_bp_raw - left_a_bp > 1)[0].size != 0:
        left_1db = left_a_ind[np.where(IAPF_bp_raw - left_a_bp > 1)[0][-1]]
    else:
        left_1db = np.nan

    if np.where(IAPF_bp_raw - right_a_bp > 3)[0].size != 0:
        right_3db = right_a_ind[np.where(IAPF_bp_raw - right_a_bp > 3)[0][0]]
    else:
        right_3db = np.nan

    if np.where(IAPF_bp_raw - right_a_bp > 1)[0].size != 0:
        right_1db = right_a_ind[np.where(IAPF_bp_raw - right_a_bp > 1)[0][0]]
    else:
        right_1db = np.nan


    feat_dict = {'IAPF': [IAPF, IAPF_raw, IAPF_smooth, IAPF_bp_raw, IAPF_bp_smooth],
                 'Lower_bound': [left_minima_freq, left_minima_bp_raw, left_minima_bp_smooth],
                 'Higher_bound': [right_minima_freq, right_minima_bp_raw, right_minima_bp_smooth],
                 '3db': [left_3db, right_3db],
                 '1db': [left_1db, right_1db],
                 }


    return feat_dict



class LogVariance(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform"""
        assert X.ndim == 3
        return np.log(np.var(X, -1))

class HT(BaseEstimator, TransformerMixin):

    def __init__(self, freq=1000.0):
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
        return self.freq * np.diff(xphase, axis=-1) / (2 * np.pi)



class FM(BaseEstimator, TransformerMixin):

    def __init__(self, freq=1000.0):
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

    def __init__(self, feat=['IA', 'IF']):
        '''instantaneous frequencies require a sampling frequency to be
        properly scaled,
        which is helpful for some algorithms. This assumes 128 if not told
        otherwise.

        '''
        self.feat = feat

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

    def _base_tf(self, X, method):

        self.fs = 1000.0
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

        if method == 'IF':
            return all_feat[method]
        else:
            return all_feat[method][:, :, 1:]

