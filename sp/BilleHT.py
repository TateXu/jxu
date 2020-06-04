import numpy as np
import scipy.linalg
import scipy.io
import scipy.signal
# jxu
from scipy import io as sio

def load_data():
    data = sio.loadmat('../data/s01.mat', squeeze_me=True, struct_as_record=False, verify_compressed_data_integrity=False)['eeg']

    imagery_left = data.imagery_left - \
        data.imagery_left.mean(axis=1, keepdims=True)
    imagery_right = data.imagery_right - \
        data.imagery_right.mean(axis=1, keepdims=True)

    eeg_data_l = np.vstack([imagery_left * 1e-6, data.imagery_event])
    eeg_data_r = np.vstack([imagery_right * 1e-6,
                            data.imagery_event * 2])
    eeg_data = np.hstack([eeg_data_l, np.zeros((eeg_data_l.shape[0], 500)),
                          eeg_data_r])

    return eeg_data 

class PeakFrequency:

    def __init__(self, channels, samples, fs, bands=None):
        self.channels = channels
        self.samples = samples
        self.fs = fs
        self.dft = scipy.linalg.dft(samples)
        self.idft = np.linalg.inv(self.dft)
        import pdb
        pdb.set_trace()
        self.hilbert = np.zeros(samples)
        if samples % 2 == 0:
            self.hilbert[0] = self.hilbert[samples // 2] = 1
            self.hilbert[1:samples // 2] = 2
        else:
            self.hilbert[0] = 1
            self.hilbert[1:(samples + 1) // 2] = 2
        if channels > 1:
            ind = [np.newaxis] * 2
            ind[-1] = slice(None)
            self.hilbert = self.hilbert[tuple(ind)]
        if bands is None:
            self.bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 45)}
        else:
            self.bands = bands

    def do(self, x):
        x = np.asarray(x)
        if x.shape[0] is not self.samples and x.shape[1] is not self.channels:
            raise ValueError("configs (", self.channels, ",", self.samples, ") do not match input dims ", x.shape)
        if np.iscomplexobj(x):
            raise ValueError("x is not a real signal.")
        H = self.dft.dot(x) * self.hilbert.T

        instant_frequency = dict()
        for band in self.bands:
            signal = np.zeros((self.samples, self.channels), dtype=complex)
            from_val = self.bands[band][0]
            to_val = self.bands[band][1]
            signal[from_val:to_val, :] = H[from_val:to_val, :]
            import pdb
            pdb.set_trace()
            from scipy.signal import hilbert
            xa = hilbert(signal)   
            signal = signal.T.dot(self.idft).T

            inst_phase = np.unwrap(np.angle(signal))
            inst_freq = np.diff(inst_phase, axis=0) / (2 * np.pi) * self.fs
            instant_frequency[band] = np.median(inst_freq, axis=0)
        # returns a dict for each frequency band containing a channel vector for each electrode's median inst freq
        return instant_frequency


if __name__ == "__main__":
    #  MAT = scipy.io.loadmat('motor-imagery-eeg.mat')
    #  dict_keys = [*MAT.keys()]
    #  X = MAT[dict_keys[3]]
    #  data = X[0, :100, :10]
    X = load_data()
    data = X[:10, :100].T
    fs = 500
    bands = {'all': (1, 45)}
    #  H = PeakFrequency(data.shape[0], data.shape[1], fs, bands)
    H = PeakFrequency(10, 100, fs, bands)
    res = H.do(data)


    print(res["all"][0])
