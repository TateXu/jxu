from jxu.data.loader import *
from jxu.audio.audiosignal import *
from auditok import AudioRegion
from scipy.io import wavfile
import noisereduce as nr

all_cnt = []
for nr_trial in range(160):
    # name = audio_denoise(audio_loader(trial=nr_trial))
    
    # wav_std(audio_loader(trial=nr_trial, std=False), sps=44100)
    region = AudioRegion.load(audio_denoise(audio_loader(trial=nr_trial), process=True))
    # regions = region.split_and_plot(energy_threshold=60) # or just region.splitp()
    aaa = region.split(energy_threshold=55)
    all_cnt.append(len(list(aaa)))
    del aaa
    print(nr_trial)
    # for segment in aaa:
    #     au_seg_s = segment.meta.start
    #     au_seg_e = segment.meta.end
    #     cnt += 1

import pdb
pdb.set_trace()

np.where( np.asarray(all_cnt) ==  0)

# audio_denoise(audio_loader(trial=nr_trial), process=True)


# Aggressive  way!
# (Pdb) np.where( np.asarray(all_cnt) ==  0)
# (array([  2,  15,  16,  19,  21,  26,  28,  29,  33,  40,  45,  47,  48,
#         49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
#         62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,
#         75,  76,  77,  78,  79,  94,  95,  97, 100, 101, 109, 117, 125,
#        135, 145, 149, 151, 158]),)


# (array([ 15,  21,  26,  28,  33,  40,  45,  47,  48,  49,  50,  51,  52,
#         53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
#         66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,
#         79,  94, 100, 125, 135, 145, 158]),)
# from scipy.signal import hilbert

# analytic_signal = hilbert(reduced_noise)
# amplitude_envelope = np.abs(analytic_signal)
# instantaneous_phase = np.unwrap(np.angle(analytic_signal))
# instantaneous_frequency = (np.diff(instantaneous_phase) /
#                            (2.0*np.pi) * fs)