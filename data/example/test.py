from jxu.data.loader import *
from jxu.audio.audiosignal import *
from auditok import AudioRegion
from scipy.io import wavfile
import noisereduce as nr
import pdb



answer = ['geöffnet',
'gehen',
None,
'spült',
'kommt',
'arbeitet',
None,
'unclear: beziehen',
'unclear: ideitzt', 
'nimmst',
'geflogen',
'kochst',
'schützen',
'gehen',
'unclear: kauf',
None,
None,
'tragen',
'geputzt',
None,
'unclear: fällt',  
None,
None,
'hat',
'gekauft',
'unclear: advoliert',
None,
'muss',
None,
None,
'schmeckt',
'wachsen',
'muss',
None,
'unclear: lehrt',
'fragen',
'wachsen',
'studiert',
'schmertzen',
'baut',
None,
'unclear: liest',
'kostet',
'unclear: gesicht',
'kann',
None,
'schneidet',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'kann',
'spielt',
'gedauert',
'guckt',
'entspannt',
'bietet',
'denkst',
'geschneidet',
'gestolen',
'kocht',
'unclear: dragt',
'wachsen',
'tantzt',
'geöffnet',
'schmerz, schmertzt',
None,
'musst',
'geht',
None,
'wachst',
'studiert',
None,
'nimmst',
'kann',
'scheinnt',
'verstehe',
'hört',
'hat',
'isst',
'unclear: blite',
'arbeitet',
'hilft',
'aufstehen',
'schneidet',
'geht',
'unclear: advoliert',
'spült',
'sind',
None,
None,
'kann',
'geschnitten',
None,
'suche',
'schützt, schützen',
'essen',
None,
'soll',
'könnten',
'gekommen',
'kann',
'kann',
'sehen',
'möchte',
'gelaufen',
'machen',
None,
'schmertzen',
'sind',
'hat',
'gekommen',
'fliegt',
'spielt, spielen',
'verstolen',
'schmeckt',
'können',
None,
'fliegt',
'können',
'gedauert',
None,
'fehlt',
None,
'musst',
'geöffnet',
'antwortet',
None,
'unclear: liest',
'spielen',
None,
'unclear: schmuzig']
all_cnt = []
pdb.set_trace()
for nr_trial in range(160):

    # name = audio_denoise(audio_loader(trial=nr_trial))
    
    # wav_std(audio_loader(trial=nr_trial, std=False), sps=44100)
    # for ind in range(5, 7):
    #     audio_denoise(audio_loader(trial=nr_trial), process=True, denoise_level=ind)
    # print(nr_trial)
    
    region = AudioRegion.load(audio_denoise(audio_loader(trial=nr_trial), process=False, denoise_level=4), skip=0.15)
    aaa = region.split(energy_threshold=51.5)
    all_cnt.append(len(list(aaa)))
    del aaa

    # regions = region.split_and_plot(energy_threshold=60) # or just region.splitp()
    # for segment in aaa:
    #     au_seg_s = segment.meta.start 
    #     au_seg_e = segment.meta.end
    #     cnt += 1

import pdb

print(all_cnt)
print('== 0: ' + str(np.where( np.asarray(all_cnt) == 0)[0].shape))
print('== 1: ' + str(np.where( np.asarray(all_cnt) == 1)[0].shape))
print('== 2: ' + str(np.where( np.asarray(all_cnt) == 2)[0].shape))
print('== 3: ' + str(np.where( np.asarray(all_cnt) == 3)[0].shape))
pdb.set_trace()


# audio_denoise(audio_loader(trial=nr_trial), process=True)


# Aggressive  way!
# (Pdb) np.where( np.asarray(all_cnt) ==  0)

# Prior:
# (array([  2,   6,  15,  16,  19,  21,  22,  26,  28,  29,  33,  40,  45,
#           94,  97, 100,  117, 118, 121, 125, 135, 145, 149, 151, 155, 158]),)

# (array([  2,   6,  15,  16,  19,  21,  22,  26,  28,  29,  33,  40,  45,
#         97, 100, 117, 118, 121, 125, 135, 145, 149, 151, 158]),)

# Level 4 - full:
# (array([  2,       15,  16,  19,  21,       26,  28,  29,  33,  40,  45,
#           94,  95,  97, 100, 101, 109, 117,           125, 135, 145, 149, 151,      158]),)
# Level 4 - 53 - 0.1:
# (array([  2,       15,  16,  19,  21,       26,  28,  29,  33,  40,  45,  
#           94,       97, 100,           117, 118, 121, 125, 135, 145, 149, 151,      158]),)

# Level 4 - 53 - 0.15:
# (array([  2,   6,  15,  16,  19,  21,  22,  26,  28,  29,  33,  40,  45,
#           94,      97, 100,            117, 118, 121, 125, 135, 145, 149, 151,    158]),)

# (array([  2,   6,  15,  16,  19,  21,  22,  26,  28,  29,  33,  40,  45,
#                    97, 100,            117, 118, 121, 125, 135, 145, 149, 151,    158]),)

# Level 4 - 51 - 0.15:
# (array([  2,   6,  15,  16,  19,  21,  22,  26,  28,  29,  33,  40,  45,
#                    97, 100,            117, 118, 121, 125, 135, 145, 149, 151,    158]),)

# Level 4 - 50 - 0.15:

# array([   2,   6,  15,  16,  19,  21,       26,  28,  29,  33,  40,  45,
#                    97, 100,            117, 118, 121, 125, 135, 145, 149, 151,    158]),





# Level 3:
# (array([  2,       15,  16,  19,  21,       26,  28,  29,  33,  40,  45, 
#           94,  95,      100, 101, 109, 117,           125, 135, 145, 149, 151, 158]),)


(array([    2,   6,  15,  16,  19,  21,       26,  28,  29,  33,  40,  45, 
            94,      97, 100, 101,       117, 118,      125, 135, 145, 149, 151,      158]),)




# Level 2 - 55 - full:

# (array([           15,  16,  19,  21,       26,  28,  29,  33,  40,  45,  
#          94,            100,          117,            125, 135, 145,                158]),)

# Level 2 - 57 - full:
# (array([  1,   2,  15,  16,  19,  21,  26,  28,  29,  33,  40,  45,  
#            94,  95, 100, 101, 109,
#        117, 125, 135, 145, 158]),)


# # Level 2 - 57 - cut:
# (array([  1,   2,   6,   8,  15,  16,  19,  21,  26,  28,  29,  30,  33,
#         40,  45,  
#            94,  95,  97, 100, 101, 106, 109, 117, 118, 125, 135, 145, 149, 151, 158]),)


# Level 1: 
# (array([ 15,  21,  26,  28,  33,  40,  45,  
#          94, 100, 125, 135, 145, 158]),)

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




# car 

# talking outside
# door (215)
