from jxu.data.loader import *
from jxu.audio.audiosignal import *
from auditok import AudioRegion
from scipy.io import wavfile
import noisereduce as nr
import pdb
import pickle

from pydub import AudioSegment
from pydub.playback import play

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

answer = np.asarray(answer)
answer_0 = np.where(answer== None)[0]

all_cnt = []
pdb.set_trace()
folder_path = '/Users/xujiachen/File/Data/NIBS/Stage_one/ZWS/ZWS_SESSION_1/Audio_Recording/Exp_data/'

preprocess_flag = False
search_opt_para_flag = False
slice_into_seg_flag = False
mark_seg_flag = False
valid_raw_seg_extract_flag = False
if preprocess_flag:
    for nr_trial in range(160):
        wav_std(audio_loader(subject=0, session=1, trial=nr_trial, std=False), sps=44100)
        for ind in range(7):
            audio_denoise(audio_loader(subject=0, session=1, trial=nr_trial, std=True), process=True, denoise_level=ind)
        print(nr_trial)


if search_opt_para_flag:

    for nr_trial in range(160):
        region = AudioRegion.load(audio_denoise(audio_loader(subject=0, session=1, trial=nr_trial, std=True),
                                                process=False, denoise_level=4), skip=0.15)
        audio_segs = list(region.split(energy_threshold=51.5))
        # if len(audio_segs) != 0:
        #     [seg.save(folder_path + 'Segments/QA_trial' + str(nr_trial) + '_seg_' + str(id_seg) + '.wav') for id_seg, seg in enumerate(audio_segs)]
        all_cnt.append(len(audio_segs))

        # regions = region.split_and_plot(energy_threshold=60) # or just region.splitp()
        # for segment in aaa:
        #     au_seg_s = segment.meta.start 
        #     au_seg_e = segment.meta.end
        #     cnt += 1
else:
    opt_skip = 0.15
    opt_noise_level = 4
    opt_ET = 51.5

if slice_into_seg_flag:
    for nr_trial in range(160):
        region = AudioRegion.load(audio_denoise(audio_loader(subject=0, session=1, trial=nr_trial, std=True),
                                                process=False, denoise_level=opt_noise_level), skip=opt_skip)
        audio_segs = list(region.split(energy_threshold=opt_ET))

        if len(audio_segs) != 0:
            for id_seg, seg in enumerate(audio_segs):
                seg_name = folder_path + 'Segments/QA_trial_' + str(nr_trial) + '_seg_' + str(id_seg) + '.wav'
                seg.save(seg_name)
                all_cnt.append([nr_trial, len(audio_segs), seg.meta.start + opt_skip, seg.meta.end + opt_skip])
        else:
            all_cnt.append([nr_trial, len(audio_segs), None, None]) 
    all_cnt = np.asarray(all_cnt)
    with open(folder_path + 'Segments/segment_list.pkl', 'wb') as f:
        pickle.dump(all_cnt, f)
else:
    with open(folder_path + 'Segments/segment_list.pkl', 'rb') as f:
        all_cnt = pickle.load(f)

if mark_seg_flag:
    mark_list = []
    pdb.set_trace()
    for nr_trial in range(160):

        trial_ind = np.where(all_cnt[:, 0] == nr_trial)[0]

        if len(trial_ind) == 1:
            if all_cnt[trial_ind, 1] == 0:
                mark_list.append([nr_trial, None])
                continue

        seg_mark = None
        for id_seg, ignore_seg in enumerate(trial_ind):
            seg_name = folder_path + 'Segments/QA_trial_' + str(nr_trial) + '_seg_' + str(id_seg) + '.wav'
            song = AudioSegment.from_wav(seg_name)
            play(song)

            noise_flag = input('Is this segment a noise clip? (y/n)?')
            while noise_flag != 'y' and noise_flag != 'n':
                noise_flag = input('Invalid input, please only input y or n!')
            if noise_flag.lower() == 'n':
                if seg_mark != None:
                    pdb.set_trace()
                    print('Duplicate seg mark!!!')
                seg_mark = id_seg
        mark_list.append([nr_trial, seg_mark])
        print(mark_list[-1])

        
                    # if noise_flag.lower() == 'y':
    mark_list = np.asarray(mark_list)
    with open(folder_path + 'Segments/marked_segment_list.pkl', 'wb') as f:
        pickle.dump(mark_list, f)
else:
    with open(folder_path + 'Segments/marked_segment_list.pkl', 'rb') as f:
        mark_list = pickle.load(f)


if valid_raw_seg_extract_flag:
    if not np.any(np.where(mark_list==None)[0] == np.where(answer==None)[0]):
        print('Segment info: ')
        print(np.where(mark_list==None)[0])
        print('Prior info: ')
        print(np.where(answer==None)[0])
        raise ValueError('Inconsistency between prior info and segment info! Please check!')
    onset_list = []
    duration_list = []

    for nr_trial in range(160):
        if mark_list[nr_trial, 1] is None:
            onset_list.append([nr_trial, None])
            duration_list.append([nr_trial, None])
        else:

            trial_ind = np.where(all_cnt[:, 0] == nr_trial)[0]

            trial_segments = all_cnt[trial_ind]
            start, end = trial_segments[mark_list[nr_trial, 1]][2:]

            extend_dur = 0.3

            crop_start = start - extend_dur if start - extend_dur >= 0 else 0.0
            crop_end = end + extend_dur

            audio_file = AudioSegment.from_wav(audio_denoise(audio_loader(subject=0, session=1, trial=nr_trial, std=True),
                                               process=False, denoise_level=4))
            audio_seg = audio_file[crop_start * 1000:crop_end * 1000]
            start_trim = detect_leading_silence(audio_seg, silence_threshold=-80.0, chunk_size=1)
            end_trim = detect_leading_silence(audio_seg.reverse(), silence_threshold=-80.0, chunk_size=1)

            onset = crop_start + start_trim / 1000.0
            duration = len(audio_seg[start_trim:-end_trim-1]) / 1000.0

            valid_audio_file = AudioSegment.from_wav(audio_denoise(audio_loader(subject=0, session=1, trial=nr_trial, std=True),
                                                     process=False, denoise_level=2))
            valid_seg = valid_audio_file[onset * 1000 : (onset+ duration) * 1000]
            valid_seg.export(folder_path + 'Valid_segs/QA_trial_' + str(nr_trial) + '.wav', format='wav')

            onset_list.append([nr_trial, onset])
            duration_list.append([nr_trial, duration])
        print("Onset")
        print(onset_list[-1])
        print("Duration")
        print(duration_list[-1])

    with open(folder_path + 'Valid_segs/onset_list.pkl', 'wb') as f:
        pickle.dump(onset_list, f)
    with open(folder_path + 'Valid_segs/duration_list.pkl', 'wb') as f:
        pickle.dump(duration_list, f)
else:
    with open(folder_path + 'Valid_segs/onset_list.pkl', 'rb') as f:
        onset_list = pickle.load(f)
    with open(folder_path + 'Valid_segs/duration_list.pkl', 'rb') as f:
        duration_list = pickle.load(f)
    duration_list = np.asarray(duration_list)
    onset_list = np.asarray(onset_list)
pdb.set_trace()



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
#                97, 100,  117, 118, 121, 125, 135, 145, 149, 151,      158]),)

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
