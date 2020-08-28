import sys
import wave
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import wave
import pickle
import string
import contextlib
import numpy as np
from scipy.io.wavfile import write as sci_write
from pydub import AudioSegment
import shutil
import pandas as pd
from pydub.silence import split_on_silence
import re
from scipy.io import wavfile
import noisereduce as nr
from scipy.io import wavfile # scipy library to read wav files
from mne.filter import notch_filter
import os
from jxu.basiccmd.mycmd import create_folder

def audio_denoise(filename, type='notch', basefreq=2000, increment=1000,
                  process=False, denoise_level=4, new_folder=True):
    if process:
        fs, Audiodata = wavfile.read(filename)
        noisy_part = [wavfile.read('noise.wav')[1],
                      wavfile.read('noise_desk.wav')[1],
                      wavfile.read('noise_desk_book.wav')[1],
                      wavfile.read('noise_mouse.wav')[1],
                      wavfile.read('noise_walking.wav')[1],
                      wavfile.read('noise_door_1.wav')[1],
                      wavfile.read('noise_door_2.wav')[1]]
        reduced_noise = notch_filter(Audiodata.astype(float), fs,
                                     freqs=np.arange(
                                         basefreq, (fs - 100) / 2, increment),
                                     method='fir')
        for incre_level in range(denoise_level):
            reduced_noise = nr.reduce_noise(
                audio_clip=reduced_noise.astype(float),
                noise_clip=noisy_part[incre_level].astype(float),
                verbose=False)

        filtered_audio_int = np.int16(reduced_noise)

        if new_folder:
            new_folder_name = '{0}/Filtered_{1}/'.format(
                '/'.join(filename.split('/')[:-1]), str(denoise_level))
            create_folder(new_folder_name)
            sci_write(new_folder_name + filename.split('/')[-1])
        else:
            sci_write(filename[:-4] + '_filtered_' + str(denoise_level) + '.wav', fs, filtered_audio_int)

    try:
        return filename[:-4] + '_filtered_' + str(denoise_level) + '.wav'
    except:
        raise ValueError('Please turn on the process flag for the first time denoising!')

def speed_change(sound, speed=1.0):
    # NOT SELF WRITTEN!!!!!
    # https://stackoverflow.com/questions/51434897/how-to-change-audio-playback-speed-using-pydub

    # Manually override the frame_rate. This tells the computer how many
    # samples to play per second
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
         "frame_rate": int(sound.frame_rate * speed)
      })
     # convert the sound with altered frame rate to a standard frame rate
     # so that regular playback programs will work right. They often only
     # know how to play audio at standard frame rate (like 44.1k)
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

def wav_std(in_filename, sps=24000, bit=16, channel=1, std_suffix='_std'):
    if in_filename[-4:] == '.wav':
        filename = in_filename[:-4]
    else:
        filename = in_filename

    subprocess.call(['ffmpeg', '-i', filename + '.wav',
                     '-ac', str(channel),
                     '-ar', str(sps),
                     '-sample_fmt', 's'+str(bit),
                     '-y', filename + std_suffix + '.wav'])

def mp3_to_wav(filename, sps=24000, bit=16, channel=1, std_suffix=''):
    """
    Audio options:
    -aframes number     set the number of audio frames to output
    -aq quality         set audio quality (codec-specific)
    -ar rate            set audio sampling rate (in Hz)
    -ac channels        set number of audio channels
    -an                 disable audio
    -acodec codec       force audio codec ('copy' to copy stream)
    -vol volume         change audio volume (256=normal)
    -af filter_graph    set audio filters

    """
    subprocess.call(['ffmpeg', '-i', filename + '.mp3',
                     '-ac', str(channel),
                     '-ar', str(sps),
                     '-sample_fmt', 's'+str(bit),
                     '-y', filename + std_suffix + '.wav'])

def wav_edit(infile, nchn=1, samplewidth=2, sps=24000, cut=False, start=0.0, end=1.0, rm_silence=False, add_silence=False,
    silence_dur=1000, silence_start=0.0, speed=1.0, std_flag=0, out_format='wav', edit_audio= None, outfile=None, speed_optimal= True):

    data= []

    wfile = wave.open(infile, 'rb')
    if std_flag or wfile.getparams()[:3] != (nchn, samplewidth, sps):
        print("Convering audio format: channel " + str(nchn) + ", sample width: " + str(samplewidth) + ', rate: ' + str(sps))
        subprocess.call(['ffmpeg', '-i', infile, '-ac', str(nchn), '-ar', str(sps), '-y', 'standard_' + infile])
        std_flag = 1

    if std_flag:
        sound = AudioSegment.from_file('standard_' + infile)
    else:
        sound = AudioSegment.from_file(infile)


    audio_len = len(sound)


    if cut:
        edit_audio = sound[int(audio_len * start):int(audio_len * end)]
        outfile = 'cut_' + str(start) + '_' + str(end) + '_' + infile


    if rm_silence:
        start_trim = detect_leading_silence(sound, silence_threshold=-80.0, chunk_size=1)
        end_trim = detect_leading_silence(sound.reverse(), silence_threshold=-80.0, chunk_size=1)
        duration_chunk = len(sound)
        if duration_chunk - end_trim - start_trim < 20:
            trimmed_sound = sound
            print(start_trim)
            print(end_trim)
            raise ValueError("Too short audio or too long silence! Not trimmed!")
        else:
            edit_audio = sound[start_trim:duration_chunk - end_trim]
            outfile = 'rm_silence_' + infile

    if add_silence:
        silence_segment = AudioSegment.silent(duration=silence_dur)
        audio_cut = [sound[:int(silence_start * audio_len)], sound[int(silence_start * audio_len):]]
        edit_audio = audio_cut[0] + silence_segment + audio_cut[1]
        outfile = 'add_silence_' + str(silence_dur) + 'ms_start_' + str(silence_start) + '_' + infile

    if speed != 1.0:
        outfile = 'speed_' + str(speed) + '_' + infile
        if speed_optimal:
            subprocess.call(['ffmpeg', '-i', infile, '-filter:a', "atempo=" + str(speed), '-vn', outfile])

            return print("Output file:" + outfile)
        else:
            edit_audio = speed_change(sound, speed)


    if outfile == None or edit_audio == None:
        edit_audio = sound
        outfile = 'copy_' + infile

    edit_audio.export(outfile, format=out_format)

    return print("Output file:" + outfile)


def wav_concat(infiles, outfile, nchn=1, samplewidth=2, sps=44100):

    data= []
    for infile in infiles:
        wfile = wave.open(infile, 'rb')
        if wfile.getparams()[:3] != (nchn, samplewidth, sps):
            temp_file_name = infile[:-4] + '_temp' + infile[-4:]
            print("Convering audio format: channel " + str(nchn) + ", sample width: " + str(samplewidth) + ', rate: ' + str(sps))
            subprocess.call(['ffmpeg', '-i', infile, '-ac', str(nchn), '-ar', str(sps), '-y', temp_file_name])
            wfile = wave.open(temp_file_name, 'rb')
            subprocess.call(['rm', temp_file_name])

        data.append( [wfile.getparams(), wfile.readframes(wfile.getnframes())] )
        wfile.close()

    output = wave.open(outfile, 'wb')
    output.setparams(data[0][0])
    for ind in range(len(infiles)):
        output.writeframes(data[ind][1])

    output.close()
    del wfile, output, data, ind

    return print("Output file:" + outfile)



def audio_spec(AudioName):
    # Not self-written, refered from:
    # https://stackoverflow.com/questions/24382832/audio-spectrum-extraction-from-audio-file-by-python

    fs, Audiodata = wavfile.read(AudioName)

    # Plot the audio signal in time
    import matplotlib.pyplot as plt
    plt.plot(Audiodata)
    plt.title('Audio signal in time',size=16)

    from scipy.signal import hilbert, chirp

    analytic_signal = hilbert(Audiodata)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) /
                               (2.0*np.pi) * fs)
    fig = plt.figure()
    ax0 = fig.add_subplot(311)
    ax0.plot(Audiodata, label='signal')
    ax0.plot(amplitude_envelope, label='envelope')
    ax0.set_xlabel("time in seconds")
    ax0.legend()
    ax1 = fig.add_subplot(312)
    ax1.plot(instantaneous_frequency)
    ax1.set_title('IF')
    ax1.set_xlabel("time in seconds")
    plt.title('Instantaneous Audio signal',size=16)

    # spectrum
    from scipy.fftpack import fft # fourier transform
    n = len(Audiodata)
    AudioFreq = fft(Audiodata)
    AudioFreq = AudioFreq[0:int(np.ceil((n+1)/2.0))] #Half of the spectrum
    MagFreq = np.abs(AudioFreq) # Magnitude
    MagFreq = MagFreq / float(n)
    # power spectrum
    MagFreq = MagFreq**2
    if n % 2 > 0: # ffte odd
        MagFreq[1:len(MagFreq)] = MagFreq[1:len(MagFreq)] * 2
    else:# fft even
        MagFreq[1:len(MagFreq) -1] = MagFreq[1:len(MagFreq) - 1] * 2

    plt.figure()
    freqAxis = np.arange(0,int(np.ceil((n+1)/2.0)), 1.0) * (fs / n);
    plt.plot(freqAxis/1000.0, 10*np.log10(MagFreq)) #Power spectrum
    plt.xlabel('Frequency (kHz)'); plt.ylabel('Power spectrum (dB)');


    #Spectrogram
    from scipy import signal
    N = 512 #Number of point in the fft
    import pdb
    pdb.set_trace()
    f, t, Sxx = signal.spectrogram(Audiodata, fs,window = signal.blackman(N),nfft=N)
    plt.figure()
    plt.pcolormesh(t, f,10*np.log10(Sxx)) # dB spectrogram
    #plt.pcolormesh(t, f,Sxx) # Lineal spectrogram
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [seg]')
    plt.title('Spectrogram with scipy.signal',size=16);

    plt.show()



def audio_onedim(filename, wav=True, metric='default', pflag=True, sps=44100.0):

    if wav:
        newfilename = filename
    else:
        subprocess.call(['ffmpeg', '-i', filename, newfilename])

    spf = wave.open(newfilename, "r")

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, "Int16")
    maxValue = np.max(np.abs(signal))

    if metric == 'default':
        sig = signal
    elif metric == 'dBFS':
        sig = 20*np.log10(abs(signal)/maxValue)

    if pflag:
        plt.figure(1)
        plt.title("Signal Wave")
        plt.plot(np.arange(len(signal)) / sps, sig)
        plt.show()

    print('Min:' + str(np.sort(np.unique(sig))[1]))
    return np.sort(np.unique(sig))[1]


def detect_leading_silence(sound, silence_threshold=-87.0, chunk_size=5):
    trim_ms = 0 # ms
    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size
    return trim_ms

def plain_beep(dur, freq=440, vol=0.5, sps=24000, bit=16, chn=1):

    esm = np.arange(dur * sps)
    wf = np.sin(2 * np.pi * esm * freq / sps)
    wf_quiet = wf * vol
    wf_int = np.int16(wf_quiet * 32767)

    beep_segment = AudioSegment(wf_int.tobytes(), frame_rate=sps, sample_width=int(bit/8), channels=chn)

    return beep_segment

# beep_generator('C5_A_tone_flat_3quater_s.wav', tone='flat', slice_duration=0.75, beep_freq=[1760.0])
def beep_generator(file_name, file_root='/home/jxu/File/Data/NIBS/Stage_one/Audio/Soundeffect/',
                   slice_duration=1.0, tone='flat', vol=0.3, sps=44100, beep_freq=440.0):

    if tone == 'flat':
        amp = vol
    elif tone == 'increase':
        amp = np.linspace(vol*0.3, vol, slice_duration * sps)
    elif tone == 'decrease':
        amp = np.linspace(vol, vol*0.3, slice_duration * sps)
    else:
        raise ValueError("Wrong input of tone!")

    if not isinstance(beep_freq, list):
        beep_freq = [beep_freq]

    esm = np.arange(slice_duration * sps)
    wf_full = np.empty(0,)

    for freq in beep_freq:
        wf = np.sin(2 * np.pi * esm * freq / sps)
        wf_slice = wf * amp
        wf_full = np.concatenate((wf_full, wf_slice))

    if tone != 'flat':
        wf_full = np.concatenate((wf_full, ))
    wf_int = np.int16(wf_full * 32767)
    if not os.path.exists(file_root + 'beep/'):
        os.mkdir(file_root + 'beep/')
    beep_file = file_root + 'beep/' + file_name
    sci_write(beep_file, sps, wf_int)

    return print('Saved file:' + beep_file)


def beep_censoring(file_root='/home/jxu/File/Data/NIBS/Stage_one/Audio/Database/', article_id=0, audio_rate=1.0,
                   pitch=0.0, beep_word_type='VERB', sps=24000, beep_freq=40.0, vol=1.0):

    folder_path = file_root + 'article_{0}_speed_{1}_pitch_{2}/'.format(article_id, audio_rate, pitch)
    n_audiofile = len([name for name in os.listdir(folder_path) if name[:-1] == 'sentence_'])
    col_name = [('PATH', 'file_root_ori'), ('PATH', 'file_root_syn'),
                ('META_INFO', 'audio_rate'), ('META_INFO', 'pitch'), ('META_INFO', 'language'),
                ('META_INFO', 'tag_list'), ('META_INFO', 'n_tag'),
                ('SENTENCE_INFO', 'article_id'), ('SENTENCE_INFO', 'sen_id'), ('SENTENCE_INFO', 'sen_content'),
                ('SENTENCE_INFO', 'beeped_sen_content'),
                ('SENTENCE_INFO', 'beep_word_type'), ('SENTENCE_INFO', 'beeped_word'), ('SENTENCE_INFO', 'beeped_word_duration'),
                ('SENTENCE_INFO', 'beeped_word_timestamp_start'), ('SENTENCE_INFO', 'beeped_word_timestamp_end'),
                ('SENTENCE_INFO', 'sentence_duration'),
                ('EXP_INFO', 'S01'), ('EXP_INFO', 'S02'), ('EXP_INFO', 'S03'), ('EXP_INFO', 'S04'), ('EXP_INFO', 'S05'),
                ('EXP_INFO', 'S06'), ('EXP_INFO', 'S07'), ('EXP_INFO', 'S08'), ('EXP_INFO', 'S09'), ('EXP_INFO', 'S10')]
    dataframe_path = file_root + 'all_beep_df.pkl'
    if not os.path.exists(dataframe_path):
        empty_df = pd.DataFrame(columns=col_name)
        empty_df.columns = pd.MultiIndex.from_tuples(empty_df.columns, names=['General','Detail'])
        empty_df.to_pickle(dataframe_path)

    all_beep_df = pd.read_pickle(dataframe_path)
    empty_sen_df = pd.DataFrame(columns=col_name)
    empty_sen_df.columns = pd.MultiIndex.from_tuples(empty_sen_df.columns, names=['General','Detail'])

    punc = string.punctuation

    for sen_ind in range(n_audiofile):

        sen_folder_path = folder_path + 'sentence_{0}/'.format(sen_ind)
        chunk_folder_path = sen_folder_path + 'chunk/'

        with open(sen_folder_path + 'taglist.pkl', 'rb') as f:
            sen_tag = pickle.load(f)

        sen_content = ' '.join([sen_tag[i][0] for i in range(len(sen_tag))])
        n_chunk = len([name for name in os.listdir(chunk_folder_path) if name[:5] == 'chunk' and name[-7:-4] != 'tmp'])

        break_list = {'VERB': 2000, 'NOUN': 1800, 'ADJ': 1600, 'DET': 1400, 'ADV': 1200, 'AUX': 1000}
        # tag_list = [word_tag for word_tag in sen_tag if (word_tag[0] not in punc ) or (word_tag[1] not in [*break_list.keys()] )]
        tag_list = [word_tag for word_tag in sen_tag if word_tag[0] not in punc]

        n_tag = len(tag_list)

        if n_chunk != n_tag:
            print("Inconsistent number of chunks and tags. Please check chunked files!!!")
            print('Tags are: ' + str(n_tag))
            print(tag_list)
            print('Chunks are: ' + str(n_chunk))
            import pdb
            pdb.set_trace()
        else:
            beep_ind = [[word_ind, word_tag] for word_ind, word_tag in enumerate(tag_list) if word_tag[1] == beep_word_type]
            if len(beep_ind) == 0:
                print("No matching word type " + beep_word_type + " in the sentence! Path: " + sen_folder_path)
            elif len(beep_ind) > 1:
                print("More than ONE word will be replaced with beeping noise!")

            for itr_ind in range(len(beep_ind)):
                chunk_ind = beep_ind[itr_ind][0]
                beeped_chunk = chunk_folder_path + "chunk{0}.wav".format(chunk_ind)
                print("Replacing " + beep_ind[itr_ind][1][1] + ': ' + beep_ind[itr_ind][1][0] + ' with beeping noise')
                try:
                    shutil.copy(beeped_chunk, chunk_folder_path + "chunk{0}_tmp.wav".format(chunk_ind))
                except:
                    import pdb
                    pdb.set_trace()
                with contextlib.closing(wave.open(beeped_chunk,'r')) as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration = frames / float(rate)
                    # print(duration)
                esm = np.arange(duration * sps)
                wf = np.sin(2 * np.pi * esm * beep_freq / sps)
                wf_quiet = wf * vol
                wf_int = np.int16(wf_quiet * 32767)
                sci_write(beeped_chunk, sps, wf_int)

                # os.system('play -nq -t alsa synth {} sine {}'.format(duration, beep_freq))

                for com_chunk_ind in range(n_chunk):
                    chunk_file = chunk_folder_path + "chunk{0}.wav".format(com_chunk_ind)
                    sound = AudioSegment.from_wav(chunk_file)
                    start_trim = detect_leading_silence(sound, silence_threshold=-80.0, chunk_size=1)
                    end_trim = detect_leading_silence(sound.reverse(), silence_threshold=-80.0, chunk_size=1)
                    duration_chunk = len(sound)
                    if (com_chunk_ind == beep_ind[0][0]) or (duration_chunk - end_trim - start_trim < 20):
                        trimmed_sound = sound
                        print(start_trim)
                        print(end_trim)
                        print("not trimmed")
                    else:
                        trimmed_sound = sound[start_trim:duration_chunk - end_trim]

                    if chunk_ind == com_chunk_ind:
                        try:
                            censor_start = len(combined_sounds) / 1000
                        except:
                            censor_start = 0.0

                    if com_chunk_ind == 0:
                        combined_sounds = trimmed_sound
                    else:
                        combined_sounds = combined_sounds + trimmed_sound

                    if chunk_ind == com_chunk_ind:
                        censor_end = len(combined_sounds) / 1000

                beeped_sentence = re.sub(r'(?is)' + beep_ind[itr_ind][1][0], '[MASK]', sen_content)

                new_fname = sen_folder_path + 'sentence_{0}_syn_{1}_{2}.wav'.format(sen_ind, beep_word_type, itr_ind)
                combined_sounds.export(new_fname, format="wav")
                sen_duration = len(combined_sounds) / 1000
                data = {('PATH', 'file_root_ori'): [sen_folder_path + 'sentence_{0}.mp3'.format(sen_ind)],
                        ('PATH', 'file_root_syn'): [new_fname],
                        ('META_INFO', 'audio_rate'): [audio_rate],
                        ('META_INFO', 'pitch'): [pitch],
                        ('META_INFO', 'language'): ['German'],
                        ('META_INFO', 'tag_list'): [tag_list],
                        ('META_INFO', 'n_tag'): [n_tag],
                        ('SENTENCE_INFO', 'article_id'): [article_id],
                        ('SENTENCE_INFO', 'sen_id'): [sen_ind],
                        ('SENTENCE_INFO', 'sen_content'): [sen_content],
                        ('SENTENCE_INFO', 'beeped_sen_content'): [beeped_sentence],
                        ('SENTENCE_INFO', 'beep_word_type'): [beep_word_type],
                        ('SENTENCE_INFO', 'beeped_word'): [beep_ind[itr_ind][1][0]],
                        ('SENTENCE_INFO', 'beeped_word_duration'): [duration],
                        ('SENTENCE_INFO', 'beeped_word_timestamp_start'): [censor_start],
                        ('SENTENCE_INFO', 'beeped_word_timestamp_end'): [censor_end],
                        ('SENTENCE_INFO', 'sentence_duration'): [sen_duration],
                        ('EXP_INFO', 'S01'): None, ('EXP_INFO', 'S02'): None,
                        ('EXP_INFO', 'S03'): None, ('EXP_INFO', 'S04'): None,
                        ('EXP_INFO', 'S05'): None, ('EXP_INFO', 'S06'): None,
                        ('EXP_INFO', 'S07'): None, ('EXP_INFO', 'S08'): None,
                        ('EXP_INFO', 'S09'): None, ('EXP_INFO', 'S10'): None}

                tmp_df = pd.DataFrame(data)
                empty_sen_df = pd.concat([empty_sen_df, tmp_df], ignore_index=True)


                # audio_onedim(new_fname, wav=True, metric='dBFS')

                shutil.copy(chunk_folder_path + "chunk{0}_tmp.wav".format(chunk_ind), beeped_chunk)
                os.remove(chunk_folder_path + "chunk{0}_tmp.wav".format(chunk_ind))


    empty_sen_df.to_pickle(folder_path + 'article_beep.pkl')

    all_beep_df = pd.concat([all_beep_df, empty_sen_df], ignore_index=True)

    col_name.pop(col_name.index(('META_INFO', 'tag_list')))
    all_beep_df.drop_duplicates(subset=col_name, keep='first', inplace=True)
    all_beep_df.columns = pd.MultiIndex.from_tuples(all_beep_df.columns, names=['Caps','Lower'])
    all_beep_df.to_pickle(dataframe_path)


def shattered_audio_to_chunk(file_root='/home/jxu/File/Data/NIBS/Stage_one/Audio/Database/', article_id=0, audio_rate=1.0, pitch=0.0):

    # folder_path = file_root + 'article_{0}/'.format(article_id)
    folder_path = file_root + 'article_{0}_speed_{1}_pitch_{2}/'.format(article_id, audio_rate, pitch)
    n_audiofile = len([name for name in os.listdir(folder_path) if name[:-1] == 'sentence_'])

    for sen_ind in range(n_audiofile):
        sen_folder_path = folder_path + 'sentence_{0}/'.format(sen_ind)

        if len([name for name in os.listdir(sen_folder_path) if name[-4:] == '.wav']) == 0:
            import subprocess
            audio_name_trim = sen_folder_path + 'sentence_{0}_ori'.format(sen_ind)
            subprocess.call(['ffmpeg', '-i', audio_name_trim + '.mp3', audio_name_trim + '.wav'])

        audio_name = sen_folder_path + 'sentence_{0}_ori'.format(sen_ind) + '.wav'
        min_amp = audio_onedim(audio_name, wav=True, metric='dBFS', pflag=False)

        sound_file = AudioSegment.from_wav(audio_name)
        audio_chunks = split_on_silence(sound_file, min_silence_len=380,silence_thresh=np.floor(min_amp))

        for i, chunk in enumerate(audio_chunks):
            chunk_folder_path = sen_folder_path + 'chunk/'
            if not os.path.exists(chunk_folder_path):
                os.mkdir(chunk_folder_path)
            out_file = chunk_folder_path + "chunk{0}.wav".format(i)
            print("exporting" + out_file)
            chunk.export(out_file, format="wav")

    return print("All audio files are chunked into word/phrase level!")


def remove_silence(sound, silence_threshold=-80.0):
    start_trim = detect_leading_silence(sound, silence_threshold=silence_threshold, chunk_size=1)
    end_trim = detect_leading_silence(sound.reverse(), silence_threshold=silence_threshold, chunk_size=1)
    duration_chunk = len(sound)
    if duration_chunk - end_trim - start_trim < 20:
        trimmed_sound = sound
        print("not trimmed")
        return sound
    else:
        return sound[start_trim:duration_chunk - end_trim]


def minimal_audio_to_chunk(audio_name, chunk_folder_path=None, save=True):

    min_amp = audio_onedim(audio_name, wav=True, metric='dBFS', pflag=False)

    sound_file = AudioSegment.from_wav(audio_name)
    audio_chunks = split_on_silence(sound_file, min_silence_len=380,silence_thresh=np.floor(min_amp))

    if save and chunk_folder_path is not None:
        for i, chunk in enumerate(audio_chunks):
            if not os.path.exists(chunk_folder_path):
                os.mkdir(chunk_folder_path)
            out_file = chunk_folder_path + "chunk{0}.wav".format(i)
            print("exporting" + out_file)
            chunk.export(out_file, format="wav")

        print("All audio files are chunked into word/phrase level and saved!")

    return audio_chunks
