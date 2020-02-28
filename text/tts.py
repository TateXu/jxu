
import os, os.path
import numpy as np
import pickle
import string
from pydub import AudioSegment
from pydub.silence import split_on_silence
from jxu.visualization.audiosignal import audio_onedim
from pattern3.de import split as pat_split
from pattern3.de import parse as pat_parse
# from google.cloud import texttospeech
"""
CC  coordinating conjunction
CD  cardinal digit
DT  determiner
EX  existential there (like: "there is" ... think of it like "there exists")
FW  foreign word
IN  preposition/subordinating conjunction
JJ  adjective   'big'
JJR adjective, comparative  'bigger'
JJS adjective, superlative  'biggest'
LS  list marker 1)
MD  modal   could, will
NN  noun, singular 'desk'
NNS noun plural 'desks'
NNP proper noun, singular   'Harrison'
NNPS    proper noun, plural 'Americans'
PDT predeterminer   'all the kids'
POS possessive ending   parent's
PRP personal pronoun    I, he, she
PRP$    possessive pronoun  my, his, hers
RB  adverb  very, silently,
RBR adverb, comparative better
RBS adverb, superlative best
RP  particle    give up
TO  to  go 'to' the store.
UH  interjection    errrrrrrrm
VB  verb, base form take
VBD verb, past tense    took
VBG verb, gerund/present participle taking
VBN verb, past participle   taken
VBP verb, sing. present, non-3d take
VBZ verb, 3rd person sing. present  takes
WDT wh-determiner   which
WP  wh-pronoun  who, what
WP$ possessive wh-pronoun   whose
WRB wh-abverb   where, when
"""

def audio_to_chunk(file_root='/home/jxu/File/Data/NIBS/Stage_one/Audio/Database/', article_id=0, audio_rate=1.0, pitch=0.0):

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
        audio_chunks = split_on_silence(sound_file, min_silence_len=390,silence_thresh=np.floor(min_amp))

        for i, chunk in enumerate(audio_chunks):
            chunk_folder_path = sen_folder_path + 'chunk/'
            if not os.path.exists(chunk_folder_path):
                os.mkdir(chunk_folder_path)
            out_file = chunk_folder_path + "chunk{0}.wav".format(i)
            print("exporting" + out_file)
            chunk.export(out_file, format="wav")

    return print("All audio files are chunked into word/phrase level!")


def google_text_to_speech(ssml_string, audio_location, speed=1.0, pitch=0.0):
    
    print("refer to /home/jxu/Code/Google_Cloud_Plattform/GCP_env/lib/python3.7/site-packages/jxu/basiccmd")


def tts(article, article_id=0, file_root='/home/jxu/File/Data/NIBS/Stage_one/Audio/Database/', audio_rate=1.0, pitch=0.0):

    print("refer to /home/jxu/Code/Google_Cloud_Plattform/GCP_env/lib/python3.7/site-packages/jxu/basiccmd")
