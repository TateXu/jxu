
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



def google_text_to_speech(ssml_string, audio_location, speed=1.0, pitch=0.0):
    
    print("refer to /home/jxu/Code/Google_Cloud_Plattform/GCP_env/lib/python3.7/site-packages/jxu/basiccmd")


def tts(article, article_id=0, file_root='/home/jxu/File/Data/NIBS/Stage_one/Audio/Database/', audio_rate=1.0, pitch=0.0):

    print("refer to /home/jxu/Code/Google_Cloud_Plattform/GCP_env/lib/python3.7/site-packages/jxu/basiccmd")
