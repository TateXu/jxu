import pickle
import spacy
import re
import pdb
import pandas as pd
from jxu.basiccmd.tts import *
# from jxu.audio.audiosignal import *

file_loc = '/home/jxu/File/Experiment/NIBS/Sync/NIBS_paradigm/text/'
file_name = 'jxu.xlsx'
df = pd.read_excel(file_loc + file_name)

tts_flag = True
audio_rate = 0.9
pitch = 0.0

nlp = spacy.load("de_core_news_md")
for index, row in df.iterrows():
    #    permanent_index, beeped_sen_content, verb, noun

    if index < 778:
        continue
    article_id = index
    article = row.beeped_sen_content
    word = row.verb
    permanent_index = row.permanent_index
    print(index)

    tts_unshattered(nlp, article, article_id=article_id, file_root='/home/jxu/File/Data/NIBS/Stage_one/Audio/Database/Unshattered/audio/',
                    audio_rate=audio_rate, pitch=pitch, lang='de-DE', word_type='VERB', word_content=word,
                    df_name='all_unshattered_ori_df_2.pkl', permanent_index=permanent_index)

# For further audio processing, refer to audio/unshattered_censoring.py
pdb.set_trace()
