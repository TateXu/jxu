import pickle
import spacy
import re
import pdb




with open('data_200_session_1.pkl', 'rb') as file:
    article = pickle.load(file)

audio_rate=0.9

tts_flag = False
if tts_flag:
    from jxu.basiccmd.tts import *

else:
    from jxu.audio.audiosignal import *

if tts_flag:
    total_len = 0
    nlp = spacy.load("de_core_news_md")

    for i in range(1):
        tts(nlp, article[i], article_id=i, audio_rate=audio_rate)

else:
    for ind_censor_tag, censor_tag in enumerate(['VERB']):   # ['VERB', 'NOUN', 'ADJ', 'DET', 'ADV', 'AUX']
        for i in range(200):
            audio_to_chunk(article_id=i, audio_rate=audio_rate)
            beep_censoring(article_id=i, beep_word_type=censor_tag,  audio_rate=audio_rate)

import pickle
import spacy
import re
import pdb
from jxu.basiccmd.tts import *
nlp = spacy.load("de_core_news_md")
article = 'Ich habe gerade meine Bewerbung bei der Universität eingereicht.'
word = 'eingereicht'
tts_unshattered(nlp, article, article_id=1, file_root='/home/jxu/File/Data/NIBS/Stage_one/Audio/Database/Unshattered/',
                audio_rate=0.9, pitch=0.0, lang='de-DE', word_type='VERB', word_content=word,
                df_name='all_unshattered_ori_df.pkl', permanent_index=0)



pdb.set_trace()


import pandas as pd

all_df = pd.read_pickle('all_beep_df.pkl')

all_missing_df = all_df[[('SENTENCE_INFO', 'beeped_sen_content'),('SENTENCE_INFO', 'beep_word_type')]]
all_missing_df.to_pickle('all_missing_df.pkl')
all_missing_df.to_excel('all_question.xlsx')
