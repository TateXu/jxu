import pickle
import spacy
import re
import pdb
import pandas as pd


file_loc = '/home/jxu/File/Experiment/NIBS/Sync/NIBS_paradigm/text/'
file_name = 'jxu.xlsx'
df = pd.read_excel(file_loc + file_name)

tts_flag = True
audio_rate = 0.9
pitch = 0.0

if tts_flag:
    total_len = 0
    nlp = spacy.load("de_core_news_md")
    from jxu.basiccmd.tts import *
else:
    from jxu.audio.audiosignal import *


empty_list = []
if tts_flag:
    total_len = 0
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
        """ 
        if tag_indices == []:
            # print(row.permanent_index)
            empty_list.append(row.permanent_index)
        print(index)
        """
        tts_unshattered(nlp, article, article_id=article_id, file_root='/home/jxu/File/Data/NIBS/Stage_one/Audio/Database/Unshattered/audio/',
                        audio_rate=audio_rate, pitch=pitch, lang='de-DE', word_type='VERB', word_content=word,
                        df_name='all_unshattered_ori_df_2.pkl', permanent_index=permanent_index)
        

else:
    for ind_censor_tag, censor_tag in enumerate(['VERB']):   # ['VERB', 'NOUN', 'ADJ', 'DET', 'ADV', 'AUX']
        for i in range(200):
            audio_to_chunk(article_id=i, audio_rate=audio_rate)
            beep_censoring(article_id=i, beep_word_type=censor_tag,  audio_rate=audio_rate)
"""
[797, 791, 786, 780, 774, 773, 772, 764, 754, 748,
 747, 739, 738, 733, 724, 723, 721, 715, 712, 711,
 710, 705, 704, 700, 699, 698, 697, 694, 687, 684,
 679, 677, 666, 655, 645, 641, 637, 628, 627, 609,
 607, 581, 578, 572, 570, 569, 568, 567, 566, 559,
 548, 542, 541, 540, 537, 535, 534, 533, 494, 486,
 466, 465, 399, 395, 391, 378, 375, 373, 363, 353,
 339, 338, 330, 325, 311, 298, 297, 291, 284, 282,
 281, 277, 264, 262, 254, 249, 233, 212]
"""
pdb.set_trace()
