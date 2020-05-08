
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

x = True

#if condition returns False, AssertionError is raised:
assert x == False, "This package must be imported under google_virtual_env"

import os
import pandas as pd
from pattern3.de import split as pat_split
from pattern3.de import parse as pat_parse
from google.cloud import texttospeech
from subprocess import call
import pickle
import string
import spacy


def google_text_to_speech(ssml_string, audio_location, speed=1.0, pitch=0.0, lang='de-DE'):
    """Synthesizes speech from the input string of text or ssml.

    Note: ssml must be well-formed according to:
        https://www.w3.org/TR/speech-synthesis/
    """

    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech.types.SynthesisInput(ssml=ssml_string)

    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice = texttospeech.types.VoiceSelectionParams(
        language_code=lang,   # de-DE, en-US, cmn-CN (cmn-CN-Standard-C)i , (de Standard-B, Wavenet-D)
        ssml_gender=texttospeech.enums.SsmlVoiceGender.MALE)

    audio_config = texttospeech.types.AudioConfig(
        audio_encoding=texttospeech.enums.AudioEncoding.MP3,   # : LINEAR 16, OGG_OPUS
        speaking_rate=speed,   # 0.25 - 4.00
        pitch=pitch)   # -20.00 - 20.00
        
    """ Other paramaters for AudioConfig
        volumn_gain_db=,
        sample_rate_hertz=,
        effects_profile_id=,
    """

    response = client.synthesize_speech(synthesis_input, voice, audio_config)
    # The response's audio_content is binary.
    with open(audio_location, 'wb') as audio_out:
        # Write the response to the output file.
        audio_out.write(response.audio_content)
        print('Audio content written to file ' + audio_location)


def tts_shattered(nlp, article, article_id=0, file_root='/home/jxu/File/Data/NIBS/Stage_one/Audio/Database/Shattered/',
        audio_rate=1.0, pitch=0.0, lang='de-DE'):

    folder_path = file_root + 'article_{0}_speed_{1}_pitch_{2}/'.format(article_id, audio_rate, pitch)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    col_name = ['file_root', 'article_id', 'sen_id', 'sen_content', 'tag_list', 'n_tag', 'speed', 'pitch', 'language']
    dataframe_path = file_root + 'all_ori_df.pkl'
    if not os.path.exists(dataframe_path):
        empty_df = pd.DataFrame(columns=col_name)
        empty_df.to_pickle(dataframe_path)
    all_df = pd.read_pickle(dataframe_path)

    empty_sen_df = pd.DataFrame(columns=col_name)
    #empty_sen_df.to_pickle(dataframe_path)

    # nlp = spacy.load("de_core_news_sm")
    article_content = nlp(article)

    parsed_sen = [sent.string.strip() for sent in article_content.sents]


    # POS tag are from: https://spacy.io/api/annotation
    break_list = {'VERB': 2000, 'NOUN': 1800, 'ADJ': 1600, 'DET': 1400, 'ADV': 1200, 'AUX': 1000,
                  'ADP': 1000, 'CONJ': 600, 'CCONJ': 800, 'INTJ': 800, 'NUM': 800, 'PART': 800,
                  'PRON':800, 'PROPN':800, 'PUNCT':800, 'SCONJ':800, 'SYM':800}
    # break_list = {'VERB': 2000}
    

    ssml_root = '<speak>', '</speak>'
    ssml_break = lambda x: "<break time=\"{0}ms\"/>".format(x)

    for sen_ind, sen_content in enumerate(parsed_sen):
        doc = nlp(sen_content)
        sen_str = sen_content
        sen_tag = [token.pos_ for token in doc]
        sen_str_list = [token.text for token in doc]
        sen_info = [[token.text, token.pos_] for token in doc]

        sen_folder_path = folder_path + 'sentence_{0}/'.format(sen_ind)
        if not os.path.exists(sen_folder_path):
            os.mkdir(sen_folder_path)

        file_path = sen_folder_path + 'sentence_{0}_ori'.format(sen_ind)        
        audio_loc = file_path + '.mp3'
        text_loc = file_path + '.txt'

        ssml_string = ssml_root[0]

        for word_ind, word_content in enumerate(sen_info):
            with open(sen_folder_path + 'taglist.pkl', 'wb') as f:
                pickle.dump(sen_info, f)
            if word_content[1] not in [*break_list.keys()]:
                ssml_string += word_content[0]
            else:
                ssml_string += ssml_break(break_list[word_content[1]])
                ssml_string += word_content[0]
                ssml_string += ssml_break(break_list[word_content[1]])
            ssml_string += ' '     
        ssml_string += ssml_root[1]

        # intialise data of lists. 

        punc = string.punctuation
        tag_list = [word_tag for word_tag in sen_info if word_tag[0] not in punc]
  
        data = {'file_root':[file_root],
                'article_id':[article_id],
                'sen_id':[sen_ind],
                'sen_content': [sen_str],
                'tag_list': [tag_list],
                'n_tag': [len(tag_list)],
                'speed': [audio_rate],
                'pitch': [pitch],
                'language': ['German']
                }
        tmp_df = pd.DataFrame(data)
        empty_sen_df = pd.concat([empty_sen_df, tmp_df], ignore_index=True)

        text_file = open(text_loc, "w")
        text_file.write(sen_str + ' \n \n'+ ssml_string)
        text_file.close()


        try:
            google_text_to_speech(ssml_string, audio_loc, speed=audio_rate, pitch=pitch, lang=lang)
        except:
            credentials = input('Would you like to assign credentials? (y/n)\n')
            if credentials.lower() == 'y':
                # cmd = 'export GOOGLE_APPLICATION_CREDENTIALS="/home/jxu/Code/Google_Cloud_Plattform/STS-PRIVATE-d0ca8d9f6870.json"'
               	print("Please run following commands under google_cloud_env. (private credentials: STS-PRIVATE-d0ca8d9f6870.json)") 
                print("export GOOGLE_APPLICATION_CREDENTIALS='/home/jxu/Code/Google_Cloud_Plattform/fsgni_nibs_tts.json'")
            else:
                raise ValueError("Credentials assignment of google account is required!")

    all_df = pd.concat([all_df, empty_sen_df], ignore_index=True)

    col_name.pop(col_name.index('tag_list'))
    all_df.drop_duplicates(subset=col_name, keep='first', inplace=True)
    all_df.to_pickle(dataframe_path)


def tts_unshattered(nlp, article, article_id=0, file_root='/home/jxu/File/Data/NIBS/Stage_one/Audio/Database/Unshattered/',
                    audio_rate=1.0, pitch=0.0, lang='de-DE', word_type='VERB', df_name='all_unshattered_ori_df.pkl'):
    # POS tag are from: https://spacy.io/api/annotation
    # ['VERB', 'NOUN', 'ADJ', 'DET', 'ADV', 'AUX', 'ADP', 'CONJ',
    #  'CCONJ', 'INTJ', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM']
    folder_path = file_root + 'article_{0}_speed_{1}_pitch_{2}/'.format(article_id, audio_rate, pitch)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    col_name = ['folder_path', 'speed', 'pitch', 'language', 'article_id', 'sen_id', 'sen_content',
        'tag_list', 'n_tag', 'censored_word_type', 'censored_word_id_relative', 'censored_word_id_abs', 'censored_word',
        'ssml_string', 'last_word_flag', 'file_path', 'audio_loc']
    dataframe_path = file_root + df_name
    if not os.path.exists(dataframe_path):
        empty_df = pd.DataFrame(columns=col_name)
        empty_df.to_pickle(dataframe_path)
    all_df = pd.read_pickle(dataframe_path)

    empty_sen_df = pd.DataFrame(columns=col_name)
    #empty_sen_df.to_pickle(dataframe_path)

    # nlp = spacy.load("de_core_news_sm")
    article_content = nlp(article)

    parsed_sen = [sent.string.strip() for sent in article_content.sents]

    break_list = 1000
    # break_list = {'VERB': 2000}


    ssml_root = '<speak>', '</speak>'
    ssml_break = lambda x: "<break time=\"{0}ms\"/>".format(x)

    for sen_ind, sen_content in enumerate(parsed_sen):

        doc = nlp(sen_content)
        sen_str = sen_content
        sen_tag = [token.pos_ for token in doc]
        sen_str_list = [token.text for token in doc]
        sen_info = [[token.text, token.pos_] for token in doc]
        tag_indices = [index for index, con in enumerate(sen_tag) if con == word_type]


        sen_folder_path = folder_path + 'sentence_{0}_{1}/'.format(sen_ind, word_type)
        if not os.path.exists(sen_folder_path):
            os.mkdir(sen_folder_path)

        with open(sen_folder_path + 'taglist.pkl', 'wb') as f:
            pickle.dump(sen_info, f)

        # intialise data of lists. 
        punc = string.punctuation
        tag_list = [word_tag for word_tag in sen_info if word_tag[0] not in punc]
      
        for i_tagged_word_ind, tagged_word_ind in enumerate(tag_indices):
            # It's forbidden to censor the first word.
            #
            if tagged_word_ind == 0: 
                continue

            if tag_list[-1] == sen_info[tagged_word_ind]:
                last_word_flag = True
            else:
                last_word_flag = False

            file_path = sen_folder_path + 'sentence_{0}_{1}_{2}_ori'.format(
                sen_ind, word_type, str(i_tagged_word_ind))        
            audio_loc = file_path + '.mp3'
            text_loc = file_path + '.txt'

            ssml_string = ssml_root[0]

            for word_ind, word_content in enumerate(sen_info):

                if word_ind != tagged_word_ind:
                    ssml_string += word_content[0]
                else:
                    ssml_string += ssml_break(break_list)
                    ssml_string += word_content[0]
                    ssml_string += ssml_break(break_list)
                ssml_string += ' '     
            ssml_string += ssml_root[1]

            data = {'folder_path':[folder_path],
                    'speed': [audio_rate],
                    'pitch': [pitch],
                    'language': ['German'],
                    'article_id':[article_id],
                    'sen_id':[sen_ind],
                    'sen_content': [sen_str],
                    'tag_list': [tag_list],
                    'n_tag': [len(tag_list)],
                    'censored_word_type': [word_type],
                    'censored_word_id_relative': [i_tagged_word_ind],
                    'censored_word_id_abs': [tagged_word_ind],
                    'censored_word': [sen_info[tagged_word_ind][0]],
                    'ssml_string': [ssml_string],
                    'last_word_flag': [last_word_flag],
                    'file_path':[file_path],
                    'audio_loc':[audio_loc],
                    }
            tmp_df = pd.DataFrame(data)
            empty_sen_df = pd.concat([empty_sen_df, tmp_df], ignore_index=True)

            text_file = open(text_loc, "w")
            text_file.write(sen_str + ' \n \n'+ ssml_string)
            text_file.close()


            try:
                google_text_to_speech(ssml_string, audio_loc, speed=audio_rate, pitch=pitch, lang=lang)
            except:
                credentials = input('Would you like to assign credentials? (y/n)\n')
                if credentials.lower() == 'y':
                    # cmd = 'export GOOGLE_APPLICATION_CREDENTIALS="/home/jxu/Code/Google_Cloud_Plattform/STS-PRIVATE-d0ca8d9f6870.json"'
                    print("Please run following commands under google_cloud_env. (private credentials: STS-PRIVATE-d0ca8d9f6870.json)") 
                    print("export GOOGLE_APPLICATION_CREDENTIALS='/home/jxu/Code/Google_Cloud_Plattform/fsgni_nibs_tts.json'")
                    return 0
                else:
                    raise ValueError("Credentials assignment of google account is required!")

    import pdb 
    pdb.set_trace()
    all_df = pd.concat([all_df, empty_sen_df], ignore_index=True)

    # Remove duplicate sentences
    no_duplicate_col_name = ['sen_content', 'censored_word', 'ssml_string']
    all_df.drop_duplicates(subset=no_duplicate_col_name, keep='first', inplace=True)
    all_df.to_pickle(dataframe_path)

