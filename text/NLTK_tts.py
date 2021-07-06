import os
import pandas as pd
from pattern3.de import split as pat_split
from pattern3.de import parse as pat_parse
from google.cloud import texttospeech
from subprocess import call
import pickle
import string

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


def tts(article, article_id=0, file_root='/home/jxu/File/Data/NIBS/Stage_one/Audio/Database/', audio_rate=1.0, pitch=0.0, lang='de-DE'):

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

    parsed_atc = pat_parse(article)
    parsed_sen = pat_split(parsed_atc)

    break_list = {'VB': 2000, 'NN': 1800, 'JJ': 1600, 'DT': 1400, 'IN': 1200, 'RB': 1000, 'PR': 800, 'SUJ': 600, 'OBJ': 400}
    ssml_root = '<speak>', '</speak>'
    ssml_break = lambda x: "<break time=\"{0}ms\"/>".format(x)

    for sen_ind, sen_content in enumerate(parsed_sen):
        sen_folder_path = folder_path + 'sentence_{0}/'.format(sen_ind)
        if not os.path.exists(sen_folder_path):
            os.mkdir(sen_folder_path)


        file_path = sen_folder_path + 'sentence_{0}_ori'.format(sen_ind)        
        audio_loc = file_path + '.mp3'
        text_loc = file_path + '.txt'


        ssml_string = ssml_root[0]
        sen_str = sen_content.string
        sen_str_list = [word_content[0] for word_content in sen_content.tagged]
        sen_tag = [word_content[1] for word_content in sen_content.tagged]

        for word_ind, word_content in enumerate(sen_content.tagged):
            with open(sen_folder_path + 'taglist.pkl', 'wb') as f:
                pickle.dump(sen_content.tagged, f)
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
        tag_list = [word_tag for word_tag in sen_content.tagged if word_tag[0] not in punc]
  
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

