from jxu.audio.audiosignal import *
import pandas as pd

file_root='/home/jxu/File/Data/NIBS/Stage_one/Audio/Database/Unshattered/'

ori_df='all_unshattered_ori_df.pkl'

beep_df='all_unshattered_beep_df.pkl'
article_id=0
audio_rate=1.0
pitch=0.0
beep_word_type='VERB'
sps=24000
beep_freq=220.0
vol=1.0
bit=16
chn=1



col_name = [('PATH', 'file_root_ori'), ('PATH', 'file_root_syn'),
            ('META_INFO', 'audio_rate'), ('META_INFO', 'pitch'), ('META_INFO', 'language'),
            ('META_INFO', 'tag_list'), ('META_INFO', 'n_tag'),
            ('SENTENCE_INFO', 'article_id'), ('SENTENCE_INFO', 'sen_id'), ('SENTENCE_INFO', 'sen_content'),
            ('SENTENCE_INFO', 'beeped_sen_content'),
            ('SENTENCE_INFO', 'beep_word_type'), ('SENTENCE_INFO', 'beeped_word'), ('SENTENCE_INFO', 'beeped_word_duration'), 
            ('SENTENCE_INFO', 'beeped_word_timestamp_start'), ('SENTENCE_INFO', 'beeped_word_timestamp_end'),
            ('SENTENCE_INFO', 'sentence_duration'), ('EXP_INFO', 'S00'),
            ('EXP_INFO', 'S01'), ('EXP_INFO', 'S02'), ('EXP_INFO', 'S03'), ('EXP_INFO', 'S04'), ('EXP_INFO', 'S05'),
            ('EXP_INFO', 'S06'), ('EXP_INFO', 'S07'), ('EXP_INFO', 'S08'), ('EXP_INFO', 'S09'), ('EXP_INFO', 'S10')]

# folder_path = file_root + 'article_{0}_speed_{1}_pitch_{2}/'.format(article_id, audio_rate, pitch)
# n_audiofile = len([name for name in os.listdir(folder_path) if 'sentence' in name and beep_word_type in name ])

try: 
    all_ori_df = pd.read_pickle(file_root + ori_df)
except:
    raise ValueError('Given invailid name for ori_df!')

dataframe_path = file_root + beep_df
if not os.path.exists(dataframe_path):
    empty_df = pd.DataFrame(columns=col_name)
    empty_df.columns = pd.MultiIndex.from_tuples(empty_df.columns, names=['General','Detail'])
    empty_df.to_pickle(dataframe_path)
all_beep_df = pd.read_pickle(dataframe_path)


empty_sen_df = pd.DataFrame(columns=col_name)
empty_sen_df.columns = pd.MultiIndex.from_tuples(empty_sen_df.columns, names=['General','Detail'])

punc = string.punctuation


selected_df = all_ori_df.copy().loc[(all_ori_df['sync_status'] == False)]

new_all_beep_df = [all_beep_df]
for index, entry in selected_df.iterrows(): 
    if index==0:
        break
    audio_name = entry.audio_loc[:-4]
    mp3_to_wav(audio_name, sps=24000, channel=1, std_suffix='_std')

    audio_chunks = minimal_audio_to_chunk(audio_name + '_std.wav', save=False)

    if len(audio_chunks) not in [2, 3]:
        raise ValueError('Number of chunks can only be either 2 or 3.')

    clean_chunks = [remove_silence(single_chunk) for single_chunk in audio_chunks]
    beep_duration = len(clean_chunks[1]) / 1000
    beep_chunk = plain_beep(dur=beep_duration, freq=beep_freq, vol=vol, sps=sps, bit=bit, chn=chn)
    try:
        combined_sounds = clean_chunks[0] + beep_chunk + clean_chunks[2]
    except:
        combined_sounds = clean_chunks[0] + beep_chunk


    new_fname = entry.folder_path + 'sentence_{0}_{1}/sentence_{0}_{1}_{2}_syn.wav'.format(
        entry['sen_id'], beep_word_type, entry['censored_word_id_relative'])
    combined_sounds.export(new_fname, format="wav")

    beeped_sentence = re.sub(r'(?is)' + entry.censored_word, '[MASK]', entry.sen_content)

    sen_duration = len(combined_sounds) / 1000

    data = {('PATH', 'file_root_ori'): [entry.audio_loc],
            ('PATH', 'file_root_syn'): [new_fname],
            ('META_INFO', 'audio_rate'): [entry.speed],
            ('META_INFO', 'pitch'): [entry.pitch],
            ('META_INFO', 'language'): [entry.language],
            ('META_INFO', 'tag_list'): [entry.tag_list],
            ('META_INFO', 'n_tag'): [entry.n_tag],
            ('SENTENCE_INFO', 'article_id'): [entry.article_id],
            ('SENTENCE_INFO', 'sen_id'): [entry.sen_id],
            ('SENTENCE_INFO', 'sen_content'): [entry.sen_content],
            ('SENTENCE_INFO', 'beeped_sen_content'): [beeped_sentence],
            ('SENTENCE_INFO', 'beep_word_type'): [beep_word_type],
            ('SENTENCE_INFO', 'beeped_word'): [entry.censored_word],
            ('SENTENCE_INFO', 'beeped_word_duration'): [len(beep_chunk) / 1000],
            ('SENTENCE_INFO', 'beeped_word_timestamp_start'): [len(clean_chunks[0]) / 1000],
            ('SENTENCE_INFO', 'beeped_word_timestamp_end'): [len(clean_chunks[0] + clean_chunks[1]) / 1000],
            ('SENTENCE_INFO', 'sentence_duration'): [sen_duration],
            ('EXP_INFO', 'S00'): None,
            ('EXP_INFO', 'S01'): None, ('EXP_INFO', 'S02'): None,
            ('EXP_INFO', 'S03'): None, ('EXP_INFO', 'S04'): None,
            ('EXP_INFO', 'S05'): None, ('EXP_INFO', 'S06'): None,
            ('EXP_INFO', 'S07'): None, ('EXP_INFO', 'S08'): None,
            ('EXP_INFO', 'S09'): None, ('EXP_INFO', 'S10'): None}

    new_all_beep_df.append(pd.DataFrame(data))


all_ori_df.loc[(all_ori_df['sync_status'] == False),'sync_status'] = True
all_beep_df = pd.concat(new_all_beep_df, ignore_index=True)

no_duplicate_col_name = ['sen_content', 'censored_word', 'ssml_string']
all_beep_df.drop_duplicates(subset=no_duplicate_col_name, keep='first', inplace=True)
all_beep_df.columns = pd.MultiIndex.from_tuples(all_beep_df.columns, names=['Caps','Lower'])
import pdb 
pdb.set_trace()
all_beep_df.to_pickle(dataframe_path)
all_ori_df.to_pickle(file_root + 'temp_' + ori_df)