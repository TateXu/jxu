import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import re
from jxu.text.textprocessing import mismatching, stemming_word, find_de_synonym
import time
import spacy


def fill_the_gaps(text, tokenizer, model):
    text = '[CLS] ' + text + '[SEP]'
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)
    results = []
    masked_index = tokenized_text.index('[MASK]')
    """
    for i, t in enumerate(tokenized_text):
        if t == '[MASK]':
            predicted_index = torch.argmax(predictions[0, i]).item()
            predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
            results.append(predicted_token)
    return predicted_token
    """
    k = 30
    predicted_index, predicted_index_values = torch.topk(predictions[0, masked_index], k)
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_index_values.tolist())
    filtered_tokens_to_remove_punctuation = []
    # Remove any predictions that contain punctuation etc as they are not relevant to us.
    for token in predicted_tokens:
        if re.match("^[a-zA-Z0-9_]*$", token):
            filtered_tokens_to_remove_punctuation.append(token)
        
    return filtered_tokens_to_remove_punctuation

aaa = pd.read_pickle('all_beep_df.pkl')
BERT = True
if BERT: 
    TOKEN_PATH, MODEL_PATH = ['bert-base-multilingual-cased'] * 2
else:
    TOKEN_PATH = '/home/jxu/File/Code/German_BERT/model/model.json' # 'bert-base-multilingual-cased'
    MODEL_PATH = '/home/jxu/File/Code/German_BERT/model/bert-base-german-cased.tar.gz' # 'bert-base-multilingual-cased' 

# 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(TOKEN_PATH)
model = BertForMaskedLM.from_pretrained(MODEL_PATH)
model.eval(); # turning off the dropout

"""
sentence = 'Anna hat [MASK] Geld .'
print(fill_the_gaps(text = sentence, tokenizer=tokenizer, model=model))

mismatching('haben', 'brauchen', language='de')
"""
for ind in range(10):
    sentence = aaa.at[ind, ('SENTENCE_INFO', 'sen_content')]
    beeped_word = aaa.at[ind, ('SENTENCE_INFO', 'beeped_word')]
    beep_sentence = sentence.replace
    beeped_sentence = re.sub(r'(?is)' + beeped_word, '[MASK]', sentence)
    answer_list = fill_the_gaps(text = beeped_sentence, tokenizer=tokenizer, model=model)

    maxscore_list = []
    for answer in answer_list:

        try: 
            maxword, maxscore, dict_eval = mismatching(answer, beeped_word, language='de')
        except:
            try:
                print("Stemming necessary!!!")
                maxword, maxscore, dict_eval = mismatching(stemming_word(answer), stemming_word(beeped_word), language='de')
            except:
                print("Synthesized vocab including!!!")
                maxscore = 0
                maxword = None
        maxscore_list.append(maxscore)
        time.sleep(0.3)
    print(beeped_sentence)
    print(beeped_word)
    print(answer_list)
    print(maxscore_list)

