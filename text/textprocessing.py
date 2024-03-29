import re 
import urllib.parse
import urllib.request
from scrapy.selector import Selector
from scrapy.http import HtmlResponse, TextResponse
import difflib
import numpy as np
import operator
import spacy

def stemming_word(vocab):
    
    nlp = spacy.load('de')# nlp = spacy.load('de_core_news_md')
    doc = nlp(vocab)

    return doc[0].lemma_

def find_synonym(vocab, language='en'):

    if language == 'en':
        return find_en_synonym(vocab)
    elif language == 'de':
        return find_de_synonym(vocab)
    else:
        try:
            return find_en_synonym(vocab, language)
        except Exception as e:
            raise ValueError("Chosen language is not yet supported!")
        

def mismatching(subject_answer, correct_answer, language='en'):

    score = {}
    answer_list = find_synonym(correct_answer, language)
    answer_list[correct_answer] = [correct_answer]

    max_score = {} 
    for key_word, value in answer_list.items():
        score[key_word] = [np.round(difflib.SequenceMatcher(None, subject_answer, vocab).ratio(), 4) for vocab in value]
        index, value = max(enumerate(score[key_word]), key=operator.itemgetter(1))
        max_score[key_word] = value

    max_index, max_value = max(enumerate(max_score.values()), key=operator.itemgetter(1))
    return [*max_score.keys()][max_index], max_value, score


def find_en_synonym(vocab, suffix='englisch'):

    req = urllib.request.urlopen('https://synonyme.woxikon.de/synonyme-' + suffix + '/' + vocab.lower() + '.php')
    return_str = req.read().decode("utf-8")
    response = Selector(text=return_str)

    divs = response.xpath('//div[contains(@class, "upper")]')
    divs_title = response.xpath('//div[contains(@class, "synonyms-list-group")]')
    synonyme_dict = {}
    title = []
    for i in range(len(divs_title)):
        try:
            title.append(divs_title[i].xpath('.//b/text()').extract()[0])
        except Exception:
            pass
    for i in range(len(divs)):
        unclean = divs[i].xpath('.//span[contains(@class, "text-black")]/text()').extract()
        synonyme_dict[title[i]] = [re.sub(r"\W+\s*", " ", x)[1: -1] for x in unclean] 

    return synonyme_dict


def find_de_synonym(vocab):

    try: 
        req = urllib.request.urlopen('https://synonyme.woxikon.de/synonyme/' + vocab.lower() + '.php')
    except:
        raise ValueError('No synonym available! Please input the original form (primitive) of the input vocab.')
    return_str = req.read().decode("utf-8")
    response = Selector(text=return_str)

    divs = response.xpath('//div[contains(@class, "upper")]')
    divs_title = response.xpath('//div[contains(@class, "synonyms-list-group")]')
    synonyme_dict = {}
    title = []
    for i in range(len(divs_title)):
        try:
            title.append(divs_title[i].xpath('.//b/text()').extract()[0])
        except Exception:
            pass
    for i in range(len(divs)):
        synonyme_dict[title[i]] = divs[i].xpath('.//a[contains(@href, "woxikon")]/text()').extract()
        
    return synonyme_dict


def _find_en_synonym(vocab, suffix=''):

    url='https://googledictionaryapi.eu-gb.mybluemix.net'
    response = urllib.request.urlopen(url + '/?define=' + vocab + suffix)
    return_str = response.read().decode("utf-8")
    cut_str_ind = re.search(r"\"synonym(\w+)\"", return_str).span()[1] + 2
    cut_str = return_str[cut_str_ind:]

    dirty_list = [ x for x in cut_str.split("  ")  if (',\n' in x) or ('"\n' in x) ]
    del_def = [ x for x in dirty_list if ('definition' not in x) and ('example' not in x)]
    clean_list = [ re.sub(r'([^\s\w]|_)+', '', x)[:-1] for x in del_def]

    deep_clean_list = [ x for x in clean_list if x is not '']   

    return deep_clean_list
