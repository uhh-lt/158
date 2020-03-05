# python3
# coding: utf-8

import sys
import pandas as pd
import random
import requests
import json

options = [1, 2, 3, 4, 5]
json_headers = {'Content-type': 'application/json'}
tokenizer_url = 'http://ltdemos.informatik.uni-hamburg.de/uwsd1580-tokenize'
disambiguator_url = 'http://ltdemos.informatik.uni-hamburg.de/uwsd158-api/disambiguate'


def disambiguate(word, text, lemmas=True):
    if not lemmas:
        question = requests.post(tokenizer_url, data=json.dumps({"text": text}), headers=json_headers)
        tokenized = question.json()
    else:
        tokenized = {'language': 'en', 'tokens': text.split()}
    print('Sentence length:', len(tokenized['tokens']))
    if word not in tokenized['tokens']:
        print('WORD NOT FOUND!', text)
        return None, None, None
    token_nr = tokenized['tokens'].index(word)
    print(tokenized)
    disambiguation_req = requests.post(disambiguator_url, data=json.dumps(tokenized), headers=json_headers)
    # print(disambiguation_req)
    try:
        disambiguation = disambiguation_req.json()
    except:
        print('ERROR!', text)
        return None, None, None
    senses = disambiguation[token_nr]
    terms = None
    keyword = None
    confidence = 0
    for sense in senses:
        if sense['confidence'] > confidence:
            terms = sense['cluster']
            keyword = sense['keyword']
            confidence = sense['confidence']
    return keyword, confidence, terms

LEMMAS = True
FIX = True

datasetfile = sys.argv[1]
dataset = pd.read_csv(datasetfile, sep='\t')
dataset['predict_sense_ids'] = dataset['predict_sense_ids'].astype(str)
dataset['predict_related'] = dataset['predict_related'].astype(str)

errors = 0
for row in dataset.itertuples():
    if FIX:
        if row.predict_sense_ids != 'nan':
            continue
    if not LEMMAS:
        position = row.target_position
        positions = position.split(',')
        positions = [int(p) for p in positions]
        assert len(positions) == 2
        target_word = row.context[positions[0]:positions[1]]
        context = row.context
        keyword, confidence, related_terms = disambiguate(target_word, context, lemmas=False)
    else:
        target_word = row.target
        context = row.lemmas
        keyword, confidence, related_terms = disambiguate(target_word, context)
    if not keyword:
        errors += 1
        continue
    print('Predicted', keyword, 'for', target_word, 'in', '\t', row.context)
    dataset.at[row.Index, 'predict_sense_ids'] = keyword
    dataset.at[row.Index, 'predict_related'] = ','.join(related_terms)

print(dataset.head())
print('Errors:', errors)

dataset.to_csv(sys.argv[2], sep='\t', index=False)
