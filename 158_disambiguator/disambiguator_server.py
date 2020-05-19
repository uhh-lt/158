#!/usr/bin/env python3

import configparser
import sys
import os

from flask import Flask, request, jsonify
from flasgger import Swagger

from egvi import WSDGensim, WSDPSQL

config = configparser.ConfigParser()
config.read('158.ini')

LANGUAGES_SQL = config['disambiguator']['sql_langs'].split(',')
LANGUAGES_GENSIM = config['disambiguator']['top_langs'].split(',')

PSQL_USER = config['postgress']['user']
PSQL_PASSWORD = config['postgress']['password']
PSQL_DB_VECTORS = config['postgress']['vectors_db']
PSQL_DB_INVENTORIES = config['postgress']['inventories_db']
PSQL_HOST = config['postgress']['host']
PSQL_PORT = config['postgress']['port']

INVENTORY_FILE_FORMAT = config['disambiguator']['inventory_file_format']
INVENTORIES_FPATH = config['disambiguator']['inventories_fpath']
INVENTORY_TOP = int(config['disambiguator']['inventory_top'])
DICTIONARY_SIZE = int(config['disambiguator']['dict_size'])


def load_gensim(lang_list, inventories_fpath, inventory_file_format, inventory_top, dict_size):
    wsd_gensim_dict = {}
    print("Start with top languages", file=sys.stderr)
    for language in lang_list:
        print('WSD[%s] model start' % language, file=sys.stderr)
        dir_path = os.path.join(inventories_fpath, language)
        inventory_file = inventory_file_format.format(lang=language, top=inventory_top)
        inventory_fpath = os.path.join(dir_path, inventory_file)
        try:
            wsd_gensim_dict[language] = WSDGensim(inventory_fpath=inventory_fpath,
                                                  language=language,
                                                  verbose=False,
                                                  skip_unknown_words=True,
                                                  dictionary=dict_size)
        except Exception as e:
            print('ERROR WSD[{lang}] model: {error}'.format(lang=language, error=e), file=sys.stderr)
        else:
            print('WSD[%s] model loaded successfully' % language, file=sys.stderr)
    return wsd_gensim_dict


wsd_gensim_dict = load_gensim(lang_list=LANGUAGES_GENSIM,
                              inventories_fpath=INVENTORIES_FPATH,
                              inventory_file_format=INVENTORY_FILE_FORMAT,
                              inventory_top=INVENTORY_TOP,
                              dict_size=DICTIONARY_SIZE)

print("Non top languages", file=sys.stderr)
print("Connect to PSQL server")
try:
    wsd_nontop = WSDPSQL(db_vectors=PSQL_DB_VECTORS,
                         db_inventory=PSQL_DB_INVENTORIES,
                         user=PSQL_USER,
                         password=PSQL_PASSWORD,
                         host=PSQL_HOST,
                         port=PSQL_PORT)
except Exception as e:
    print(e, file=sys.stderr)
else:
    print("Connection succeed")

app = Flask(__name__)

swagger = Swagger(app)


def sense_to_dict(sense):
    return {"word": sense.word,
            "keyword": sense.keyword,
            "cluster": sense.cluster}


@app.route("/disambiguate", methods=['POST'])
def disambiguate():
    """
        Disambiguates whole context
        Call this api passing a language name and a list of tokens
        ---
        tags:
          - 158 disambiguator
        parameters:
          - name: language
            in: body
            type: string
            required: true
            description: The language name
          - name: tokens
            in: body
            type: array
            items:
              type: string
            required: true
            description: list of context tokens to disambiguate
        responses:
          500:
            description: Bad request
          200:
            description: A list of words, each is a list of possible senses.
            schema:
              type: array
              items:
                type: array
                items:
                  type: object
                  properties:
                      cluster:
                        type: array
                        description: Tokens in the sense cluster.
                        items:
                          type: string
                      confidence:
                        type: number
                        description: Confidence of the sense.
                      keyword:
                        type: string
                        description: The centroid of the sense.
                      token:
                        type: string
                        description: Token from the request.
                      word:
                        type: string
                        description: Identified token from the request.
        """

    if request.is_json:
        req_json = request.json
    else:
        raise Exception("Request is not json")

    req_language = req_json['language']
    tokens = req_json['tokens']

    print("Language: {lang}\nTokens: {tokens}".format(lang=req_language, tokens=tokens))

    if req_language in LANGUAGES_GENSIM:
        wsd = wsd_gensim_dict[req_language]
        senses_list = wsd.disambiguate_text(tokens)
    elif req_language in LANGUAGES_SQL:
        senses_list = wsd_nontop.disambiguate_text(tokens, language=req_language)
    else:
        raise Exception("Unknown language: {}".format(req_language))

    print("Results \nLang: {lang}\nTokens: {tokens}\n--------------\n{senses}".format(lang=req_language,
                                                                                      tokens=tokens,
                                                                                      senses=senses_list))

    results_json = jsonify(senses_list)
    return results_json


@app.route("/senses", methods=['POST'])
def senses():
    """
        Returns all known senses for token
        Call this api passing a language name and a token
        ---
        tags:
          - 158 disambiguator
        parameters:
          - name: language
            in: body
            type: string
            required: true
            description: The language name
          - name: word
            in: body
            type: string
            description: token to get senses
        responses:
          500:
            description: Bad Request
          200:
            description: list of known word senses
            schema:
              type: array
              items:
                type: object
                properties:
                  cluster:
                    type: array
                    description: Tokens in the sense cluster.
                    items:
                      type: string
                  keyword:
                    type: string
                    description: The centroid of the sense.
                  word:
                    type: string
                    description: Token from the request.
        """
    if request.is_json:
        req_json = request.json
    else:
        raise Exception("Request is not json")

    req_language = req_json['language']
    word = req_json['word'].strip()

    if req_language in LANGUAGES_GENSIM:
        wsd = wsd_gensim_dict[req_language]
        word_senses = wsd.get_senses(word)
    elif req_language in LANGUAGES_SQL:
        word_senses = wsd_nontop.get_senses(word, language=req_language)
    else:
        raise Exception("Unknown language: {}".format(req_language))

    results = []
    for sense in word_senses:
        results_dict = sense_to_dict(sense)
        results.append(results_dict)

    results_json = jsonify(results)
    return results_json


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
