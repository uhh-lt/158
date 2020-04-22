#!/usr/bin/env python3

import configparser
import sys
import os

from flask import Flask, request, jsonify
from flasgger import Swagger

from egvi_psql import WSD as WSDPSQL
# from egvi_sqlite import WSD as WSDSQL
from egvi import WSD as WSDGensim

INVENTORY_TOP = 200
DICTIONARY_SIZE = 100

# sqlite_db = "./models/Vectors.db"
# inventory_db = "./models/Inventory.db"

PSQL_USER = "158_user"
PSQL_PASSWORD = "158"
PSQL_DB_VECTORS = "fasttext_vectors"
PSQL_DB_INVENTORIES = "inventory"
PSQL_HOST = "database"
PSQL_PORT = "5432"

config = configparser.ConfigParser()
config.read('158.ini')
language_list_sql = config['disambiguator']['sql_langs'].split(',')
language_list_gensim = config['disambiguator']['top_langs'].split(',')

wsd_top_dict = dict()
print("Start with top languages", file=sys.stderr)
for language in language_list_gensim:
    print('WSD[%s] model start' % language, file=sys.stderr)
    dir_path = os.path.join("models", "inventories", language)
    inventory_file = "cc.{lang}.300.vec.gz.top{top}.inventory.tsv".format(lang=language, top=INVENTORY_TOP)
    inventory_fpath = os.path.join(dir_path, inventory_file)
    try:
        wsd_top_dict[language] = WSDGensim(inventory_fpath=inventory_fpath,
                                           language=language,
                                           verbose=False,
                                           skip_unknown_words=True,
                                           dictionary=DICTIONARY_SIZE)
    except Exception as e:
        print('ERROR WSD[{lang}] model: {error}'.format(lang=language, error=e), file=sys.stderr)
    else:
        print('WSD[%s] model loaded successfully' % language, file=sys.stderr)

print("Non top languages", file=sys.stderr)
print("Connect to PSQL server")
try:
    wsd_nontop = WSDPSQL(db_vectors=PSQL_DB_VECTORS, db_inventory=PSQL_DB_INVENTORIES,
                         user=PSQL_USER, password=PSQL_PASSWORD,
                         host=PSQL_HOST, port=PSQL_PORT)
except Exception as e:
    print(e, file=sys.stderr)
else:
    print("Connection succeed")

# for language in language_list_sql:
#     print('WSD[%s] model start' % language, file=sys.stderr)
#     try:
#         wsd_dict[language] = WSDSQL(inventories_db_fpath=inventory_db,
#                                     vectors_db_fpath=sqlite_db,
#                                     language=language,
#                                     verbose=True)
#     except Exception as e:
#         print('ERROR WSD[{lang}] model: {error}'.format(lang=language, error=e), file=sys.stderr)
#     else:
#         print('WSD[%s] model loaded successfully' % language, file=sys.stderr)

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

    if req_language in language_list_gensim:
        wsd = wsd_top_dict[req_language]
        senses_list = wsd.disambiguate_text(tokens)
    elif req_language in language_list_sql:
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

    if req_language in language_list_gensim:
        wsd = wsd_top_dict[req_language]
        word_senses = wsd.get_senses(word)
    elif req_language in language_list_sql:
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
