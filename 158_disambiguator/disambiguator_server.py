#!/usr/bin/env python3

import configparser
import sys

from egvi_sqlite import WSD
from flask import Flask, request, jsonify
from flasgger import Swagger

INVENTORY_TOP = 200
sqlite_db = "./models/Vectors.db"
inventory_db = "./models/Inventory.db"

config = configparser.ConfigParser()
config.read('158.ini')
language_list = config['disambiguator']['dis_langs'].split(',')

wsd_dict = dict()
for language in language_list:
    print('WSD[%s] model start' % language, file=sys.stderr)
    try:
        wsd_dict[language] = WSD(inventories_db_fpath=inventory_db,
                                 vectors_db_fpath=sqlite_db,
                                 language=language,
                                 verbose=True)
    except Exception as e:
        print('ERROR WSD[{lang}] model: {error}'.format(lang=language, error=e), file=sys.stderr)
    else:
        print('WSD[%s] model loaded successfully' % language, file=sys.stderr)

app = Flask(__name__)

swagger_config = {
    "headers": [
    ],
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/apispec_1.json',
            "rule_filter": lambda rule: True,  # all in
            "model_filter": lambda tag: True,  # all in
        }
    ],
    "static_url_path": "/flasgger_static",
    # "static_folder": "static",  # must be set by user
    "swagger_ui": True,
    "specs_route": "/"
}

swagger = Swagger(app, config=swagger_config)


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

    language = req_json['language']
    tokens = req_json['tokens']

    print("Language: {lang}\nTokens: {tokens}".format(lang=language, tokens=tokens))

    if language in language_list:
        wsd = wsd_dict[language]
        senses_list = wsd.disambiguate_text(tokens)
    else:
        raise Exception("Unknown language: {}".format(language))

    print("Results \nLang: {lang}\nTokens: {tokens}\n--------------\n{senses}".format(lang=language,
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

    language = req_json['language']
    word = req_json['word']

    if language in language_list:
        wsd = wsd_dict[language]
        word_senses = wsd.get_senses(word)
    else:
        raise Exception("Unknown language: {}".format(language))

    results = []
    for sense in word_senses:
        results_dict = sense_to_dict(sense)
        results.append(results_dict)

    results_json = jsonify(results)
    return results_json


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
