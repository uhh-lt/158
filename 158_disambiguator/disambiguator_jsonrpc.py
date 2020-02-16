#!/usr/bin/env python3

import configparser
import sys

from egvi_sqlite import WSD
from flask import Flask, request, jsonify

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


def sense_to_dict(sense):
    return {"word": sense.word,
            "keyword": sense.keyword,
            "cluster": sense.cluster}


@app.route("/disambiguate", methods=['POST'])
def disambiguate():
    req_json = request.json
    language = req_json['language']
    tokens = req_json['tokens']

    if language in language_list:
        wsd = wsd_dict[language]
        senses_list = wsd.disambiguate_text(tokens)
    else:
        senses_list = None
        print('Error: unknown language: {}'.format(language))

    results_json = jsonify(senses_list)
    return results_json


@app.route("/senses", methods=['POST'])
def senses():
    req_json = request.json
    language = req_json['language']
    word = req_json['word']

    if language in language_list:
        wsd = wsd_dict[language]
        word_senses = wsd.get_senses(word)
    else:
        word_senses = None
        print('Error: unknown language: {}'.format(language))

    results = []
    for sense in word_senses:
        results_dict = sense_to_dict(sense)
        results.append(results_dict)

    results_json = jsonify(results)
    return results_json


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
