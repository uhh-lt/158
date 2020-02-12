#!/usr/bin/env python3

import configparser
import sys

from egvi_sqlite import WSD
from jsonrpcserver import dispatch, method
from werkzeug.wrappers import Request, Response
from flask import Flask, request, Response

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


@method
def disambiguate(context, language, tokens):
    # Different library versions pass variable in different ways
    if tokens[0] is list:
        tokens = tokens[0]

    if language in language_list:
        wsd = wsd_dict[language]
    else:
        wsd = None
        print('Error: unknown language: {}'.format(language))

    results = wsd.disambiguate_text(tokens)

    return results


@method
def senses(context, language, word):
    # Different library versions pass variable in different ways
    if word is list:
        word = word[0]

    if language in language_list:
        wsd = wsd_dict[language]
    else:
        wsd = None
        print('Error: unknown language: {}'.format(language))

    senses_result = wsd.get_senses(word)

    return senses_result


@app.route("/", methods=['GET', 'POST'])
def index():
    req = request.get_data().decode()
    response = dispatch(req, context={'config': config})
    return Response(str(response), response.http_status, mimetype="application/json")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
