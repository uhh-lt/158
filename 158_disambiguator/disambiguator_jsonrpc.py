#!/usr/bin/env python3

import configparser
import sys
import logging

from egvi_sqlite import WSD
from jsonrpcserver import dispatch, method
from werkzeug.wrappers import Request, Response
from flask import Flask, request, Response

logging.basicConfig(filename="disambiguator.log", level=logging.INFO, filemode='w')

INVENTORY_TOP = 200
sqlite_db = "./models/Vectors.db"
inventory_db = "./models/Inventory.db"

config = configparser.ConfigParser()
config.read('158.ini')
language_list = config['disambiguator']['dis_langs'].split(',')

wsd_dict = dict()
for language in language_list:
    print('WSD[%s] model start' % language, file=sys.stderr)
    logging.info('WSD[%s] model start' % language)
    wsd_dict[language] = WSD(inventories_db_fpath=inventory_db, vectors_db_fpath=sqlite_db,
                             language=language, verbose=True)
    print('WSD[%s] model loaded successfully' % language, file=sys.stderr)
    logging.info('WSD[%s] model loaded successfully' % language)

app = Flask(__name__)


@method
def disambiguate(context, language, *tokens):
    # Different library versions pass variable in different ways
    if type(tokens[0]) is list:
        tokens = tokens[0]

    results = list()

    if language in language_list:
        wsd = wsd_dict[language]
    else:
        wsd = None
        print('Error: unknown language: {}'.format(language))

    for token in tokens:
        token_sense = list()

        if wsd is not None:
            senses = wsd.disambiguate(tokens, token, 5)
        else:
            senses = None

        # Could be no senses at all
        if senses is None:
            sense_dict = {"token": token,
                          "word": "UNKNOWN",
                          "keyword": "UNKNOWN",
                          "cluster": [],
                          "confidence": 1.0
                          }
            token_sense.append(sense_dict)
        else:
            for sense in senses:
                sense_dict = {"token": token,
                              "word": sense[0].word,
                              "keyword": sense[0].keyword,
                              "cluster": sense[0].cluster,
                              "confidence": sense[1]
                              }
                token_sense.append(sense_dict)
        results.append(token_sense)

    return results


@app.route("/", methods=['GET', 'POST'])
def index():
    req = request.get_data().decode()
    response = dispatch(req, context={'config': config})
    return Response(str(response), response.http_status, mimetype="application/json")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
