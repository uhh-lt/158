#!/usr/bin/env python3

import configparser
import sys
import os

from egvi import WSD
from jsonrpcserver import dispatch, method
from werkzeug.serving import run_simple
from werkzeug.wrappers import Request, Response
from flask import Flask, request, Response


config = configparser.ConfigParser()
config.read('158.ini')

language = os.environ.get('LANGUAGE', 'en')

try:
    inventory = config.get('models', language.lower())
except:
    raise Exception('No language available')


wsd = WSD(inventory, language=language, verbose=True)
print('WSD[%s] model loaded successfully' % language, file=sys.stderr)

app = Flask(__name__)


@method
def disambiguate(context, *tokens):
    wsd = context['wsd']
    
    results = list()
    
    for token in tokens:
        token_sense = list()
        senses = wsd.disambiguate(tokens, token, 5)
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
    response = dispatch(req, context={'config': config, 'wsd': wsd})
    return Response(str(response), response.http_status, mimetype="application/json")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
