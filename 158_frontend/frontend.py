#!/usr/bin/env python3

import configparser
import random

import jsonrpcclient
from flask import Flask, render_template, send_from_directory, redirect, url_for, request

import frontend_assets

config = configparser.ConfigParser()
config.read('158.ini')

if 'services' not in config:
    config['services'] = {}

if 'tokenizer' not in config['services']:
    config['services']['tokenizer'] = 'http://localhost:5001'

if 'disambiguator' not in config['services']:
    config['services']['disambiguator'] = 'http://localhost:5002'

tokenizers = [url for url in config['services']['tokenizer'].split('\n') if url]
print(tokenizers)

disambiguators = [url for url in config['services']['disambiguator'].split('\n') if url]
print(disambiguators)

app = Flask(__name__)

frontend_assets.init(app)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/uwsd158')
def wsd_redirect():
    return redirect(url_for('.index'), code=302)


@app.route('/uwsd158/', methods=['POST'])
def wsd():
    tokenizer_url = random.choice(tokenizers)
    tokenization = jsonrpcclient.request(tokenizer_url, 'tokenize', request.form['text']).data.result

    disambiguator_url = random.choice(disambiguators)
    disambiguation = jsonrpcclient.request(disambiguator_url, 'disambiguate', tokenization['language'],
                                           tokenization['tokens']).data.result

    result = []

    for senses in disambiguation:
        max_sense = max(senses, key=lambda sense: sense['confidence'])
        result.append(max_sense)
    return render_template('wsd.html', tokenization=tokenization, disambiguation=result)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.root_path, 'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == '__main__':
    app.run()
