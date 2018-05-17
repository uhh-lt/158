#!/usr/bin/env python3

import configparser
import random

import jsonrpcclient
from flask import Flask, render_template, send_from_directory, redirect, url_for, request

config = configparser.ConfigParser()
config.read('158.ini')

if 'services' not in config:
    config['services'] = {}

if 'tokenizer' not in config['services']:
    config['services']['tokenizer'] = 'http://localhost:5001'

if 'disambiguator' not in config['services']:
    config['services']['disambiguator'] = 'http://localhost:5002'

tokenizers = [url for url in config['services']['tokenizer'].split('\n') if url]
disambiguators = [url for url in config['services']['disambiguator'].split('\n') if url]

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/wsd')
def wsd_redirect():
    return redirect(url_for('.index'), code=302)


@app.route('/wsd', methods=['POST'])
def wsd():
    tokenizer_url = random.choice(tokenizers)
    tokenization = jsonrpcclient.request(tokenizer_url, 'tokenize', request.form['text'])
    return render_template('wsd.html', tokenization=tokenization)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.root_path, 'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == '__main__':
    app.run()
