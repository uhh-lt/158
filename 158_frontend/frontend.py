#!/usr/bin/env python3

import configparser
import random
import json

import jsonrpcclient
from flask import Flask, render_template, send_from_directory, redirect, url_for, request, jsonify

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

with open("langs.json") as json_file:
    languages_dict = json.load(json_file)
    languages_values = list(languages_dict.items())


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/wsd/', methods=['POST'])
def wsd():
    text_input = request.form['text']

    tokenizer_url = random.choice(tokenizers)
    disambiguator_url = random.choice(disambiguators)

    tokenization = []
    disambiguation = []

    try:
        tokenization = jsonrpcclient.request(tokenizer_url, 'tokenize', text_input).data.result
        disambiguation = jsonrpcclient.request(disambiguator_url, 'disambiguate',
                                               tokenization['language'], tokenization['tokens']).data.result
    except Exception as e:
        # TODO: add logging
        print(e)

    result = []

    for senses in disambiguation:
        max_sense = max(senses, key=lambda sense: sense['confidence'])
        result.append(max_sense)
    return render_template('wsd.html', tokenization=tokenization, disambiguation=result)


@app.route('/word_inventory', methods=['GET'])
def word_senses():
    return render_template('word_inventory.html', langs_dict=languages_values)


@app.route('/senses/', methods=['POST'])
def senses():
    disambiguator_url = random.choice(disambiguators)

    language = request.form["selected_language"]
    word = request.form["word"]

    try:
        senses_list = jsonrpcclient.request(disambiguator_url, 'senses', language, word).data.result
    except Exception as e:
        print(e)
        senses_list = [[word, "SERVER ERROR", ["SERVER ERROR"]]]

    if len(senses_list) == 0:
        senses_list = [[word, "UNKNOWN", ["UNKNOWN"]]]

    return render_template('senses.html', word=word, senses=senses_list)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.root_path, 'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == '__main__':
    app.run()
