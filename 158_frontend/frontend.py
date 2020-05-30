#!/usr/bin/env python3

import configparser
import random
import json
import requests
import os

from flask import Flask, render_template, send_from_directory, redirect, url_for, request, current_app as app

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

json_headers = {'Content-type': 'application/json'}

app = Flask(__name__)
app.url_map.strict_slashes = False

frontend_assets.init(app)

with open("langs.json") as json_file:
    languages_dict = json.load(json_file)
    languages_values = list(languages_dict.items())


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', langs_dict=languages_values)


@app.route('/uwsd158')
def wsd_redirect():
    return redirect(url_for('.index'), code=302)


def disambiguate_text(input_text: str, tokenizer_url: str, disambiguator_url: str, chosen_language: str = None):
    # tokenization
    text_data = {"text": input_text}
    tokenization_req = requests.post(tokenizer_url, data=json.dumps(text_data), headers=json_headers)
    tokenized_data = tokenization_req.json()

    if chosen_language:
        tokenized_data["language"] = chosen_language

    # disambiguation
    disambiguation_req = requests.post(disambiguator_url, data=json.dumps(tokenized_data), headers=json_headers)
    disambiguation = disambiguation_req.json()

    return tokenized_data, disambiguation


@app.route('/wsd', methods=['POST'])
def wsd():
    text_input = request.form['text']
    if 'known_language' in request.form:
        chosen_lang = request.form['selected_language_main']
    else:
        chosen_lang = None
    if 'dis_paragraph' in request.form:
        disambiguate_by_paragraph = True
    else:
        disambiguate_by_paragraph = False

    tokenizer_url = random.choice(tokenizers)
    disambiguator_url = os.path.join(random.choice(disambiguators), "disambiguate")

    disambiguation = []

    if disambiguate_by_paragraph:
        found_languages = []
        paragraphs = text_input.splitlines()

        for paragraph in paragraphs:
            if paragraph.strip() == "":
                continue
            tokenization_par, disambiguation_par = disambiguate_text(paragraph, tokenizer_url, disambiguator_url)
            found_languages.append(tokenization_par["language"])
            disambiguation.extend(disambiguation_par)

            # Add line for a new paragraph
            disambiguation.extend("\n")

        if chosen_lang:
            output_language = chosen_lang
        else:
            # Check if more than 1 languages were found
            # If so - return an error
            found_languages_set = set(found_languages)
            if len(found_languages_set) == 1:
                output_language = list(found_languages_set)[0]
            else:
                output_message = "Couldn't choose one language"
                print(output_message)
                disambiguation = []
                output_language = None

    else:
        tokenization, disambiguation = disambiguate_text(text_input, tokenizer_url, disambiguator_url, chosen_lang)
        output_language = tokenization["language"]

    result = []

    for result_senses in disambiguation:

        # Treat new paragraph
        if result_senses == "\n":
            result.append("\n")
            continue

        max_sense = max(result_senses, key=lambda sense: sense['confidence'])
        result.append(max_sense)
    return render_template('wsd.html', output_language=output_language, disambiguation=result)


@app.route('/word_inventory', methods=['GET'])
def word_senses():
    return render_template('word_inventory.html', langs_dict=languages_values)


@app.route('/senses', methods=['POST'])
def senses():
    disambiguator_url = os.path.join(random.choice(disambiguators), "senses")

    language = request.form["selected_language"]
    word = request.form["word"].strip().lower()

    data = {"language": language,
            "word": word}

    try:
        answer = requests.post(disambiguator_url, data=json.dumps(data), headers=json_headers)
        senses_list = answer.json()
    except Exception as e:
        print(e)
        senses_list = [[word, "SERVER ERROR", ["SERVER ERROR"]]]

    if len(senses_list) == 0:
        senses_list = [[word, "UNKNOWN WORD", ["UNKNOWN WORD"]]]

    return render_template('senses.html', word=word, senses=senses_list, language=language)


@app.route('/plots/<lang>/<word>')
def send_pdf(lang, word):
    fpath = "./plots/{lang}/".format(lang=lang)
    filename = '{word}.pdf'.format(word=word.lower())
    return send_from_directory(fpath, filename)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.root_path, 'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == '__main__':
    app.run()
