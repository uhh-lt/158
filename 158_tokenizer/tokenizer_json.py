#!/usr/bin/env python3

import configparser
import subprocess
import tempfile
import codecs

import MeCab
import icu
from werkzeug.serving import run_simple
from werkzeug.wrappers import Request, Response
from flask import Flask, request, jsonify

config = configparser.ConfigParser()
config.read('158.ini')
icu_langs = set(config.get('tokenizer', 'icu_langs').strip().split(','))

mecab = MeCab.Tagger("-Owakati")  # Japanese tokenizer

app = Flask(__name__)


def tokenize_sentence(text, exotic_langs):
    with subprocess.Popen(['language_identification/fasttext', 'predict', 'language_identification/lid.176.ftz', '-'],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE) as identifier:
        language = identifier.communicate(text.encode('utf-8'))[0]

    language = language.decode('utf-8').strip().split('__')[-1]

    if language == 'zh':
        tokens = tokenize_chinese(text)
    elif language == 'vi':
        tokens = tokenize_vietnamese(text)
    elif language == 'ja':
        tokens = tokenize_japanese(text)
    elif language in exotic_langs:
        tokens = tokenize_icu(text, language)
    else:
        with subprocess.Popen(['Europarl/tokenizer.perl', '-l', language], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE) as tokenizer:
            tokens = tokenizer.communicate(text.encode('utf-8'))[0]

        tokens = tokens.decode('utf-8').strip()

    return {'language': language, 'tokens': tokens.split()}


def tokenize_chinese(text):

    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write(text)
        f.flush()

        with subprocess.Popen(['stanford_segmenter/segment.sh', 'pku', f.name, 'UTF-8', '0'],
                              stdout=subprocess.PIPE) as tokenizer:
            tokens = tokenizer.communicate()[0].decode('utf-8')

    return tokens


def tokenize_japanese(text):
    tokenized = mecab.parse(text)
    return tokenized.strip()


def tokenize_vietnamese(text):
    with tempfile.NamedTemporaryFile() as fr, tempfile.NamedTemporaryFile('r') as fw:
        fr.write(text.encode('utf-8'))
        fr.flush()

        subprocess.call(
            ['java', '-jar', 'UETSegmenter/uetsegmenter.jar', '-r', 'seg', '-m', 'UETSegmenter/models/',
             '-i', fr.name, '-o', fw.name])

        fw.seek(0)

        tokens = fw.read()

    return tokens


def tokenize_icu(text, language):
    bd = icu.BreakIterator.createWordInstance(icu.Locale(language))
    bd.setText(text)
    start_pos = 0
    tokens = ''
    for obj in bd:
        tokens += text[start_pos:obj]
        tokens += ' '
        start_pos = obj
    return tokens


@app.route("/", methods=['POST'])
def tokenize():
    req_json = request.json
    text = req_json['text']
    result = tokenize_sentence(text.strip(), icu_langs)

    results_json = jsonify(result)
    return results_json


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
