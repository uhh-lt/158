#!/usr/bin/env python3

import configparser
import subprocess
import tempfile

import MeCab
import icu
from jsonrpcserver import dispatch, method
from werkzeug.serving import run_simple
from werkzeug.wrappers import Request, Response

config = configparser.ConfigParser()
config.read('158.ini')
icu_langs = set(config.get('tokenizer', 'icu_langs').strip().split(','))

mecab = MeCab.Tagger("-Owakati")  # Japanese tokenizer


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
    with tempfile.NamedTemporaryFile() as f:
        f.write(text.encode('utf-8'))

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


@method
def tokenize(context, text):
    return tokenize_sentence(text.strip(), icu_langs)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('158.ini')


    @Request.application
    def app(request):
        r = dispatch(request.data.decode(), context={'config': config})
        return Response(str(r), r.http_status, mimetype='application/json')


    run_simple('0.0.0.0', 5001, app)
