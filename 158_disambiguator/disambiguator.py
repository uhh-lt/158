#!/usr/bin/env python3

import configparser

from jsonrpcserver import dispatch, method
from werkzeug.serving import run_simple
from werkzeug.wrappers import Request, Response

from egvi import WSD


@method
def disambiguate(language, *tokens):
    print('Choosing language')
    if language == 'en':
        inventory = "models/cc.en.300.vec.gz.top200.inventory.tsv"
    elif language == 'ru':
        inventory = "models/cc.ru.300.vec.gz.top200.wsi-inventory.tsv"
    else:
        return 'Wrong lang'
    text = " ".join(tokens)
    print('Loading wsd-inventory')
    wsd = WSD(inventory, language=language, verbose=True)
    print('Making results')
    results = [(word, wsd.get_best_sense_id(text, word, 5)) for word in tokens]

    return results


@Request.application
def app(request):
    r = dispatch(request.data.decode())
    return Response(str(r), r.http_status, mimetype='application/json')


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('158.ini')

    run_simple('localhost', 5002, app)
