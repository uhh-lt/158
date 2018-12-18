#!/usr/bin/env python3

import configparser
import sys
import os

from egvi import WSD
from jsonrpcserver import dispatch, method
from werkzeug.serving import run_simple
from werkzeug.wrappers import Request, Response


@method
def disambiguate(context, *tokens):
    wsd = context['wsd']

    results = list(tokens)

    for i, token in enumerate(results):
        sense_id, confidence = wsd.get_best_sense_id(tokens, token, 5)
        # confidence is NumPy's float32, but not float
        results[i] = (token, sense_id, float(confidence))

    return results


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('158.ini')

    language = os.environ.get('LANGUAGE', 'en')

    if 'ru' == language:
        inventory = 'models/cc.ru.300.vec.gz.top200.wsi-inventory.tsv'
    elif 'en' == language:
        inventory = 'models/cc.en.300.vec.gz.top200.inventory.tsv'
    else:
        raise Exception('No language available')

    wsd = WSD(inventory, language=language, verbose=True)
    print('WSD[%s] model loaded successfully' % language, file=sys.stderr)


    @Request.application
    def app(request):
        r = dispatch(request.data.decode(), context={'config': config, 'wsd': wsd})
        return Response(str(r), r.http_status, mimetype='application/json')


    run_simple('localhost', 5002, app)
