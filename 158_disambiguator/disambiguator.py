#!/usr/bin/env python3

import configparser

from jsonrpcserver import methods
from werkzeug.serving import run_simple
from werkzeug.wrappers import Request, Response

config = configparser.ConfigParser()
config.read('158.ini')


@methods.add
def disambiguate(language, tokens):
    pass


@Request.application
def app(request):
    r = methods.dispatch(request.data.decode())
    return Response(str(r), r.http_status, mimetype='application/json')


if __name__ == '__main__':
    run_simple('localhost', 5000, app)
