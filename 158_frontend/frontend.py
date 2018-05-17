#!/usr/bin/env python3

from flask import Flask, render_template, send_from_directory, redirect, url_for, request
import jsonrpcclient

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/wsd')
def wsd_redirect():
    return redirect(url_for('.index'), code=302)


@app.route('/wsd', methods=['POST'])
def wsd():
    tokenization = jsonrpcclient.request('http://localhost:5001', 'tokenize', request.form['text'])
    return render_template('wsd.html', tokenization=tokenization)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.root_path, 'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == '__main__':
    app.run()
