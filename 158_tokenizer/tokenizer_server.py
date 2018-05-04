#!/usr/bin/python3
# coding: utf-8

import os
import socket
import threading
import datetime
import sys
import json
import configparser
import subprocess
import hashlib
import time


def tokenize(sentence):
    results = {}
    identifier = \
        subprocess.Popen(['language_identification/fasttext', 'predict', 'language_identification/lid.176.ftz', '-'],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    language = identifier.communicate(sentence)[0].strip()
    language = language.decode('utf-8').split('__')[-1]
    results['lang'] = language
    if language == 'zh':
        tokens = tokenize_chinese(sentence)
    else:
        tokenizer = \
            subprocess.Popen(['Europarl/tokenizer.perl', '-l', language], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        tokens = tokenizer.communicate(sentence)[0].strip().decode('utf-8')
    results['tokens'] = tokens.split()
    return results


def tokenize_chinese(text):
    dochash = hashlib.sha1()
    timestring = str(time.time()).encode('utf-8')
    dochash.update(timestring)
    filename = 'chinese_tokens_' + dochash.hexdigest() + '.txt'
    doc = open(filename, 'w')
    doc.write(text.decode('utf-8'))
    doc.close()
    tokenizer = \
        subprocess.Popen(['stanford_segmenter/segment.sh', 'pku', filename, 'UTF-8', '0'], stdout=subprocess.PIPE)
    tokens = tokenizer.communicate()[0].decode('utf-8')
    os.remove(filename)
    return tokens


class TokThread(threading.Thread):
    def __init__(self, connect, address):
        threading.Thread.__init__(self)
        self.connect = connect
        self.address = address

    def run(self):
        threadLimiter.acquire()
        try:
            clientthread(self.connect, self.address)
        finally:
            threadLimiter.release()


def clientthread(connect, address):
    # Function for handling connections. This will be used to create threads

    # infinite loop so that function do not terminate and thread do not end.
    while True:
        # Receiving from client
        data = connect.recv(4096)
        if not data:
            break
        queries = data.decode('utf-8').strip().split('\n')
        for query in queries:
            output = tokenize(query.encode('utf-8').strip())
            now = datetime.datetime.now()
            print(now.strftime("%Y-%m-%d %H:%M"), '\t', address[0] + ':' + str(address[1]), '\t',
                  data.decode('utf-8').strip(), file=sys.stderr)
            reply = json.dumps(output, ensure_ascii=False).encode('utf-8')
            connect.sendall(reply)
        break

    # came out of loop
    connect.close()


config = configparser.RawConfigParser()
config.read('tokenizer.cfg')

root = config.get('Files and directories', 'root')
HOST = config.get('Sockets', 'host')  # Symbolic name meaning all available interfaces
PORT = config.getint('Sockets', 'port')  # Arbitrary non-privileged port
maxthreads = config.getint('Other', 'maxthreads')  # Maximum number of threads

threadLimiter = threading.BoundedSemaphore(maxthreads)

# Bind socket to local host and port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created', file=sys.stderr)

try:
    s.bind((HOST, PORT))
except socket.error as msg:
    print('Bind failed. Error Code : ' + str(msg), file=sys.stderr)
    sys.exit()

print('Socket bind complete', file=sys.stderr)

# Start listening on socket
s.listen(100)
print('Socket now listening on port', PORT, file=sys.stderr)

# now keep talking with the client
while 1:
    # wait to accept a connection
    conn, addr = s.accept()

    # start new thread takes 1st argument as a function name to be run, 2nd is the tuple of arguments to the function.
    thread = TokThread(conn, addr)
    thread.start()
