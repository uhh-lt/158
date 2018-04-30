#!/usr/bin/python3
# coding: utf-8

import socket
from _thread import *
import datetime
import sys
import json
import configparser
import subprocess


def tokenize(sentence):
    results = {}
    identifier = \
        subprocess.Popen(['language_identification/fasttext', 'predict', 'language_identification/lid.176.ftz', '-'],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    language = identifier.communicate(sentence.strip())[0].strip()
    language = language.decode('utf-8').split('__')[-1]
    results['lang'] = language
    return results


def clientthread(connect, addres):
    # Function for handling connections. This will be used to create threads

    # infinite loop so that function do not terminate and thread do not end.
    while True:
        # Receiving from client
        data = connect.recv(2048)
        if not data:
            break
        query = data
        output = tokenize(query)
        now = datetime.datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M"), '\t', addres[0] + ':' + str(addres[1]), '\t', data.decode('utf-8'),
              file=sys.stderr)
        # print(output, file=sys.stderr)
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

# Bind socket to local host and port

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created', file=sys.stderr)

try:
    s.bind((HOST, PORT))
except socket.error as msg:
    print('Bind failed. Error Code : ' + msg, file=sys.stderr)
    sys.exit()

print('Socket bind complete', file=sys.stderr)

# Start listening on socket
s.listen(100)
print('Socket now listening on port', PORT, file=sys.stderr)

# now keep talking with the client
while 1:
    # wait to accept a connection - blocking call
    conn, addr = s.accept()

    # start new thread takes 1st argument as a function name to be run, 2nd is the tuple of arguments to the function.
    start_new_thread(clientthread, (conn, addr))
