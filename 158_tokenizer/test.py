#!/usr/bin/env python3

from jsonrpcclient import request

text = 'I wrote my program in Python.'
result = request('http://localhost:5001/', 'tokenize', text=text).data.result
print(result)
