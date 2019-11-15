#!/usr/bin/env python3

from jsonrpcclient import request

text = '我的人这年会'

result = request('http://localhost:10151/', 'tokenize', text).data.result
print(result)
