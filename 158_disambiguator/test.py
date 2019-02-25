#!/usr/bin/env python3

from jsonrpcclient import request

tokens = ['I', 'wrote', 'my', 'program', 'in', 'Python', '.']

result = request('http://localhost:5002/', 'disambiguate', tokens).data.result
print(result)
