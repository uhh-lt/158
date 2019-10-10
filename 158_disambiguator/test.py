#!/usr/bin/env python3

from jsonrpcclient import request

tokens = ['I', 'wrote', 'my', 'program', 'in', 'Python', '.']
language = 'en'

result = request('http://localhost:5002/', 'disambiguate', language, tokens).data.result
print(result)
