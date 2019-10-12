#!/usr/bin/env python3

from jsonrpcclient import request

language_en = 'en'
tokens_en = ['I', 'wrote', 'my', 'program', 'in', 'Python', '.']

result_en = request('http://localhost:5002/', 'disambiguate', language_en, tokens_en).data.result
print(result_en)

language_ru = 'ru'
tokens_ru = ['пытка', 'дезертир']

result_ru = request('http://localhost:5002/', 'disambiguate', language_ru, tokens_ru).data.result
print(result_ru)
