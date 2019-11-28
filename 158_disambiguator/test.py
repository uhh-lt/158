#!/usr/bin/env python3

from jsonrpcclient import request

language_en = 'en'

word_en = 'Ruby'
senses_en = request('http://localhost:5002/', 'senses', language_en, word_en).data.result
print(senses_en)

tokens_en = ['Ruby', 'Perl']
result_en = request('http://localhost:5002/', 'disambiguate', language_en, tokens_en).data.result
print(result_en)

language_ru = 'ru'
tokens_ru = ['щенок', 'такса']

result_ru = request('http://localhost:5002/', 'disambiguate', language_ru, tokens_ru).data.result
print(result_ru)
