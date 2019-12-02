#!/usr/bin/env python3

from jsonrpcclient import request

text = 'I wrote my program in Python.'

tokenization = request('http://localhost:10151/', 'tokenize', text).data.result
print((tokenization['language'], tokenization['tokens']))

disambiguation = request('http://localhost:10152/', 'disambiguate', tokenization['tokens']).data.result
for senses in disambiguation:
    sense = max(senses, key=lambda sense: sense['confidence'])
    print('{}\t{}\t{:.2f}\t{}'.format(sense['token'], sense['keyword'], sense['confidence'],
                                      ', '.join(sense['cluster']) if sense['cluster'] else ''))
