#!/bin/bash

# Detects the language of each stdin line.
# Then tokenizes it and outputs space-separated tokens to stdout.

# Example usage:
# echo 'Norsk er et vanskelig språk.' | ./158_tokenizer.sh > tokenized.txt
# ./158_tokenizer.sh < text.txt > tokenized.txt

while read x
do
    DET_LANG=`echo ${x} | language_identification/fasttext predict language_identification/lid.176.ftz -`
    echo ${x} | Europarl/tokenizer.perl -l ${DET_LANG##*__}
done


