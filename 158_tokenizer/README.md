# Tokenizer

`tokenizer_server.py` runs in the background and listens on the port stated in `tokenizer.cfg`.

It accepts raw utf-8 text lines and detects the language of each line using the fastText model in the `language_identification` folder.

Then the line is tokenized with the preferred tokenizer for this language.

The detected language and the list of tokens are returned as a JSON-serialized dictionary.

# Usage

Run the server with:

`python3 tokenizer_server.py`

Then a command like that:

`echo 'Norsk er et vanskelig språk.' | nc localhost 42420`

will produce the JSON:

`{"tokens": ["Norsk", "er", "et", "vanskelig", "språk", "."], "lang": "nn"}`


# Dependencies

- For Chinese tokenizer to work, the files `dict-chris6.ser.gz` and `pku.gz` must be placed in the `stanford_segmenter/data` directory.
They can be downloaded at https://nlp.stanford.edu/software/stanford-segmenter-2018-02-27.zip
- For Vietnamese tokenizer to work, the model files must be placed in the `UETSegmenter/models` directory.
They can be downloaded at https://github.com/phongnt570/UETsegmenter/tree/master/models
- For Japanese tokenizer to work, MeCab must be installed system-wide:
`apt install mecab libmecab-dev mecab-ipadic-utf8`. Then, the `mecab-python3` module for Python must be installed.
- See also the `requirements.txt` for the required Python modules.

