FROM python:3

# FIXME: Java needed

ENV FASTTEXT 0.1.0

WORKDIR /usr/src/app

COPY . /usr/src/app

RUN \
apt-get update && \
apt-get install --no-install-recommends -y -o Dpkg::Options::="--force-confold" build-essential libicu-dev mecab libmecab-dev mecab-ipadic-utf8 && \
apt-get clean && \
rm -rf /var/lib/apt/lists/* && \
pip install -r requirements.txt && \
wget https://github.com/facebookresearch/fastText/archive/v$FASTTEXT.tar.gz && \
tar zxf v$FASTTEXT.tar.gz && \
(cd fastText-$FASTTEXT && make && mv -fv fasttext /usr/src/app/language_identification/fasttext) && \
rm -rf fastText-$FASTTEXT v$FASTTEXT.tar.gz

CMD ["uwsgi", "--http", "0.0.0.0:5000", "--master", "--module", "tokenizer_jsonrpc:application", "--processes", "4", "--threads", "1", "--harakiri", "30"]
