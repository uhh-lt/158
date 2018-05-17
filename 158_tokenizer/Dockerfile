FROM python:3

# FIXME: Java needed

ENV FASTTEXT 0.1.0

EXPOSE 5000

WORKDIR /usr/src/app

RUN \
apt-get update && \
apt-get install --no-install-recommends -y -o Dpkg::Options::="--force-confold" build-essential libicu-dev mecab libmecab-dev mecab-ipadic-utf8 && \
apt-get clean && \
rm -rf /var/lib/apt/lists/* && \
mkdir language_identification && \
wget https://github.com/facebookresearch/fastText/archive/v$FASTTEXT.tar.gz && \
tar zxf v$FASTTEXT.tar.gz && \
(cd fastText-$FASTTEXT && make && mv -fv fasttext /usr/src/app/language_identification/fasttext) && \
rm -rf fastText-$FASTTEXT v$FASTTEXT.tar.gz

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

USER nobody

CMD ["uwsgi", "--http", "0.0.0.0:5000", "--master", "--module", "tokenizer_jsonrpc:app", "--processes", "4", "--threads", "1", "--harakiri", "30"]
