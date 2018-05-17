FROM python:3

# FIXME: Java needed

RUN \
apt-get update && \
apt-get install --no-install-recommends -y -o Dpkg::Options::="--force-confold" build-essential libicu-dev mecab libmecab-dev mecab-ipadic-utf8 && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY . /usr/src/app

RUN pip install -r requirements.txt

CMD ["uwsgi", "--http", "0.0.0.0:5000", "--master", "--module", "tokenizer_jsonrpc:application", "--processes", "4", "--threads", "1", "--harakiri", "30"]
