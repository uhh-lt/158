FROM python:3

WORKDIR /usr/src/app

COPY . /usr/src/app

RUN \
apt-get update && \
apt-get install --no-install-recommends -y -o Dpkg::Options::="--force-confold" build-essential libicu-dev mecab libmecab-dev mecab-ipadic-utf8 && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

CMD /usr/src/app/tokenizer_server.py
