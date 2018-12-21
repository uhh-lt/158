FROM python:3

# FIXME: Java needed

ENV FASTTEXT=0.1.0 EUROPARL=c29b00df9949fc23ab70e8b5d1542ec61ea553af

EXPOSE 5001

WORKDIR /usr/src/app

RUN \
apt-get update && \
apt-get install --no-install-recommends -y -o Dpkg::Options::="--force-confold" build-essential libicu-dev mecab libmecab-dev mecab-ipadic-utf8 swig && \
apt-get clean && \
rm -rf /var/lib/apt/lists/* && \
mkdir language_identification && \
wget https://github.com/facebookresearch/fastText/archive/v$FASTTEXT.tar.gz && \
tar zxf v$FASTTEXT.tar.gz && \
(cd fastText-$FASTTEXT && make && mv -fv fasttext /usr/src/app/language_identification/fasttext) && \
rm -rf fastText-$FASTTEXT v$FASTTEXT.tar.gz && \
wget https://github.com/akivajp/europarl-extractor/archive/$EUROPARL.tar.gz && \
tar zxf $EUROPARL.tar.gz && \
mv europarl-extractor-$EUROPARL/tools Europarl && \
rm -rf europarl-extractor-$EUROPARL $EUROPARL.tar.gz

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

USER nobody

CMD ["python3", "tokenizer_jsonrpc.py"]
