FROM python:3

# FIXME: Java needed
# Install OpenJDK-8
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

##########

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
