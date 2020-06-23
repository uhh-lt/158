# 158

This repository contains all code for the [demo of word sense induction and disambiguation for 158 languages](http://ltdemos.informatik.uni-hamburg.de/uwsd158/) based on the pretrained fastText word embeddings.

It accompanies the paper 

[Word Sense Disambiguation for 158 Languages using Word Embeddings Only](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.728.pdf) (2020) by Varvara Logacheva, Denis Teslenko, Artem Shelmanov, Steffen Remus, Dmitry Ustalov, Andrey Kutuzov, Ekaterina Artemova, Chris Biemann, Simone Paolo Ponzetto and Alexander Panchenko. Proceedings of The 12th Language Resources and Evaluation Conference (LREC). [bibtex entry](http://www.lrec-conf.org/proceedings/lrec2020/bib/2020.lrec-1.728.bib)   

The **inventories** for all languages are available for download here: [http://ltdata1.informatik.uni-hamburg.de/158/](http://ltdata1.informatik.uni-hamburg.de/158/).

Please commit all the code related to this project here, including small datasets up to a few megabytes.

## Running

You need Docker and Docker Compose to run the microservices. After cloning the repository, run `docker-compose build` to build the container images.

In case you have [SELinux enabled](https://stopdisablingselinux.com/), run the following command in advance: `chcon -t svirt_sandbox_file_t 158-docker.ini`.

### Tokenization Service

The entry point is `158_tokenizer/tokenizer_json.py`. Running `docker-compose up tokenizer` starts the tokenization service on the port `10151`. The service exposes the following JSON-RPC API:

* `tokenize(text) # => {'language': 'language', tokens: ['Token', '...']}`

#### Tokenization Dependencies

- For Chinese tokenizer to work, the files `dict-chris6.ser.gz` and `pku.gz` must be placed in the `stanford_segmenter/data` directory.
They can be downloaded at https://nlp.stanford.edu/software/stanford-segmenter-2018-02-27.zip
- For Vietnamese tokenizer to work, the model files must be placed in the `UETSegmenter/models` directory.
They can be downloaded at https://github.com/phongnt570/UETsegmenter/tree/master/models
- For Japanese tokenizer to work, MeCab must be installed system-wide:
`apt install mecab libmecab-dev mecab-ipadic-utf8`. Then, the `mecab-python3` module for Python must be installed.
- See also the `requirements.txt` for the required Python modules.

### Disambiguation Service

The entry point is `158_disambiguator/disambiguator_server.py`. Running `docker-compose up disambiguator` starts the tokenization service on the port `10152`. The service exposes the following JSON-RPC API:

* `disambiguate(language, tokens) # => ?`

#### Disambiguation Dependencies

Before running server, you need to put fastText models in /models/fasttext_models/{lang}/ and inventories in /models/inventories/{lang}/ (separate folders for each language) if you want to keep them in RAM, otherwise use PostgreSQL Service. You can find useful scripts in /models/ folder to load fastText vectors (load_fasttext.py), to create your own inventory (graph_induction.py) and to upload data to a postgresql database if needed (fasttext_to_psql.py, inventory_to_psql.py).

### PostgreSQL Service

Running `docker-compose up database` starts the tokenization service on the port `10153`. The service is a postgreSQL server. It is used to store fastText vectors and inventories if you don't want to keep them in RAM.

### Frontend

The entry point is `158_frontend/frontend.py`. Running `docker-compose up frontend` starts the HTTP-based front-end on the port `10150`. In order to balance the workload, the frontend sends each processing request to a random host listed in the configuration file (see below).

### Everything Together

Just run `docker-compose up`; the provided example [docker-compose.yml](docker-compose.yml) is self-sufficient (as soon the files are placed correctly).

## Configuration

Every microservice reads the `158.ini` configuration file, see example [158-docker.ini](158-docker.ini). It is a good idea to share the same read-only configuration file between all the containers.

### Section `[services]`

* `tokenizer`: comma-separated list of hostnames and ports with the tokenizer servers
* `disambiguator`: comma-separated list of hostnames and ports with the disambiguator servers

### Section `[tokenizer]`

* `icu_langs`: list of exotic languages for which [ICU](https://github.com/ovalhub/pyicu) tokenization is used

### Section `[disambiguator]`

* `sql_langs`: comma-separated list of languages that are stored in postgresql server
* `top_langs`: comma-separated list of languages that are stored in RAM
* `inventories_fpath`: path for the inventories files
* `inventory_file_format`: format of the inventory filenames
* `dict_size`: limit of the fastText vocabulary stored in RAM
* `inventory_top`: how many neighbors were used to build inventory

### Section `[postgress]`

* `user`: username for the postgress server
* `password`: password for the postgress server (we don't hide this parameter as the data is not a secret)
* `vectors_db`: name of the database with fastText vectors
* `inventories_db`: name of the database with inventories
* `host`: postgress server host
* `port`: postgress server port
