# 158

This repository will contain all code for a demo of word sense induction and disambiguation for 158 languages based on the pretrained fastText word embeddings. Please commit all the code related to this project here, including small datasets up to a few megabytes.

The **inventories** for all languages are available for download here: [http://ltdata1.informatik.uni-hamburg.de/158/](http://ltdata1.informatik.uni-hamburg.de/158/).

## Running

You need Docker and Docker Compose to run the microservices. After cloning the repository, run `docker-compose build` to build the container images.

In case you have [SELinux enabled](https://stopdisablingselinux.com/), run the following command in advance: `chcon -t svirt_sandbox_file_t 158-docker.ini`.

### Tokenization Service

The entry point is `158_tokenizer/tokenizer_jsonrpc.py`. Running `docker-compose up tokenizer` starts the tokenization service on the port `5001`. The service exposes the following JSON-RPC API:

* `tokenize(text) # => {'language': 'language', tokens: ['Token', '...']}`

### Disambiguation Service

The entry point is `158_disambiguator/disambiguator_jsonrpc.py`. Running `docker-compose up disambiguator` starts the tokenization service on the port `5002`. The service exposes the following JSON-RPC API:

* `disambiguate(language, tokens) # => ?`

### Frontend

The entry point is `158_frontend/frontend.py`. Running `docker-compose up frontend` starts the HTTP-based front-end on the port `5000`. In order to balance the workload, the frontend sends each processing request to a random host listed in the configuration file (see below).

### Everything Together

Just run `docker-compose up`; the provided example [docker-compose.yml](docker-compose.yml) is self-sufficient (as soon the files are placed correctly).

## Configuration

Every microservice reads the `158.ini` configuration file, see example [158-docker.ini](158-docker.ini). It is a good idea to share the same read-only configuration file between all the containers.

### Section `[services]`

* `tokenizer`: comma-separated list of hostnames and ports with the tokenizer servers

### Section `[disambiguator]`

* `en/ru/...`: comma-separated list of disambiguators for the specified language, one list per language

### Section `[models]`

* `en/ru/...`: path to the sense inventory corresponding to the specified language

### Section `[tokenizer]`

* `icu_langs`: list of exotic languages for which [ICU](https://github.com/ovalhub/pyicu) tokenization is used
