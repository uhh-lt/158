# 158

This repository will contain all code for a demo of word sense induction and disambiguation for 158 languages based on the pretrained fastText word embeddings. Please commit all the code related to this project here, including small datasets up to a few megabytes.

## Running

You need Docker and Docker Compose to run the microservices. After cloning the repository, run `docker-compose build` to build the container images.

### Tokenization Service

Running `docker-compose up tokenizer` runs the tokenization service on the port `5001`. The service exposes the following JSON-RPC API:

* `tokenize(text) # => {'language': 'language', tokens: ['Token', '...']}`

### Frontend

Running `docker-compose up frontend` runs the front-end on the port `5000`.
