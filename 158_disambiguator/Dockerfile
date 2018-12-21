FROM python:3

EXPOSE 5002

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

USER nobody

CMD ["python3", "disambiguator_jsonrpc.py"]
