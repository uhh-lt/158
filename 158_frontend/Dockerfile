FROM python:3

EXPOSE 5000

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

USER nobody

CMD ["uwsgi", "--http", "0.0.0.0:5000", "--master", "--module", "frontend:app", "--processes", "4", "--threads", "1", "--harakiri", "30"]
