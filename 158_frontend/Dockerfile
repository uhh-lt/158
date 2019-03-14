FROM python:3

EXPOSE 5000

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN conda install -y -c conda-forge uwsgi libiconv&& \
sed -rn '/(uwsgi)/!p' -i requirements.txt && \
pip install --no-cache-dir -r requirements.txt && \
conda clean -a

COPY . .

RUN ./frontend_assets.py

USER nobody

CMD ["uwsgi", "--http", "0.0.0.0:5000", "--master", "--module", "frontend:app", "--processes", "4", "--threads", "1", "--harakiri", "30"]
