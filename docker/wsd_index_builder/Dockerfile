#FROM python:3
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get clean && apt-get update

RUN apt-get install -yqq curl
RUN curl -sL https://deb.nodesource.com/setup_8.x | bash

RUN apt-get install -y -yqq build-essential libbz2-dev libreadline-dev \
                         libsqlite3-dev libpng-dev libfreetype6-dev git \
                         cmake wget gfortran libatlas-base-dev  \
                         libatlas3-base libhdf5-dev libxml2-dev libxslt-dev \
                         zlib1g-dev pkg-config graphviz liblapacke-dev 

RUN apt-get install -y liblzma-dev

RUN apt-get install -y libssl-dev

RUN apt-get install locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
        dpkg-reconfigure --frontend=noninteractive locales
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

ENV PYENV_ROOT /opt/.pyenv
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
ENV PATH /opt/.pyenv/shims:/opt/.pyenv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN pyenv install 3.6.7
RUN pyenv global 3.6.7

RUN pip  install -U pip
RUN python -m pip install -U cython

WORKDIR /usr/src/app

COPY graph_vector/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install faiss-gpu
RUN pip install matplotlib
#RUN pip install --no-cache-dir -r ./graph_vector/requirements.txt

COPY . .

USER nobody

CMD ["python3", "/usr/src/app/graph_vector/induction.py"]
