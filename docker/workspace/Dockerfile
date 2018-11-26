FROM nvidia/cuda:9.1-cudnn7-devel

MAINTAINER Yuta Oikawa

RUN apt-get upgrade -y && apt-get update -y
RUN apt-get install -y \
    wget \
    curl \
    git \
    zsh \
    vim \
    tree \
    gcc \
    openssl \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    language-pack-ja-base \
    language-pack-ja \
    ibus-mozc \
    build-essential \
    python3-dev \
    python3-pip \
    zip \
    unzip \
    python3-tk

# ffmpeg
RUN apt-get install apt-file -y
RUN apt-get update
RUN apt-file update
RUN apt-get install software-properties-common -y
RUN add-apt-repository -y ppa:jonathonf/ffmpeg-3
RUN apt-get -y install ffmpeg

# language
RUN update-locale LANG=ja_JP.UTF-8 LANGUAGE=ja_JP:ja
ENV LANG ja_JP.UTF-8
ENV LC_ALL ja_JP.UTF-8
ENV LC_CTYPE ja_JP.UTF-8

# python3, pip3 を python, pip にシンボルリンク
RUN ln -s /usr/bin/python3 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

RUN pip install --upgrade pip
RUN pip install \
    setuptools \
    cupy-cuda91 \
    chainer \
    chainercv \
    scipy \
    pandas \
    scikit-learn \
    pillow \
    h5py \
    matplotlib \
    memory_profiler \
    --no-cache-dir

RUN nvidia-smi

RUN mkdir workspace
