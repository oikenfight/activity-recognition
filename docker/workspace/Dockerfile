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

#
# chainerMN
# （これ見た。https://qiita.com/minhal/items/0bd75f2a80fe8af473af）
#

# TODO: ほんとは然るべき場所に追加したいけど、キャッシュが取り消される時間も惜しいので。
RUN apt install -y --allow-change-held-packages \
    libnccl2 \
    libnccl-dev \
    libopenmpi-dev

RUN wget https://download.open-mpi.org/release/open-mpi/v3.0/openmpi-3.0.3.tar.gz
RUN tar -zxvf openmpi-3.0.3.tar.gz
RUN cd ./openmpi-3.0.3 && \
    ./configure --with-cuda --prefix=$HOME/local/openmpi && \
    make -j4 && \
    make install

ENV LD_LIBRARY_PATH=$HOME/local/openmpi/lib:${LD_LIBRARY_PATH}
ENV PATH=$HOME/local/openmpi/bin:${PATH}
ENV NCCL_ROOT=/usr/local/nccl
ENV CPATH=$NCCL_ROOT/include:$CPATH
ENV LD_LIBRARY_PATH=$NCCL_ROOT/lib/:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=$NCCL_ROOT/lib/:$LIBRARY_PATH

RUN pip install --upgrade pip
RUN pip install \
    cython \
    chainermn

RUN nvidia-smi

RUN mkdir workspace
