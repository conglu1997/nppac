FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04
MAINTAINER Cong Lu

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

ENV CUDNN_VERSION 7.6.5.32
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --allow-change-held-packages --no-install-recommends \
    libcudnn7=$CUDNN_VERSION-1+cuda9.2 \
    libcudnn7-dev=$CUDNN_VERSION-1+cuda9.2 \
    && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

# Ubuntu Packages
RUN apt-get update && apt-get install -y --allow-unauthenticated --no-install-recommends \
         vim build-essential cmake git wget curl ca-certificates \
         libjpeg-dev libpng-dev libgl1-mesa-glx zlib1g-dev \
         libopenmpi-dev python3-dev zlib1g-dev \
         libsm6 libxrender1 libfontconfig1 && \
         rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get update -y --allow-unauthenticated && \
    apt-get install -y --allow-unauthenticated \
    software-properties-common \
    apt-utils \
    nano \
    vim \
    man \
    build-essential \
    wget \
    sudo \
    git \
    mercurial \
    subversion && \
    rm -rf /var/lib/apt/lists/* \
    nvidia-profiler

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

RUN apt-get update && apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENV PATH /opt/conda/bin:$PATH
RUN apt-get update --fix-missing && sudo apt-get install -y libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev libglfw3-dev

# pip packages
RUN pip install pandas hashfs pydevd remote_pdb rpdb matplotlib visdom
RUN pip install sacred GitPython pymongo tinydb tinydb-serialization tensorflow pptree progressbar2 ipdb namedlist pyyaml cython click gtimer snakeviz
RUN pip install -e git+https://github.com/openai/baselines.git#egg=baselines

# mujoco
RUN apt-get install -y libosmesa6-dev libglew1.5-dev unzip
RUN apt-get install -y libglfw3-dev
RUN pip install lockfile glfw imageio
RUN conda install patchelf
ENV MUJOCO_PY_MJKEY_PATH /.mujoco/mjkey.txt
ENV MUJOCO_PY_MUJOCO_PATH /.mujoco/mujoco200
WORKDIR /
RUN mkdir -p /.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /.mujoco \
    && rm mujoco.zip
COPY ./mjkey.txt /.mujoco/mjkey.txt
RUN mv /.mujoco/mujoco200_linux /.mujoco/mujoco200
ENV LD_LIBRARY_PATH /.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
RUN pip install mujoco-py seaborn

# Variable packages
RUN pip install torch==1.7.1+cu92 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install gpytorch==1.0.1
RUN pip install gym==0.12.1 gym[mujoco]==0.12.1 
RUN pip install numpy==1.20.1
RUN pip install -e git+https://github.com/aravindr93/mjrl.git#egg=mjrl

RUN git clone --recursive https://github.com/conglu1997/mj_envs.git
WORKDIR /mj_envs
RUN git submodule update --remote
RUN pip install -e .

RUN pip install dill

WORKDIR /rlkit
