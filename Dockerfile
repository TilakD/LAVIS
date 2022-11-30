FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
        git \
        make build-essential g++ gcc libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
        libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev  \
        ffmpeg libsm6 libxext6 cmake libgl1-mesa-glx libglib2.0-0\
                && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
  apt-get install -y curl && \
  apt-get install -y openssl && \
  apt-get install -y libssl-dev && \
  apt-get install -y ca-certificates

RUN apt-get update && apt-get install -y python3.8 python3-pip

RUN pip3 install pyopenssl

RUN git clone https://github.com/salesforce/LAVIS.git
WORKDIR LAVIS
RUN pip3 install .
RUN pip3 uninstall -y torch
RUN pip3 uninstall -y torch
RUN pip3 install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install jupyter
