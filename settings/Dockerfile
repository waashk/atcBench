# Docker file for textbench experiments
FROM nvcr.io/nvidia/pytorch:23.05-py3 as base
#LABEL mantainer "spiderman"

ADD https://cmake.org/files/v3.7/cmake-3.7.2-Linux-x86_64.sh /cmake-3.7.2-Linux-x86_64.sh
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    curl \
    enchant-2 \
    llvm \
    git \
    cmake \
    neovim \
    vim \
    python3.10-venv \
    aspell-* \
    hunspell-ko \
    hunspell-ar \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/* && \
    export HOME=/home/textbench

WORKDIR /home/textbench

SHELL ["/usr/bin/bash", "-c"]

# Set-up pyenv
RUN export HOME=/home/textbench && \
    pip install --upgrade pip wheel setuptools
COPY requirements.txt .
RUN pip install -r requirements.txt