##
## Copyright 2021 Ocean Protocol Foundation
## SPDX-License-Identifier: Apache-2.0
##
FROM ubuntu:20.04

ARG VERSION

RUN apt-get update && \
   apt-get install --no-install-recommends -y \
   gcc \
   python3.8 \
   python3-pip \
   python3.8-dev \
   gettext-base

RUN python3.8 -m pip install --upgrade pip
WORKDIR /app

# Install dependencies:
COPY requirements.txt .
RUN python3.8 -m pip install -r requirements.txt

# add BITTENSOR commune fork
ADD ./ocean.py /app/ocean.py
RUN python3.8 -m pip install -e ./ocean.py

# add ipfspy commune fork
ADD ./ipfspy /app/ipfspy
RUN python3.8 -m pip install -e ./ipfspy
# add ipfsspec commune fork
ADD ./ipfsspec /app/ipfsspec
RUN python3.8 -m pip install -e ./ipfsspec
RUN python3.8 -m pip install jupyterlab
RUN alias python=python3.8