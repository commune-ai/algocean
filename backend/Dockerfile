# Use nvidia/cuda image
FROM nvidia/cuda:11.2.1-base-ubuntu20.04


ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

RUN mkdir app
WORKDIR app
RUN apt update
RUN apt install -y git



WORKDIR /app

# Install dependencies:
COPY requirements.txt .
# Install dependencies:
COPY environment.yaml .

RUN conda env create -f ./environment.yaml
RUN echo "source activate algocean" > ~/.bashrc
ENV PATH /opt/conda/envs/algocean/bin:$PATH
RUN /bin/bash -c "source activate algocean"
RUN apt install -y gcc
RUN /bin/bash -c "pip install -r requirements.txt"


ADD ./ocean.py /app/ocean.py
RUN /bin/bash -c "pip install -e ./ocean.py"
ADD ./ipfspy /app/ipfspy
RUN /bin/bash -c "pip install -e ./ipfspy"

ADD ./ipfsspec /app/ipfsspec
RUN /bin/bash -c "pip install -e ./ipfsspec"
RUN /bin/bash -c "pip install hub"

RUN apt-get install curl -y
# RUN curl -sSL https://get.docker.com/ | sh
RUN apt-get update && \
    apt-get -qy full-upgrade && \
    apt-get install -qy curl && \
    apt-get install -qy curl && \
    curl -sSL https://get.docker.com/ | sh
# RUN conda activate algocean
# add BITTENSOR commune fork

ENV NODE_VERSION=16.17.1
RUN apt install -y curl
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
ENV NVM_DIR=/root/.nvm
RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm use v${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm alias default v${NODE_VERSION}
ENV PATH="/root/.nvm/versions/node/v${NODE_VERSION}/bin/:${PATH}"
RUN node --version
RUN npm --version

# fix npm - not the latest version installed by apt-get

# RUN cp hardhat.config.barge.js hardhat.config.js
# ENV NETWORK=barge
# ENV NETWORK_RPC_URL=127.0.0.1:8545
# RUN npx hardhat compile
# ENTRYPOINT ["/ocean-contracts/scripts/deploy_docker.sh"

RUN npm install --save-dev hardhat
RUN npm install --save-dev @nomicfoundation/hardhat-toolbox

COPY hardhat.config.js .
RUN npx hardhat
RUN npm install @openzeppelin/contracts
# RUN npx hardhat compile


